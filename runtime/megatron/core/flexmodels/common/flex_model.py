# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.distributed

from megatron.core.transformer.module import MegatronModule
from functools import reduce
import operator
import numpy as np
import os
from megatron.core import mpu
from megatron.core.flexmodels.common.flex_model_config import FlexModelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.pipeline_parallel.schedules import (
    reset_checkpointed_activations_memory_buffer,
)
from megatron.core.tensor_parallel.random import checkpoint

NUM_BATCHES = 0
DEBUG_OUTPUT = os.environ.get("DEBUG_OUTPUT", "0") == "1"

# def print_tensors(op_name, output_tensors):
#     args = get_args()
#     if mpu.get_tensor_model_parallel_rank() == 0 and (NUM_BATCHES * args.micro_batch_size)% args.global_batch_size == 0:
#         string = f"rank {torch.distributed.get_rank()} | micro batch #{NUM_BATCHES} | {op_name} output"
#         for key in output_tensors:
#             string += f"\n[{key}] (shape: {list(output_tensors[key].size())}) = {output_tensors[key]}"
#         with open(f"{args.log_path}{args.log_name}_debug_output_rank{torch.distributed.get_rank()}.log", "a+") as f:
#             f.write(string+"\n")


def print_ops_info(ops, recompute_ops):
    ck_ops = ""
    all_ops = ""
    for i in range(len(ops)):
        all_ops += '"' + ops[i].op_name + '",'
        if recompute_ops[i] == True:
            ck_ops += ops[i].op_name + " "
    print(f"[rank {torch.distributed.get_rank()} all ops] {all_ops}")
    print(f"[rank {torch.distributed.get_rank()} recompute ops] {ck_ops}")


def get_prev_stage_index(pipeline_rank, virtual_pipeline_rank):
    assert pipeline_rank > 0 or virtual_pipeline_rank > 0
    prev_pipeline_rank = pipeline_rank - 1
    prev_virtual_pipeline_rank = virtual_pipeline_rank
    if prev_pipeline_rank < 0:
        prev_pipeline_rank = mpu.get_pipeline_model_parallel_world_size() - 1
        prev_virtual_pipeline_rank -= 1
    op_start_index_prev_stage = mpu.get_op_start_index(
        prev_pipeline_rank, prev_virtual_pipeline_rank
    )
    op_end_index_prev_stage = mpu.get_op_end_index(
        prev_pipeline_rank, prev_virtual_pipeline_rank
    )
    return op_start_index_prev_stage, op_end_index_prev_stage


def get_next_stage_index(pipeline_rank, virtual_pipeline_rank):
    assert (
        pipeline_rank < mpu.get_pipeline_model_parallel_world_size() - 1
        or virtual_pipeline_rank
        < mpu.get_virtual_pipeline_model_parallel_world_size() - 1
    )
    next_pipeline_rank = pipeline_rank + 1
    next_virtual_pipeline_rank = virtual_pipeline_rank
    if next_pipeline_rank > mpu.get_pipeline_model_parallel_world_size() - 1:
        next_pipeline_rank = 0
        next_virtual_pipeline_rank += 1
    op_start_index_next_stage = mpu.get_op_start_index(
        next_pipeline_rank, next_virtual_pipeline_rank
    )
    op_end_index_next_stage = mpu.get_op_end_index(
        next_pipeline_rank, next_virtual_pipeline_rank
    )
    return op_start_index_next_stage, op_end_index_next_stage


## we don't consider any communication optimization in this place,
## calculate all the shape as if there is no communication optimization,
## leave the P2P optimization in p2p_communication.py
def initialize_comm_info(
    flex_config: FlexModelConfig,
    tensors_info,
    dst_stage,
    src_op_index=0,
    dst_op_index=0,
):

    num_ops = sum(mpu.get_num_ops_list())
    if dst_op_index < 0:
        dst_op_index = num_ops - 1
    elif dst_op_index > num_ops - 1:
        dst_op_index = 0
    tp_size = mpu.get_op_tp_size(src_op_index)
    dp_size = mpu.get_op_dp_size(src_op_index)
    dst_tp_size = mpu.get_op_tp_size(dst_op_index)
    dst_dp_size = mpu.get_op_dp_size(dst_op_index)

    ranks_in_this_stage = mpu.get_ranks_via_pipeline_stage(
        mpu.get_pipeline_model_parallel_rank()
    )
    rank = torch.distributed.get_rank()
    for i in range(len(ranks_in_this_stage)):
        if rank == ranks_in_this_stage[i]:
            tp_id = i % tp_size
            dp_id = i // tp_size

    recv_info = {"size": 0, "tensors": {}}
    send_info = {"tensors": {}}

    for key in sorted(tensors_info):
        if key not in ["input_tensor"]:
            tp_split_dim = tensors_info[key]["tp_split_dim"]
            dp_split_dim = tensors_info[key]["dp_split_dim"]

            num_tp_chunks = 1
            num_dp_chunks = 1

            recv_info["tensors"][key] = {
                "tp_split_dim": tp_split_dim,
                "num_tp_chunks": num_tp_chunks,
                "dp_split_dim": dp_split_dim,
                "num_dp_chunks": num_dp_chunks,
            }
            send_info["tensors"][key] = {
                "tp_split_dim": tp_split_dim,
                "num_tp_chunks": num_tp_chunks,
                "tp_chunks_index": [0],
                "dp_split_dim": dp_split_dim,
                "num_dp_chunks": num_dp_chunks,
                "dp_chunks_index": [0],
            }

            shape = tensors_info[key]["shape"]

            if tp_split_dim != -1:
                shape[tp_split_dim] //= tp_size
            if dp_split_dim != -1:
                shape[dp_split_dim] //= dp_size

            if dst_tp_size > tp_size:
                ratio = dst_tp_size // tp_size
                num_tp_chunks = ratio
                if tp_split_dim != -1:
                    shape[tp_split_dim] //= ratio
                elif (
                    tp_split_dim == -1
                    and not flex_config.scatter_gather_tensors_in_pipeline
                ):
                    recv_info["tensors"][key]["tp_split_dim"] = 0
                    shape[0] //= ratio
                recv_info["tensors"][key]["num_tp_chunks"] = num_tp_chunks

                send_info["tensors"][key]["num_tp_chunks"] = num_tp_chunks
                send_info["tensors"][key]["tp_chunks_index"] = range(num_tp_chunks)

            if dst_tp_size < tp_size:
                if flex_config.scatter_gather_tensors_in_pipeline:
                    send_info["tensors"][key]["tp_chunks_index"] = range(num_tp_chunks)
                else:
                    if tp_split_dim != -1:
                        send_info["tensors"][key]["tp_chunks_index"] = range(
                            num_tp_chunks
                        )
                    else:
                        ratio = tp_size // dst_tp_size
                        num_tp_chunks = ratio
                        send_info["tensors"][key]["tp_split_dim"] = 0
                        send_info["tensors"][key]["num_tp_chunks"] = num_tp_chunks
                        send_info["tensors"][key]["tp_chunks_index"] = [
                            tp_id % num_tp_chunks
                        ]

            if dst_dp_size > dp_size:
                ratio = dst_dp_size // dp_size
                num_dp_chunks = ratio

                if dp_split_dim != -1:
                    recv_info["tensors"][key]["dp_split_dim"] = dp_split_dim
                    shape[dp_split_dim] //= ratio
                else:
                    recv_info["tensors"][key]["dp_split_dim"] = 0
                    shape[0] //= ratio
                recv_info["tensors"][key]["num_dp_chunks"] = num_dp_chunks

                send_info["tensors"][key]["dp_split_dim"] = dp_split_dim
                send_info["tensors"][key]["num_dp_chunks"] = num_dp_chunks
                send_info["tensors"][key]["dp_chunks_index"] = range(num_dp_chunks)

            if dst_dp_size < dp_size:
                recv_info["tensors"][key]["dp_split_dim"] = dp_split_dim
                recv_info["tensors"][key]["num_dp_chunks"] = num_dp_chunks

                if dp_split_dim != -1:
                    send_info["tensors"][key]["dp_split_dim"] = dp_split_dim
                    send_info["tensors"][key]["num_dp_chunks"] = num_dp_chunks
                    send_info["tensors"][key]["dp_chunks_index"] = range(num_dp_chunks)
                else:
                    ratio = dp_size // dst_dp_size
                    num_dp_chunks = ratio
                    send_info["tensors"][key]["dp_split_dim"] = 0
                    send_info["tensors"][key]["num_dp_chunks"] = num_dp_chunks
                    send_info["tensors"][key]["dp_chunks_index"] = [
                        dp_id % num_dp_chunks
                    ]

            recv_info["tensors"][key]["shape"] = shape
            recv_info["size"] += reduce(operator.mul, shape, 1)

    return send_info, recv_info


def initialize_communication(flex_config: FlexModelConfig, model_chunk_op_list):
    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    virtual_pipeline_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    op_start_index = mpu.get_op_start_index(pipeline_rank, virtual_pipeline_rank)
    op_end_index = mpu.get_op_end_index(pipeline_rank, virtual_pipeline_rank)

    input_tensors_info = model_chunk_op_list[0].input_tensors_info
    output_tensors_info = model_chunk_op_list[
        op_end_index - op_start_index - 1
    ].output_tensors_info

    input_extra_tensors_dict = {}
    if pipeline_rank > 0 or virtual_pipeline_rank > 0:
        op_start_index_prev_stage, op_end_index_prev_stage = get_prev_stage_index(
            pipeline_rank, virtual_pipeline_rank
        )
        for op_index in range(op_start_index, op_end_index):
            op = model_chunk_op_list[op_index - op_start_index]
            for key in sorted(op.input_extra_tensors_info):
                op_index_recv_from = op_index + op.input_extra_tensors_info[key]["recv_from"]
                
                if op_index_recv_from < op_start_index:
                    assert (
                        op_index_recv_from >= op_start_index_prev_stage
                        and op_index_recv_from < op_end_index_prev_stage
                    ), f"op_index_recv_from = {op_index_recv_from}, op_start_index_prev_stage = {op_start_index_prev_stage}, op_end_index_prev_stage = {op_end_index_prev_stage}"
                    input_extra_tensors_dict[key] = {
                        "info": {key: op.input_extra_tensors_info[key]},
                        "src_op": op_index,
                        "dst_op": op_index_recv_from,
                    }

    output_extra_tensors_dict = {}
    if (
        pipeline_rank < mpu.get_pipeline_model_parallel_world_size() - 1
        or virtual_pipeline_rank
        < mpu.get_virtual_pipeline_model_parallel_world_size() - 1
    ):
        op_start_index_next_stage, op_end_index_next_stage = get_next_stage_index(
            pipeline_rank, virtual_pipeline_rank
        )
        for op_index in range(op_start_index, op_end_index):
            op = model_chunk_op_list[op_index - op_start_index]
            for key in sorted(op.output_extra_tensors_info):
                op_index_send_to = (
                    op_index + op.output_extra_tensors_info[key]["send_to"]
                )
                if op_index_send_to >= op_end_index:
                    assert (
                        op_index_send_to >= op_start_index_next_stage
                        and op_index_send_to < op_end_index_next_stage
                    ), f"rank {torch.distributed.get_rank()}, virtual {mpu.get_virtual_pipeline_model_parallel_rank()}, op.op_name = {op.op_name} op_index_send_to = {op_index_send_to}, op_start_index_next_stage = {op_start_index_next_stage}, op_end_index_next_stage = {op_end_index_next_stage}"
                    op.output_extra_tensors_info[key]["cross_stage"] = True
                    output_extra_tensors_dict[key] = {
                        "info": {key: op.output_extra_tensors_info[key]},
                        "src_op": op_index,
                        "dst_op": op_index_send_to,
                    }
                else:
                    op.output_extra_tensors_info[key]["cross_stage"] = False
    else:
        for op_index in range(op_start_index, op_end_index):
            op = model_chunk_op_list[op_index - op_start_index]
            for key in sorted(op.output_extra_tensors_info):
                op.output_extra_tensors_info[key]["cross_stage"] = False

    prev_stage = mpu.get_prev_pipeline_model_parallel_rank()
    next_stage = mpu.get_next_pipeline_model_parallel_rank()

    bwd_send_info, fwd_recv_info = initialize_comm_info(
        flex_config,
        input_tensors_info,
        prev_stage,
        src_op_index=model_chunk_op_list[0].op_index,
        dst_op_index=model_chunk_op_list[0].op_index - 1,
    )
    fwd_send_info, bwd_recv_info = initialize_comm_info(
        flex_config,
        output_tensors_info,
        next_stage,
        src_op_index=model_chunk_op_list[-1].op_index,
        dst_op_index=model_chunk_op_list[-1].op_index + 1,
    )

    for key in input_extra_tensors_dict:
        _bwd_send_info, _fwd_recv_info = initialize_comm_info(
            flex_config,
            input_extra_tensors_dict[key]["info"],
            prev_stage,
            src_op_index=model_chunk_op_list[0].op_index,
            dst_op_index=model_chunk_op_list[0].op_index - 1,
        )
        fwd_recv_info["size"] += _fwd_recv_info["size"]
        fwd_recv_info["tensors"][key] = _fwd_recv_info["tensors"][key]
        bwd_send_info["tensors"][key] = _bwd_send_info["tensors"][key]

    for key in output_extra_tensors_dict:
        _fwd_send_info, _bwd_recv_info = initialize_comm_info(
            flex_config,
            output_extra_tensors_dict[key]["info"],
            next_stage,
            src_op_index=model_chunk_op_list[-1].op_index,
            dst_op_index=model_chunk_op_list[-1].op_index + 1,
        )
        bwd_recv_info["size"] += _bwd_recv_info["size"]
        bwd_recv_info["tensors"][key] = _bwd_recv_info["tensors"][key]
        fwd_send_info["tensors"][key] = _fwd_send_info["tensors"][key]

    for key in fwd_recv_info["tensors"]:
        if key in input_extra_tensors_dict:
            fwd_recv_info["tensors"][key]["extra_tensor"] = True
        else:
            fwd_recv_info["tensors"][key]["extra_tensor"] = False
    for key in bwd_recv_info["tensors"]:
        if key in output_extra_tensors_dict:
            bwd_recv_info["tensors"][key]["extra_tensor"] = True
        else:
            bwd_recv_info["tensors"][key]["extra_tensor"] = False

    mpu.set_comm_info(bwd_send_info, fwd_recv_info, fwd_send_info, bwd_recv_info)


def pre_forward_hook(op, input):
    pass


def post_forward_hook(op, input, output):
    if DEBUG_OUTPUT:
        pass
        # print_tensors(op.name, output)


class FlexPipeModel(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        flex_config: FlexModelConfig,
        full_model_op_list,
        pre_process=True,
        post_process=True,
    ):
        super(FlexPipeModel, self).__init__(config)

        self.saved_tensors = {}
        self.pre_process = pre_process
        self.post_process = post_process

        self.input_tensor = None
        rank_in_pipeline = mpu.get_pipeline_model_parallel_rank()
        self.recompute_ops = flex_config.recompute_ops[rank_in_pipeline]
        self.flex_recompute_activations = flex_config.flex_recompute_activations[
            rank_in_pipeline
        ]

        full_model_op_list[0].prev_name = None
        full_model_op_list[-1].is_last_op = True
        self.ops = torch.nn.ModuleList(full_model_op_list)
        self.resharding = flex_config.resharding_stages[rank_in_pipeline]
        assert self.resharding == False, "Not support resharding"
        pre_hook = pre_forward_hook
        post_hook = post_forward_hook
        for op in self.ops:
            op.register_forward_pre_hook(pre_hook)
            op.register_forward_hook(post_hook)

        self.num_ops = len(full_model_op_list)
        initialize_communication(flex_config, full_model_op_list)
        print_ops_info(self.ops, self.recompute_ops)

    def _checkpointed_forward(
        self,
        start,
        end,
        input_tensors,
        input_extra_tensors,
        output_extra_tensors,
        tmp_input_extra_tensors={},
    ):

        # The input_tensors and input_extra_tensors are dicts of tensors
        # First transform the dict into tuple of tensors before calling mpu.checkpoint()
        # Then transfer the tuple back to dict in the custom function
        # Then call the op with dict of tensors
        # return tuple of tensors
        # Then transform tuple back to dict
        # return dict of tensors

        is_start_op = start == 0
        is_end_op = end == self.num_ops

        def custom(
            start,
            end,
            input_tensor_names,
            input_extra_tensor_names,
            output_tensor_names,
            tmp_input_extra_tensors,
            output_extra_tensors,
        ):
            def custom_forward(*inputs):
                x_ = {}
                input_extra_tensors_dict = {}

                for i in range(len(input_tensor_names)):
                    x_[input_tensor_names[i]] = inputs[i]

                len_inputs = len(input_tensor_names)
                for i in range(len(input_extra_tensor_names)):
                    input_extra_tensors_dict[input_extra_tensor_names[i]] = inputs[
                        i + len_inputs
                    ]

                for index in range(start, end):
                    op = self.ops[index]
                    x_ = op(x_, input_extra_tensors_dict, output_extra_tensors)

                output_tensor_list = []
                for key in sorted(x_):
                    output_tensor_names.append(key)
                    output_tensor_list.append(x_[key])

                for key in input_extra_tensors_dict:
                    tmp_input_extra_tensors[key] = input_extra_tensors_dict[key]

                return tuple(output_tensor_list)

            return custom_forward

        def custom_reshard(
            start,
            end,
            input_tensor_names,
            input_extra_tensor_names,
            output_tensor_names,
            tmp_input_extra_tensors,
            output_extra_tensors,
            input_tensors_specs_mats,
            output_tensors_specs_mats,
        ):
            def custom_forward(*inputs):
                x_ = {}
                input_extra_tensors_dict = {}

                x_tensors = {}
                for i in range(len(input_tensor_names)):
                    x_tensors[input_tensor_names[i]] = inputs[i]

                if not is_start_op:
                    x_["tensors"] = x_tensors
                    x_["specs"] = input_tensors_specs_mats["specs"]
                    x_["mats"] = input_tensors_specs_mats["mats"]
                    x_["input_extra_tensor_specs"] = input_tensors_specs_mats[
                        "input_extra_tensor_specs"
                    ]
                    x_["input_extra_tensor_mats"] = input_tensors_specs_mats[
                        "input_extra_tensor_mats"
                    ]
                else:
                    x_ = x_tensors

                len_inputs = len(input_tensor_names)
                for i in range(len(input_extra_tensor_names)):
                    input_extra_tensors_dict[input_extra_tensor_names[i]] = inputs[
                        i + len_inputs
                    ]

                for index in range(start, end):
                    op = self.ops[index]
                    x_ = op(x_, input_extra_tensors_dict, output_extra_tensors)

                output_tensor_list = []
                if not is_end_op:
                    for key in sorted(x_["tensors"]):
                        output_tensor_names.append(key)
                        output_tensor_list.append(x_["tensors"][key])

                    output_tensors_specs_mats["specs"] = x_["specs"]
                    output_tensors_specs_mats["mats"] = x_["mats"]
                    output_tensors_specs_mats["input_extra_tensor_specs"] = x_[
                        "input_extra_tensor_specs"
                    ]
                    output_tensors_specs_mats["input_extra_tensor_mats"] = x_[
                        "input_extra_tensor_mats"
                    ]
                else:
                    for key in sorted(x_):
                        output_tensor_names.append(key)
                        output_tensor_list.append(x_[key])

                for key in input_extra_tensors_dict:
                    tmp_input_extra_tensors[key] = input_extra_tensors_dict[key]

                return tuple(output_tensor_list)

            return custom_forward

        # Make sure memory is freed.
        reset_checkpointed_activations_memory_buffer()

        input_tensor_names = []
        input_extra_tensor_names = []
        output_tensor_names = []

        # dict -> tuple (list)
        input_tensor_list = []
        input_extra_tensor_list = []

        if (
            self.resharding and not is_start_op
        ):  # The first op does not have resharding hook.
            for key in sorted(input_tensors["tensors"]):
                input_tensor_names.append(key)
                input_tensor_list.append(input_tensors["tensors"][key])
        else:
            for key in sorted(input_tensors):
                input_tensor_names.append(key)
                input_tensor_list.append(input_tensors[key])

        for key in sorted(input_extra_tensors):
            input_extra_tensor_names.append(key)
            input_extra_tensor_list.append(input_extra_tensors[key])

        list_inputs = input_tensor_list + input_extra_tensor_list

        if self.resharding:
            output_tensors_specs_mats = {}
            output_tensors = checkpoint(
                custom_reshard(
                    start,
                    end,
                    input_tensor_names,
                    input_extra_tensor_names,
                    output_tensor_names,
                    tmp_input_extra_tensors,
                    output_extra_tensors,
                    input_tensors,
                    output_tensors_specs_mats,
                ),
                self.config.distribute_saved_activations,
                *list_inputs,
            )
        else:
            output_tensors = checkpoint(
                custom(
                    start,
                    end,
                    input_tensor_names,
                    input_extra_tensor_names,
                    output_tensor_names,
                    tmp_input_extra_tensors,
                    output_extra_tensors,
                ),
                self.config.distribute_saved_activations,
                *list_inputs,
            )

        # return output_tensors
        if self.resharding and not is_end_op:
            output_tensors_dict = {}
            output_tensors_dict["tensors"] = {}
            for i in range(len(output_tensors)):
                output_tensors_dict["tensors"][output_tensor_names[i]] = output_tensors[
                    i
                ]
        else:
            output_tensors_dict = {}
            for i in range(len(output_tensors)):
                output_tensors_dict[output_tensor_names[i]] = output_tensors[i]
        return output_tensors_dict

    def get_inputs(self, op_name):
        if len(self.saved_tensors[op_name]) > 0:
            saved_tensors = self.saved_tensors[op_name].pop(0)
        else:
            saved_tensors = None
        return saved_tensors

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, inputs, input_extra_tensors):
        global NUM_BATCHES
        output_extra_tensors = {} 
        if not self.pre_process:
            hidden_states = self.input_tensor
        else:
            hidden_states = inputs

        if self.flex_recompute_activations:
            start_index = 0
            end_index = self.num_ops

            while start_index < end_index:
                if not self.recompute_ops[start_index]:
                    op = self.ops[start_index]
                    hidden_states = op(
                        hidden_states, input_extra_tensors, output_extra_tensors
                    )
                    
                    start_index += 1
                else:
                    checkpoint_end_index = start_index
                    ## NOTE: this is important for recomputation, set the recomputation breaking point.
                    while (
                        checkpoint_end_index < end_index
                        and self.recompute_ops[checkpoint_end_index]
                    ):
                        checkpoint_end_index += 1
                        if checkpoint_end_index < end_index and (
                            self.ops[checkpoint_end_index].op_name
                            in ["dec-self-attention"]
                        ):
                            break

                    tmp_input_extra_tensors = {}
                    hidden_states = self._checkpointed_forward(
                        start_index,
                        checkpoint_end_index,
                        hidden_states,
                        input_extra_tensors,
                        output_extra_tensors,
                        tmp_input_extra_tensors,
                    )
                    
                    for key in tmp_input_extra_tensors:
                        input_extra_tensors[key] = tmp_input_extra_tensors[key]

                    start_index = checkpoint_end_index
        else:
            for index in range(self.num_ops):
                op = self.ops[index]
                hidden_states = op(
                    hidden_states, input_extra_tensors, output_extra_tensors
                )
                
        NUM_BATCHES = NUM_BATCHES + 1
        output = hidden_states

        return output, output_extra_tensors


def get_flex_model(
    config: TransformerConfig,
    flex_config: FlexModelConfig,
    full_model_op_list,
    pre_process=True,
    post_process=True,
):
    language_model = FlexPipeModel(
        config,
        flex_config,
        full_model_op_list,
        pre_process=pre_process,
        post_process=post_process,
    )

    return language_model
