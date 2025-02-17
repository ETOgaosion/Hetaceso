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


def print_ops_info(ops):
    all_ops = ""
    for i in range(len(ops)):
        all_ops += '"' + ops[i].op_name + '",'
    print(f"[rank {torch.distributed.get_rank()} all ops] {all_ops}")


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

def initialize_comm_info2(
    tensors_info,
    dst_stage: int,
    src_op_index: int,
    dst_op_index: int,
):
    '''
    src_op_index: the op in my rank
    dst_op_index: the op in prev/next rank
    '''
    num_ops = sum(mpu.get_num_ops_list())
    if dst_op_index < 0:
        dst_op_index = num_ops - 1
    elif dst_op_index > num_ops - 1:
        dst_op_index = 0

    if src_op_index < dst_op_index:
      send_reshard = mpu.get_p2p_fwd_reshard()
      recv_reshard = mpu.get_p2p_bwd_reshard()
    else:
      send_reshard = mpu.get_p2p_bwd_reshard()
      recv_reshard = mpu.get_p2p_fwd_reshard()
    assert dst_stage == mpu.get_pipeline_stage_via_op_index(dst_op_index), 'dst_stage should be the same'
    dst_ranks: list[int] = mpu.get_ranks_via_pipeline_stage(dst_stage)
    src_stage: int = mpu.get_pipeline_stage_via_op_index(src_op_index)
    src_ranks: list[int] = mpu.get_ranks_via_pipeline_stage(src_stage)
    my_rank: int = torch.distributed.get_rank()
    assert my_rank in src_ranks
    

    recv_info = {"size": 0, "tensors": {}}
    send_info = {"tensors": {}}

    for key in sorted(tensors_info):
        if key in ["input_tensor"]:
            # Theoretically it shouldn't be here
            continue
        tp_split_dim = tensors_info[key]["tp_split_dim"]
        cp_split_dim = tensors_info[key]["cp_split_dim"]
        dp_split_dim = tensors_info[key]["dp_split_dim"]
        assert tp_split_dim == -1, "Not split TP"
        recv_info["tensors"][key] = {
            "tp_split_dim": tp_split_dim,
            "cp_split_dim": cp_split_dim,
            "dp_split_dim": dp_split_dim,
            "shape": tensors_info[key]["shape"],
            "data_slice": recv_reshard[my_rank]["data_slice"],
            "split": [],
        }
        send_info["tensors"][key] = {
            "tp_split_dim": tp_split_dim,
            "cp_split_dim": cp_split_dim,
            "dp_split_dim": dp_split_dim,
            "shape": tensors_info[key]["shape"],
            "split": []
        }

        for dst_rank in dst_ranks:
            print(f'{my_rank} -> {dst_rank} {send_reshard[dst_rank]["split"]}')
            for (send_rank, ds_0, ds_1) in send_reshard[dst_rank]["split"]:
                if send_rank != my_rank:
                    continue
                send_info["tensors"][key]["split"].append(
                    {
                        "data_slices": (ds_0, ds_1),
                        "rank": dst_rank
                    }
                )
        
        for (recv_from_rank, ds_0, ds_1) in recv_reshard[my_rank]["split"]:
            recv_info["tensors"][key]["split"].append(
                {
                    "data_slices": (ds_0, ds_1),
                    "rank": recv_from_rank
                }
            )
        recv_info["size"] += reduce(operator.mul, tensors_info[key]["shape"], 1)
    return send_info, recv_info


## we don't consider any communication optimization in this place,
## calculate all the shape as if there is no communication optimization,
## leave the P2P optimization in p2p_communication.py
def initialize_comm_info(
    tensors_info,
    dst_stage: int,
    src_op_index: int = 0,
    dst_op_index: int = 0,
):

    num_ops = sum(mpu.get_num_ops_list())
    if dst_op_index < 0:
        dst_op_index = num_ops - 1
    elif dst_op_index > num_ops - 1:
        dst_op_index = 0
    tp_size = mpu.get_op_tp_size(src_op_index)
    cp_size = mpu.get_op_cp_size(src_op_index)
    dp_size = mpu.get_op_dp_size(src_op_index)
    dst_tp_size = mpu.get_op_tp_size(dst_op_index)
    dst_cp_size = mpu.get_op_cp_size(dst_op_index)
    dst_dp_size = mpu.get_op_dp_size(dst_op_index)

    ranks_in_this_stage = mpu.get_ranks_via_pipeline_stage(
        mpu.get_pipeline_model_parallel_rank()
    )
    rank = torch.distributed.get_rank()
    for i in range(len(ranks_in_this_stage)):
        if rank == ranks_in_this_stage[i]:
            tp_id = i % tp_size
            cp_id = i // tp_size % cp_size
            dp_id = i // tp_size // cp_size

    recv_info = {"size": 0, "tensors": {}}
    send_info = {"tensors": {}}

    for key in sorted(tensors_info):
        if key not in ["input_tensor"]:
            tp_split_dim = tensors_info[key]["tp_split_dim"]
            cp_split_dim = tensors_info[key]["cp_split_dim"]
            dp_split_dim = tensors_info[key]["dp_split_dim"]

            num_tp_chunks = 1
            num_cp_chunks = 1
            num_dp_chunks = 1

            recv_info["tensors"][key] = {
                "tp_split_dim": tp_split_dim,
                "num_tp_chunks": num_tp_chunks,
                "cp_split_dim": cp_split_dim,
                "num_cp_chunks": num_cp_chunks,
                "dp_split_dim": dp_split_dim,
                "num_dp_chunks": num_dp_chunks,
            }
            send_info["tensors"][key] = {
                "tp_split_dim": tp_split_dim,
                "num_tp_chunks": num_tp_chunks,
                "tp_chunks_index": [0],
                "cp_split_dim": cp_split_dim,
                "num_cp_chunks": num_cp_chunks,
                "cp_chunks_index": [0],
                "dp_split_dim": dp_split_dim,
                "num_dp_chunks": num_dp_chunks,
                "dp_chunks_index": [0],
            }

            shape = tensors_info[key]["shape"]

            if tp_split_dim != -1:
                shape[tp_split_dim] //= tp_size
            if cp_split_dim != -1:
                shape[cp_split_dim] //= cp_size
            if dp_split_dim != -1:
                shape[dp_split_dim] //= dp_size

            if dst_tp_size > tp_size:
                ratio = dst_tp_size // tp_size
                num_tp_chunks = ratio
                if tp_split_dim != -1:
                    shape[tp_split_dim] //= ratio
                elif (
                    tp_split_dim == -1
                ):
                    recv_info["tensors"][key]["tp_split_dim"] = 0
                    shape[0] //= ratio
                recv_info["tensors"][key]["num_tp_chunks"] = num_tp_chunks

                send_info["tensors"][key]["num_tp_chunks"] = num_tp_chunks
                send_info["tensors"][key]["tp_chunks_index"] = range(num_tp_chunks)

            if dst_tp_size < tp_size:
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

            if dst_cp_size > cp_size:
                ratio = dst_cp_size // cp_size
                num_cp_chunks = ratio

                if cp_split_dim != -1:
                    recv_info["tensors"][key]["cp_split_dim"] = cp_split_dim
                    shape[cp_split_dim] //= ratio
                else:
                    recv_info["tensors"][key]["cp_split_dim"] = 0
                    shape[0] //= ratio
                recv_info["tensors"][key]["num_cp_chunks"] = num_cp_chunks

                send_info["tensors"][key]["cp_split_dim"] = cp_split_dim
                send_info["tensors"][key]["num_cp_chunks"] = num_cp_chunks
                send_info["tensors"][key]["cp_chunks_index"] = range(num_cp_chunks)

            if dst_cp_size < cp_size:
                recv_info["tensors"][key]["cp_split_dim"] = cp_split_dim
                recv_info["tensors"][key]["num_cp_chunks"] = num_cp_chunks

                if cp_split_dim != -1:
                    send_info["tensors"][key]["cp_split_dim"] = cp_split_dim
                    send_info["tensors"][key]["num_cp_chunks"] = num_cp_chunks
                    send_info["tensors"][key]["cp_chunks_index"] = range(num_cp_chunks)
                else:
                    ratio = cp_size // dst_cp_size
                    num_cp_chunks = ratio
                    send_info["tensors"][key]["cp_split_dim"] = 0
                    send_info["tensors"][key]["num_cp_chunks"] = num_cp_chunks
                    send_info["tensors"][key]["cp_chunks_index"] = [
                        cp_id % num_cp_chunks
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

def initialize_communication(model_chunk_op_list):
    # get input_tensors_info and output_tensors_info
    pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    virtual_pipeline_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    op_start_index = mpu.get_op_start_index(pipeline_rank, virtual_pipeline_rank)
    op_end_index = mpu.get_op_end_index(pipeline_rank, virtual_pipeline_rank)

    input_tensors_info = model_chunk_op_list[0].input_tensors_info
    output_tensors_info = model_chunk_op_list[
        op_end_index - op_start_index - 1
    ].output_tensors_info

    # get input_extra_tensor_dict
    input_extra_tensors_dict = {}
    if pipeline_rank > 0 or virtual_pipeline_rank > 0:
        op_start_index_prev_stage, op_end_index_prev_stage = get_prev_stage_index(
            pipeline_rank, virtual_pipeline_rank
        )
        for op_index in range(op_start_index, op_end_index):
            op = model_chunk_op_list[op_index - op_start_index]
            # if the op in my stage rank need input_extra_tensor recved from prev stage
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
    # get output_extra_tensors_dict
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
            # if the op in my stage rank need output_extra_tensor sended to next stage
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

    if mpu.get_pipeline_model_parallel_world_size() == 1:
        fwd_send_info = {}
        bwd_send_info = {}
        fwd_recv_info = {}
        bwd_recv_info = {}
    else:
        bwd_send_info, fwd_recv_info = initialize_comm_info2(
            input_tensors_info,
            prev_stage,
            src_op_index=model_chunk_op_list[0].op_index,
            dst_op_index=model_chunk_op_list[0].op_index - 1,
        )
        fwd_send_info, bwd_recv_info = initialize_comm_info2(
            output_tensors_info,
            next_stage,
            src_op_index=model_chunk_op_list[-1].op_index,
            dst_op_index=model_chunk_op_list[-1].op_index + 1,
        )

        for key in input_extra_tensors_dict:
            _bwd_send_info, _fwd_recv_info = initialize_comm_info2(
                input_extra_tensors_dict[key]["info"],
                prev_stage,
                src_op_index=model_chunk_op_list[0].op_index,
                dst_op_index=model_chunk_op_list[0].op_index - 1,
            )
            fwd_recv_info["size"] += _fwd_recv_info["size"]
            fwd_recv_info["tensors"][key] = _fwd_recv_info["tensors"][key]
            bwd_send_info["tensors"][key] = _bwd_send_info["tensors"][key]

        for key in output_extra_tensors_dict:
            _fwd_send_info, _bwd_recv_info = initialize_comm_info2(
                output_extra_tensors_dict[key]["info"],
                next_stage,
                src_op_index=model_chunk_op_list[-1].op_index,
                dst_op_index=model_chunk_op_list[-1].op_index + 1,
            )
            bwd_recv_info["size"] += _bwd_recv_info["size"]
            bwd_recv_info["tensors"][key] = _bwd_recv_info["tensors"][key]
            fwd_send_info["tensors"][key] = _fwd_send_info["tensors"][key]
    if mpu.is_pipeline_first_stage():
        bwd_send_info["tensors"] = {}
        fwd_recv_info["tensors"] = {}
        fwd_recv_info["size"] = 0
    elif mpu.is_pipeline_last_stage():
        fwd_send_info["tensors"] = {}
        bwd_recv_info["tensors"] = {}
        bwd_recv_info["size"] = 0

    # mark the tensor is extra_tensor or not
    if mpu.get_pipeline_model_parallel_world_size() > 1:
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
        self.resharding = False

        full_model_op_list[0].prev_name = None
        full_model_op_list[-1].is_last_op = True
        self.ops = torch.nn.ModuleList(full_model_op_list)
        pre_hook = pre_forward_hook
        post_hook = post_forward_hook
        for op in self.ops:
            op.register_forward_pre_hook(pre_hook)
            op.register_forward_hook(post_hook)

        self.num_ops = len(full_model_op_list)
        initialize_communication(full_model_op_list)
        print_ops_info(self.ops)

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

        for index in range(self.num_ops):
            op = self.ops[index]
            print(f'rank {torch.distributed.get_rank()} op {op.op_name} index {index}')
            if self.config.timers:
                self.config.timers(f"{op.op_name}-forward-outside", log_level=1).start()
            hidden_states = op(
                hidden_states, input_extra_tensors, output_extra_tensors
            )
            if self.config.timers:
                self.config.timers(f"{op.op_name}-forward-outside").stop()
                
        NUM_BATCHES = NUM_BATCHES + 1
        output = hidden_states

        return output, output_extra_tensors


def get_flex_model(
    config: TransformerConfig,
    full_model_op_list,
    pre_process=True,
    post_process=True,
):
    language_model = FlexPipeModel(
        config,
        full_model_op_list,
        pre_process=pre_process,
        post_process=post_process,
    )

    return language_model
