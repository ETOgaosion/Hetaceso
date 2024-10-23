# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import operator
from functools import reduce
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron.training import get_args, get_timers
from megatron import core
from megatron.core import mpu
from megatron.core import ModelParallelConfig
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_pipeline_model_parallel_rank,
)
from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.legacy.model import Float16Module

from megatron.training.utils import unwrap_model

from megatron.core.utils import debug_mem_report, report_memory
import os
import time
import inspect

DEBUG_COMMUNICATE = os.environ.get("DEBUG_COMMUNICATE", '0') == '1'
EXTRA_TENSOR_TRANSFER = os.environ.get("EXTRA_TENSOR_TRANSFER", '1') == '1'

def get_debug_file() -> str:
    args = get_args()
    current_rank = torch.distributed.get_rank()
    file_name = f"{args.log_path}/p2p_debug_{current_rank}.log"
    return file_name

def get_code_info() -> str:
    frame = inspect.currentframe().f_back
    # 提取文件名和行号信息
    frame_info = inspect.getframeinfo(frame)
    # 使用 os.path.basename 只获取文件名
    filename = os.path.basename(frame_info.filename)
    return f"[{filename}: {frame_info.lineno}] "

# Types
Shape = Union[List[int], torch.Size]

def print_tensor_dict_info(name, tensor_dict):
    args = get_args()
    string = f"rank {torch.distributed.get_rank()} {name} dict: \n"
    for key in sorted(tensor_dict):
        if tensor_dict[key] is not None:
            string += f"{key}: {list(tensor_dict[key].size())} size = {reduce(operator.mul, list(tensor_dict[key].size()), 1)}\n"
        else:
            string += f"{key}: {None}\n"

    with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{torch.distributed.get_rank()}.log", "a+") as f:
        f.write(string+"\n")

def print_communication_info(current_rank, op, other_rank, tensor_size):
    args = get_args()
    string = f"rank {current_rank} | {op} {other_rank}. size = {tensor_size}."
    with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{current_rank}.log", "a+") as f:
        f.write(string+"\n")


def _create_recv_placeholder(forward=True):
    args = get_args()
    dtype = args.params_dtype
    if args.fp32_residual_connection:
        dtype = torch.float   

    recv_info = mpu.get_recv_info(forward)
    flatten_tensor_recv_prev = {}
    for key in sorted(recv_info["tensors"]):
        flatten_tensor_recv_prev[key] = []
        num_chunks = recv_info["tensors"][key]["num_tp_chunks"] * recv_info["tensors"][key]["num_dp_chunks"]
        recv_shape = list(recv_info["tensors"][key]["shape"])
        if recv_info["tensors"][key]["tp_split_dim"] == -1 and args.scatter_gather_tensors_in_pipeline:
            rank = mpu.get_pipeline_model_parallel_rank()
            if forward:
                op_index = mpu.get_op_start_index(rank)
            else:
                op_index = mpu.get_op_end_index(rank) - 1

            assert recv_shape[0] % mpu.get_op_tp_size(op_index) == 0, f'rank {torch.distributed.get_rank()} recv_shape {recv_shape} recv_shape[0] {recv_shape[0]} is not divisible by mpu.get_op_tp_size(op_index) {mpu.get_op_tp_size(op_index)} recv_info: {recv_info} op_index: {op_index}'
            recv_shape[0] //= mpu.get_op_tp_size(op_index)
            recv_shape[0] //= recv_info["tensors"][key]["num_tp_chunks"]
        for _ in range(num_chunks):
            flatten_tensor_recv_prev[key].append(torch.empty(recv_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype))

    return flatten_tensor_recv_prev

def _partition(tensor, info, forward):
    """
    This function first partition each tensor and extra tensor according to number of receivers.
    Then flatten all the tensors and concat them into one large tensor.
    """
    
    tp_split_dim = info["tp_split_dim"]
    dp_split_dim = info["dp_split_dim"]
    num_tp_chunks = info["num_tp_chunks"]
    num_dp_chunks = info["num_dp_chunks"]
    tp_chunks_index = info["tp_chunks_index"]
    dp_chunks_index = info["dp_chunks_index"]
    args = get_args()

    if dp_split_dim != -1:
        _tmp_list = list(torch.chunk(tensor, chunks=num_dp_chunks, dim=dp_split_dim)) 
        tensor_split = []
        for i in range(len(dp_chunks_index)):
            tensor_split.append(_tmp_list[dp_chunks_index[i]].contiguous())
    else:
        tensor_split = [tensor for _ in range(num_dp_chunks)]
    
    if tp_split_dim != -1:
        for i in range(len(tensor_split)):
            _tmp_list = list(torch.chunk(tensor_split[i], chunks=num_tp_chunks, dim=tp_split_dim)) 
            tensor_split[i] = []
            for j in range(len(tp_chunks_index)):
                tensor_split[i].append(_tmp_list[tp_chunks_index[j]].contiguous())                
    else:
        for i in range(len(tensor_split)):
            if args.scatter_gather_tensors_in_pipeline:
                rank = mpu.get_pipeline_model_parallel_rank()
                if forward:
                    op_index = mpu.get_op_end_index(rank) - 1
                else:
                    op_index = mpu.get_op_start_index(rank)

                assert tensor_split[i].size()[0] >= num_tp_chunks * mpu.get_op_tp_size(op_index), "scatter_gather_tensors_in_pipeline is only available when mciro batch size >= num_splits"
                _tmp_list = list(torch.chunk(tensor_split[i], chunks=num_tp_chunks * mpu.get_op_tp_size(op_index), dim=0)) 
                tp_rank = torch.distributed.get_rank(group=mpu.get_tensor_model_parallel_group(op_index))
                new_tensor_split = [_tmp_list[num_tp_chunks * tp_rank + j].contiguous() for j in range(num_tp_chunks)]
            else:
                new_tensor_split = [tensor_split[i] for _ in range(num_tp_chunks)]
            tensor_split[i] = new_tensor_split

    _tensor_split = [n for a in tensor_split for n in a]

    return _tensor_split

def _reshape(recv_tensor, recv_info, forward):
    args = get_args()
    tensor_dict = {}
    extra_tensor_dict = {}

    for key in sorted(recv_info["tensors"]):
        num_tp_chunks = recv_info["tensors"][key]["num_tp_chunks"]
        num_dp_chunks = recv_info["tensors"][key]["num_dp_chunks"]
        tp_split_dim = recv_info["tensors"][key]["tp_split_dim"]
        dp_split_dim = recv_info["tensors"][key]["dp_split_dim"]
        tensor_list = recv_tensor[key]       

        if not EXTRA_TENSOR_TRANSFER and recv_info["tensors"][key]["extra_tensor"]:
            data_size = tensor_list[0].size()
            data_type = torch.float16
            for i in range(len(tensor_list)):
                tensor_list[i] = torch.ones(data_size, requires_grad=True, device=torch.cuda.current_device(), dtype=data_type) 

        if num_tp_chunks > 1:
            if tp_split_dim == -1 and args.scatter_gather_tensors_in_pipeline:
                _tensor_list = []
                for i in range(len(tensor_list)):
                    _tensor_list.append(torch.cat(tensor_list[i: i+num_tp_chunks], dim=0))
                    i += num_tp_chunks
                tensor_list = _tensor_list  
            else:
                _tensor_list = []
                for i in range(len(tensor_list)):
                    _tensor_list.append(torch.cat(tensor_list[i: i+num_tp_chunks], dim=tp_split_dim))
                    i += num_tp_chunks
                tensor_list = _tensor_list  

        if num_dp_chunks > 1:
            _tensor_list = []
            for i in range(len(tensor_list)):
                _tensor_list.append(torch.cat(tensor_list[i: i+num_dp_chunks], dim=dp_split_dim))
                i += num_dp_chunks
            tensor_list = _tensor_list  

        if tp_split_dim == -1 and args.scatter_gather_tensors_in_pipeline:
            rank = mpu.get_pipeline_model_parallel_rank()
            if forward:
                op_index = mpu.get_op_start_index(rank)  
            else:
                op_index = mpu.get_op_end_index(rank) - 1
            tp_size = mpu.get_op_tp_size(op_index)

            gather_list = [torch.empty_like(tensor_list[0]) for _ in range(tp_size)]
            torch.distributed.all_gather(gather_list, tensor_list[0], group=mpu.get_tensor_model_parallel_group(op_index))
            output = torch.cat(gather_list, dim=0).contiguous()

            if recv_info["tensors"][key]["extra_tensor"]:
                extra_tensor_dict[key] = output
            else:
                tensor_dict[key] = output
        else:
            if recv_info["tensors"][key]["extra_tensor"]:
                extra_tensor_dict[key] = tensor_list[0]  
            else:
                tensor_dict[key] = tensor_list[0]

    if DEBUG_COMMUNICATE:
        print_tensor_dict_info("recieved tensors", tensor_dict)
        print_tensor_dict_info("received extra tensors", extra_tensor_dict)

    return tensor_dict, extra_tensor_dict


def _communicate_flexpipe(
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    extra_tensor_send_next: Optional[torch.Tensor],
    extra_tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    timers = get_timers()

    prev_ranks = mpu.get_stage_comm_recv_ranks()
    next_ranks = mpu.get_stage_comm_send_ranks()
    num_parents = len(prev_ranks)
    num_childs = len(next_ranks) 
    tensor_recv_prev, extra_tensor_recv_prev, tensor_recv_next, extra_tensor_recv_next = None, None, None, None 

    # Create placeholder tensors for receive in forward and backward directions if needed.
    with torch.no_grad():
        if recv_prev:
            flatten_tensor_recv_prev = _create_recv_placeholder(forward=True)
        if recv_next:
            flatten_tensor_recv_next = _create_recv_placeholder(forward=False)

    if tensor_send_prev is not None:
        send_info = mpu.get_send_info(forward=False)
        for key in sorted(send_info["tensors"]):
            ops = []
            with torch.no_grad():
                if key in tensor_send_prev:
                    tensor_partitioned = _partition(tensor_send_prev[key], send_info["tensors"][key], forward=False)
                elif key in extra_tensor_send_prev:
                    if EXTRA_TENSOR_TRANSFER:
                        tensor_partitioned = _partition(extra_tensor_send_prev[key], send_info["tensors"][key], forward= False)
                    else:
                        continue
                else:
                    print(f"[rank {torch.distributed.get_rank()}] trying to send to prev, tensor name = {key}. send_info = {send_info['tensors']}")
            for i in range(num_parents):
                send_prev_op = torch.distributed.P2POp(torch.distributed.isend, tensor_partitioned[i], prev_ranks[i])
                ops.append(send_prev_op)  
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"send [{key} ({tensor_partitioned[i].dtype})] to ", prev_ranks[i], list(tensor_partitioned[i].size()))
            if recv_prev:
                recv_info = mpu.get_recv_info(forward=True)
                for i in range(num_parents):
                    recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_prev[key][i], prev_ranks[i])
                    ops.append(recv_prev_op)
                    if DEBUG_COMMUNICATE:
                        print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", prev_ranks[i], list(flatten_tensor_recv_prev[key][i].size()))                

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            # torch.cuda.synchronize()
    elif recv_prev:
        recv_info = mpu.get_recv_info(forward=True)
        for key in sorted(recv_info["tensors"]): 
            if recv_info["tensors"][key]["extra_tensor"] and not EXTRA_TENSOR_TRANSFER:
                continue
            ops = []    
            for i in range(num_parents):
                recv_prev_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_prev[key][i], prev_ranks[i])
                ops.append(recv_prev_op)
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", prev_ranks[i], list(flatten_tensor_recv_prev[key][i].size()))

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()  
            # torch.cuda.synchronize()        

    if tensor_send_next is not None:
        send_info = mpu.get_send_info(forward=True)
        for key in sorted(send_info["tensors"]):
            ops = []
            with torch.no_grad():
                if key in tensor_send_next:
                    tensor_partitioned = _partition(tensor_send_next[key], send_info["tensors"][key], forward=True)
                elif key in extra_tensor_send_next:
                    if EXTRA_TENSOR_TRANSFER:
                        tensor_partitioned = _partition(extra_tensor_send_next[key], send_info["tensors"][key], forward=True) 
                    else:
                        continue
            for i in range(num_childs):
                send_next_op = torch.distributed.P2POp(torch.distributed.isend, tensor_partitioned[i], next_ranks[i])
                ops.append(send_next_op)  
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"send [{key}] to ", next_ranks[i], list(tensor_partitioned[i].size()))
            if recv_next:
                recv_info = mpu.get_recv_info(forward=False)
                for i in range(num_childs):
                    recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_next[key][i], next_ranks[i])
                    ops.append(recv_next_op)
                    if DEBUG_COMMUNICATE:
                        print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", next_ranks[i], list(flatten_tensor_recv_next[key][i].size()))                

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()
            # torch.cuda.synchronize()

    elif recv_next:
        recv_info = mpu.get_recv_info(forward=False)
        for key in sorted(recv_info["tensors"]): 
            if recv_info["tensors"][key]["extra_tensor"] and not EXTRA_TENSOR_TRANSFER:
                continue            
            ops = []          
            for i in range(num_childs):
                recv_next_op = torch.distributed.P2POp(torch.distributed.irecv, flatten_tensor_recv_next[key][i], next_ranks[i])
                ops.append(recv_next_op)
                if DEBUG_COMMUNICATE:
                    print_communication_info(torch.distributed.get_rank(), f"recv [{key}] from ", next_ranks[i], list(flatten_tensor_recv_next[key][i].size()))  

            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()  
    # if len(ops) > 0:
    #     reqs = torch.distributed.batch_isend_irecv(ops)
    #     for req in reqs:
    #         req.wait()
    # To protect against race condition when using batch_isend_irecv().
    torch.cuda.synchronize()

    with torch.no_grad():
        if recv_prev:
            tensor_recv_prev, extra_tensor_recv_prev = _reshape(flatten_tensor_recv_prev, recv_info, forward=True)
        if recv_next:
            tensor_recv_next, extra_tensor_recv_next = _reshape(flatten_tensor_recv_next, recv_info, forward=False)

    if recv_prev:
        for key in sorted(tensor_recv_prev):
            tensor_recv_prev[key].requires_grad = True
        for key in sorted(extra_tensor_recv_prev):
            extra_tensor_recv_prev[key].requires_grad = True    
    if recv_next:
        for key in sorted(tensor_recv_next):
            tensor_recv_next[key].requires_grad = True
        for key in sorted(extra_tensor_recv_next):
            extra_tensor_recv_next[key].requires_grad = True                    

    return tensor_recv_prev, extra_tensor_recv_prev, tensor_recv_next, extra_tensor_recv_next



def _communicate_shapes(tensor_send_next, tensor_send_prev, recv_prev, recv_next, config):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.

    Args:
        tensor_send_next: tensor to send to next rank (no tensor sent if
                          set to None).
        tensor_send_prev: tensor to send to prev rank (no tensor sent if
                          set to None).
        recv_prev: boolean for whether tensor should be received from
                   previous rank.
        recv_next: boolean for whether tensor should be received from
                   next rank.
    Returns:
        (recv_prev_shape, recv_next_shape)
    """

    recv_prev_shape_tensor = None
    recv_next_shape_tensor = None
    send_prev_shape_tensor = None
    send_next_shape_tensor = None
    if recv_prev:
        recv_prev_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if recv_next:
        recv_next_shape_tensor = torch.empty(
            (3), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    if config.use_ring_exchange_p2p:
        torch.distributed.ring_exchange(
            tensor_send_prev=send_prev_shape_tensor,
            tensor_recv_prev=recv_prev_shape_tensor,
            tensor_send_next=send_next_shape_tensor,
            tensor_recv_next=recv_next_shape_tensor,
            group=get_pipeline_model_parallel_group(),
        )
    else:
        ops = []
        if send_prev_shape_tensor is not None:
            send_prev_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(send_prev_op)
        if recv_prev_shape_tensor is not None:
            recv_prev_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_prev_shape_tensor,
                get_pipeline_model_parallel_prev_rank(),
            )
            ops.append(recv_prev_op)
        if send_next_shape_tensor is not None:
            send_next_op = torch.distributed.P2POp(
                torch.distributed.isend,
                send_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(send_next_op)
        if recv_next_shape_tensor is not None:
            recv_next_op = torch.distributed.P2POp(
                torch.distributed.irecv,
                recv_next_shape_tensor,
                get_pipeline_model_parallel_next_rank(),
            )
            ops.append(recv_next_op)
        if len(ops) > 0:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

        # To protect against race condition when using batch_isend_irecv().
        # should take this out once the bug with batch_isend_irecv is resolved.
        torch.cuda.synchronize()

    recv_prev_shape = [0, 0, 0]
    if recv_prev_shape_tensor is not None:
        recv_prev_shape = recv_prev_shape_tensor.tolist()

    recv_next_shape = [0, 0, 0]
    if recv_next_shape_tensor is not None:
        recv_next_shape = recv_next_shape_tensor.tolist()

    return recv_prev_shape, recv_next_shape


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_prev,
            get_pipeline_model_parallel_prev_rank(),
            group,
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend,
            tensor_send_next,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv,
            tensor_recv_next,
            get_pipeline_model_parallel_next_rank(),
            group,
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup
):
    reqs = []
    rank = get_pipeline_model_parallel_rank()
    if get_pipeline_model_parallel_rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(recv_prev_req)

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(send_next_req)

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=get_pipeline_model_parallel_next_rank(), group=group,
            )
            reqs.append(recv_next_req)

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=get_pipeline_model_parallel_prev_rank(), group=group,
            )
            reqs.append(send_prev_req)
    return reqs


def _communicate(
    *,
    tensor_send_next: Optional[torch.Tensor],
    tensor_send_prev: Optional[torch.Tensor],
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    wait_on_reqs: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Communicate tensors between stages. Used as helper method in other
    communication methods that are used in megatron/schedules.py.

    Args:
        tensor_send_next (torch.Tensor, optional):
            Tensor to send to next rank (no tensor sent if None)

        tensor_send_prev (torch.Tensor, optional):
            Tensor to send to prev rank (no tensor sent if None)

        recv_prev (boolean, required):
            whether tensor should be received from previous rank.

        recv_next (boolean, required):
            whether tensor should be received from next rank.

        tensor_shape (List[int] or torch.Size, required):
            shape of tensor to receive (this method assumes that all
            tensors sent and received in a single function call are
            the same shape).

        wait_on_reqs (boolean, optional, default=False):
            For non-batched p2p communication, wait on each request
            before returning.

    Returns:
        tuple containing

        - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
        - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

    """

    # Create placeholder tensors for receive in forward and backward directions
    # if needed.
    tensor_recv_prev = None
    tensor_recv_next = None

    if not config.variable_seq_lengths:
        recv_prev_shape = tensor_shape
        recv_next_shape = tensor_shape
    else:
        recv_prev_shape, recv_next_shape = _communicate_shapes(
            tensor_send_next, tensor_send_prev, recv_prev, recv_next, config
        )

    if recv_prev:
        if config.pipeline_dtype is None:
            raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_prev is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_prev = torch.empty(
            recv_prev_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )
    if recv_next:
        if config.pipeline_dtype is None:
            raise RuntimeError("dtype must be provided if recv_next is True")
        if tensor_shape is None:
            raise RuntimeError(
                "tensor_shape must be specified if recv_next is True. "
                "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
            )
        tensor_recv_next = torch.empty(
            recv_next_shape,
            requires_grad=True,
            device=torch.cuda.current_device(),
            dtype=config.pipeline_dtype,
        )

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        assert wait_on_reqs
        p2p_func = _batched_p2p_ops
    else:
        p2p_func = _p2p_ops

    reqs = p2p_func(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=get_pipeline_model_parallel_group(),
    )

    if wait_on_reqs and len(reqs) > 0:
        for req in reqs:
            req.wait()
        reqs = None

    if config.batch_p2p_comm and config.batch_p2p_sync:
        # To protect against race condition when using batch_isend_irecv().
        # User should assert that we have a modern enough PyTorch to not need this
        torch.cuda.synchronize()

    return tensor_recv_prev, tensor_recv_next, reqs


def recv_forward(tensor_shape: Shape, config: ModelParallelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """ Receive tensor from previous rank in pipeline (forward receive).

    See _communicate for argument details.
    """

    if core.parallel_state.is_pipeline_first_stage():
        input_tensors = None
        input_extra_tensors = None
    else:
        if config.timers is not None:
            config.timers('forward-recv', log_level=2).start()
        input_tensors, input_extra_tensors, _, _ = _communicate_flexpipe(
            tensor_send_next=None,
            tensor_send_prev=None,
            extra_tensor_send_next=None,
            extra_tensor_send_prev=None,
            recv_prev=True,
            recv_next=False,
        )
        if config.timers is not None:
            config.timers('forward-recv').stop()
    return input_tensors, input_extra_tensors


def recv_backward(tensor_shape: Shape, config: ModelParallelConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Receive tensor from next rank in pipeline (backward receive).

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
    else:
        if config.timers is not None:
            config.timers('backward-recv', log_level=2).start()
        _, _, output_tensor_grad, output_extra_tensors_grad = _communicate_flexpipe(
            tensor_send_next=None,
            tensor_send_prev=None,
            extra_tensor_send_next=None,
            extra_tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
        )
        if config.timers is not None:
            config.timers('backward-recv').stop()
    return output_tensor_grad, output_extra_tensors_grad


def send_forward(output_tensor: torch.Tensor, config: ModelParallelConfig, output_extra_tensors: torch.Tensor = None) -> None:
    """Send tensor to next rank in pipeline (forward send).

    See _communicate for argument details.
    """

    if not core.parallel_state.is_pipeline_last_stage():
        if config.timers is not None:
            config.timers('forward-send', log_level=2).start()
        _communicate_flexpipe(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            extra_tensor_send_next=output_extra_tensors,
            extra_tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
        )
        if config.timers is not None:
            config.timers('forward-send').stop()


def send_backward(input_tensor_grad: torch.Tensor, config: ModelParallelConfig, extra_tensors_grad: torch.Tensor = None) -> None:
    """Send tensor to previous rank in pipeline (backward send).

    See _communicate for argument details.
    """
    if not core.parallel_state.is_pipeline_first_stage():
        if config.timers is not None:
            config.timers('backward-send', log_level=2).start()
        _communicate_flexpipe(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            extra_tensor_send_next=extra_tensors_grad,
            extra_tensor_send_prev=None,
            recv_prev=False,
            recv_next=False,
        )
        if config.timers is not None:
            config.timers('backward-send').stop()


def send_forward_recv_backward(
    output_tensor: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig, output_extra_tensors: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched send and recv with next rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_last_stage():
        output_tensor_grad = None
        output_extra_tensors_grad = None
    else:
        if config.timers is not None:
            config.timers('forward-send-backward-recv', log_level=2).start()
        _, _, output_tensor_grad, output_extra_tensors_grad = _communicate_flexpipe(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            extra_tensor_send_next=output_extra_tensors,
            extra_tensor_send_prev=None,
            recv_prev=False,
            recv_next=True,
        )
        if config.timers is not None:
            config.timers('forward-send-backward-recv').stop()
    return output_tensor_grad, output_extra_tensors_grad


def send_backward_recv_forward(
    input_tensor_grad: torch.Tensor, tensor_shape: Shape, config: ModelParallelConfig, extra_tensors_grad: torch.Tensor = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched send and recv with previous rank in pipeline.

    See _communicate for argument details.
    """
    if core.parallel_state.is_pipeline_first_stage():
        input_tensor = None
        extra_tensors = None
    else:
        if config.timers is not None:
            config.timers('backward-send-forward-recv', log_level=2).start()
        input_tensor, extra_tensors, _, _ = _communicate_flexpipe(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            extra_tensor_send_next=None,
            extra_tensor_send_prev=extra_tensors_grad,
            recv_prev=True,
            recv_next=False,
        )
        if config.timers is not None:
            config.timers('backward-send-forward-recv').stop()
    return input_tensor, extra_tensors


def send_forward_recv_forward(
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    # overlap_p2p_comm: bool = False,
    output_extra_tensors: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched recv from previous rank and send to next rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-send-forward-recv', log_level=2).start()
    input_tensor, extra_tensors, _, _ = _communicate_flexpipe(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        extra_tensor_send_next=output_extra_tensors,
        extra_tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
    )
    if config.timers is not None:
        config.timers('forward-send-forward-recv').stop()
    # if overlap_p2p_comm:
    #     return input_tensor, wait_handles
    return input_tensor, extra_tensors


def send_backward_recv_backward(
    input_tensor_grad: torch.Tensor,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    # overlap_p2p_comm: bool = False,
    extra_tensors_grad: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched recv from next rank and send to previous rank in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('backward-send-backward-recv', log_level=2).start()
    _, _, output_tensor_grad, output_extra_tensors_grad = _communicate_flexpipe(
        tensor_send_next=None,
        tensor_send_prev=input_tensor_grad,
        extra_tensor_send_next=None,
        extra_tensor_send_prev=extra_tensors_grad,
        recv_prev=False,
        recv_next=recv_next,
    )
    if config.timers is not None:
        config.timers('backward-send-backward-recv').stop()
    # if overlap_p2p_comm:
    #     return output_tensor_grad, wait_handles
    return output_tensor_grad, output_extra_tensors_grad

def send_forward_backward_recv_forward_backward(
    output_tensor: torch.Tensor,
    input_tensor_grad: torch.Tensor,
    recv_prev: bool,
    recv_next: bool,
    tensor_shape: Shape,
    config: ModelParallelConfig,
    output_extra_tensors: torch.Tensor = None,
    extra_tensors_grad: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched send and recv with previous and next ranks in pipeline.

    See _communicate for argument details.
    """
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
    input_tensor, extra_tensors, output_tensor_grad, output_extra_tensors_grad = _communicate_flexpipe(
        tensor_send_next=output_tensor,
        tensor_send_prev=input_tensor_grad,
        extra_tensor_send_next=output_extra_tensors,
        extra_tensor_send_prev=extra_tensors_grad,
        recv_prev=recv_prev,
        recv_next=recv_next,
    )
    if config.timers is not None:
        config.timers('forward-backward-send-forward-backward-recv').stop()
    return input_tensor, extra_tensors, output_tensor_grad, output_extra_tensors_grad


def get_op_via_index(op_index, models):
    for model in models:
        model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
        for op in model.language_model.ops:
            if op.op_index == op_index:
                return op
    return None


def send_shared_tensors(op, models, grads=False):
    
    args = get_args()
    shared_tensor = op.get_shared_tensor(grads=grads)

    for key in sorted(shared_tensor):
        for op_index in op.shared_weights_info[key]["sharing_with_ops"]:
            print(f'{torch.distributed.get_rank()}, op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"]: {op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"]}, op.shared_weights_info[key]["sharing_weights_with_ranks": {op.shared_weights_info[key]["sharing_weights_with_ranks"]}')
            if not op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index]:
                recv_ranks = op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index]
                if len(recv_ranks) > 0:
                    send_ops = []
                    split_dim = op.shared_weights_info[key]["tp_split_dim"]

                    for recv_tp_groups in recv_ranks:
                        if len(recv_tp_groups) == 1:
                            tensor_list = [shared_tensor[key]]
                        else:
                            if split_dim != -1:
                                tensor_list = list(torch.chunk(shared_tensor[key], chunks=len(recv_tp_groups), dim=split_dim)) 
                            else:
                                tensor_list = []
                                for _ in range(len(recv_tp_groups)):
                                    tensor_list.append(shared_tensor[key])

                        for i in range(len(tensor_list)):
                            send_op = torch.distributed.P2POp(
                                torch.distributed.isend, tensor_list[i].contiguous(), recv_tp_groups[i])
                            send_ops.append(send_op) 

                            if DEBUG_COMMUNICATE:
                                current_rank = torch.distributed.get_rank()
                                string = f"(shared) rank {current_rank} send to {recv_tp_groups[i]} size = {list(tensor_list[i].size())}"
                                with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{current_rank}.log", "a+") as f:
                                    f.write(string+"\n")    

                    if len(send_ops) > 0:
                        reqs = torch.distributed.batch_isend_irecv(send_ops)
                        for req in reqs:
                            req.wait()
                        torch.cuda.synchronize()

def recv_shared_tensors(op, models, grads=False):
    args = get_args()

    recv_dict = {}
    shared_tensor = op.get_shared_tensor(grads=False)
    for key in sorted(shared_tensor):
        recv_dict[key] = []

    for key in sorted(shared_tensor):
        if key == "position_embeddings" and not grads:
            dtype = torch.float32
        else:
            dtype = args.params_dtype        
        for op_index in op.shared_weights_info[key]["sharing_with_ops"]:
            print(f'{torch.distributed.get_rank()}, op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"]: {op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"]}, op.shared_weights_info[key]["sharing_weights_with_ranks": {op.shared_weights_info[key]["sharing_weights_with_ranks"]}')
            if op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index]:
                src_op = get_op_via_index(op_index, models)
                recv_tensor = src_op.get_shared_tensor(grads=grads)
                recv_dict[key].append(recv_tensor[key])
            else:
                send_ranks = op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index]
                if len(send_ranks) > 0: 
                    recv_ops = []
                    tensor_list = []
                    receive_size = list(shared_tensor[key].size())
                    split_dim = op.shared_weights_info[key]["tp_split_dim"]
                    if split_dim != -1:
                        receive_size[split_dim] //= len(send_ranks[0])
                        
                    for send_tp_groups in send_ranks:
                        tmp_tensor_list = []
                        for _ in range(len(send_tp_groups)):
                            tmp_tensor_list.append(torch.empty(receive_size, requires_grad=True, device=torch.cuda.current_device(), dtype=dtype))
                        for i in range(len(tmp_tensor_list)):
                            recv_op = torch.distributed.P2POp(
                                torch.distributed.irecv, tmp_tensor_list[i], send_tp_groups[i])
                            recv_ops.append(recv_op)
                            if DEBUG_COMMUNICATE:
                                current_rank = torch.distributed.get_rank()
                                string = f"(shared) rank {current_rank} recv from {send_tp_groups[i]} size = {list(tmp_tensor_list[i].size())}"
                                with open(f"{args.log_path}{args.log_name}_debug_communicate_rank{current_rank}.log", "a+") as f:
                                    f.write(string+"\n")    
                        tensor_list.append(tmp_tensor_list)

                    if len(recv_ops) > 0:
                        reqs = torch.distributed.batch_isend_irecv(recv_ops)
                        for req in reqs:
                            req.wait()
                        torch.cuda.synchronize()

                    if split_dim != -1:
                        if len(tensor_list) == 1:
                            recv_dict[key].append(torch.cat(tensor_list[0], dim=split_dim))
                        else:
                            result_tensor = torch.sum(torch.stack([torch.cat(tensor_list[i], dim=split_dim) for i in range(len(tensor_list))]), dim=0)
                            recv_dict[key].append(result_tensor)
                    else:
                        if len(tensor_list) == 1:
                            recv_dict[key].append(tensor_list[0][0])
                        else:
                            result_tensor = torch.sum(torch.stack([tensor_list[i][0] for i in range(len(tensor_list))]), dim=0)
                            recv_dict[key].append(result_tensor)
    return recv_dict

def initialize_weights_sharing(models):
    pipeline_rank = mpu.get_pipeline_model_parallel_rank() 
    rank = torch.distributed.get_rank()
    # initialize the ranks
    for model in models:
        model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
        for op in model.language_model.ops:
            if len(op.shared_weights_info) > 0:
                for key in sorted(op.shared_weights_info):
                    op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"] = {}   
                    op.shared_weights_info[key]["sharing_weights_with_ranks"] = {}                      
                    if op.shared_weights_info[key]["root"]:
                        # calculate & store the destination ranks. 
                        for op_index in op.shared_weights_info[key]["sharing_with_ops"]:
                            dest_pipeline_rank = mpu.get_pipeline_rank_via_op_index(op_index)
                            if dest_pipeline_rank == pipeline_rank:
                                op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = True
                            else:
                                op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = False

                                ranks_in_send_stage = mpu.get_ranks_via_pipeline_stage(pipeline_rank)
                                ranks_in_receive_stage = mpu.get_ranks_via_pipeline_stage(dest_pipeline_rank)
                                num_ranks_in_send_stage = len(ranks_in_send_stage)
                                num_ranks_in_receive_stage = len(ranks_in_receive_stage)

                                tp_size, dp_size = mpu.get_op_tp_size(op.op_index), mpu.get_op_dp_size(op.op_index)
                                tp_size_next, dp_size_next = mpu.get_op_tp_size(op_index), mpu.get_op_dp_size(op_index)

                                for i in range(num_ranks_in_send_stage):
                                    if ranks_in_send_stage[i] == rank:
                                        dp_id = i // tp_size
                                        tp_id = i % tp_size

                                next_dp_id = [dp_id]
                                next_tp_id = [tp_id]

                                if tp_size_next > tp_size:
                                    ratio = tp_size_next // tp_size
                                    next_tp_id = range(tp_id * ratio, (tp_id + 1)*ratio)                                    
                                if tp_size_next < tp_size:
                                    ratio = tp_size // tp_size_next
                                    next_tp_id = [tp_id // ratio]  
                                if dp_size_next > dp_size:
                                    ratio = dp_size_next // dp_size
                                    next_dp_id = range(dp_id * ratio, (dp_id + 1)*ratio)                                      
                                if dp_size_next < dp_size:
                                    ratio = dp_size // dp_size_next
                                    if dp_id % ratio == 0:
                                        next_dp_id = [dp_id // ratio] 
                                    else:
                                        next_dp_id = []

                                op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index] = []
                                if len(next_dp_id) > 0:
                                    for _dp_id in next_dp_id:
                                        tmp_list = []
                                        for _tp_id in next_tp_id:
                                            tmp_list.append(ranks_in_receive_stage[_dp_id * tp_size_next + _tp_id])
                                        op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index].append(list(tmp_list))
                    else:
                        assert len(op.shared_weights_info[key]["sharing_with_ops"]) == 1
                        op_index = op.shared_weights_info[key]["sharing_with_ops"][0]
                        src_pipeline_rank = mpu.get_pipeline_rank_via_op_index(op_index)
                        if src_pipeline_rank == pipeline_rank:
                            op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = True
                        else:
                            op.shared_weights_info[key]["sharing_weights_in_same_pipeline_rank"][op_index] = False

                            ranks_in_send_stage = mpu.get_ranks_via_pipeline_stage(src_pipeline_rank)
                            ranks_in_receive_stage = mpu.get_ranks_via_pipeline_stage(pipeline_rank)
                            num_ranks_in_send_stage = len(ranks_in_send_stage)
                            num_ranks_in_receive_stage = len(ranks_in_receive_stage)

                            tp_size, dp_size = mpu.get_op_tp_size(op.op_index), mpu.get_op_dp_size(op.op_index)
                            tp_size_next, dp_size_next = mpu.get_op_tp_size(op_index), mpu.get_op_dp_size(op_index)

                            for i in range(num_ranks_in_receive_stage):
                                if ranks_in_receive_stage[i] == rank:
                                    dp_id = i // tp_size
                                    tp_id = i % tp_size

                            next_dp_id = [dp_id]
                            next_tp_id = [tp_id]

                            if tp_size_next > tp_size:
                                ratio = tp_size_next // tp_size
                                next_tp_id = range(tp_id * ratio, (tp_id + 1)*ratio)                                    
                            if tp_size_next < tp_size:
                                ratio = tp_size // tp_size_next
                                next_tp_id = [tp_id // ratio]  
                            if dp_size_next > dp_size:
                                ratio = dp_size_next // dp_size
                                next_dp_id = [dp_id * ratio]                                 
                            if dp_size_next < dp_size:
                                ratio = dp_size // dp_size_next
                                next_dp_id = [dp_id // ratio]   

                            op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index] = []

                            for _dp_id in next_dp_id:
                                tmp_list = []
                                for _tp_id in next_tp_id:
                                    tmp_list.append(ranks_in_send_stage[_dp_id * tp_size_next + _tp_id])
                                op.shared_weights_info[key]["sharing_weights_with_ranks"][op_index].append(list(tmp_list))

    # send & receive tensors
    for model in models:
        model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
        for op in model.language_model.ops:
            if len(op.shared_weights_info) > 0:
                is_root = False 
                for key in op.shared_weights_info:
                    if op.shared_weights_info[key]["root"]:
                        is_root = True
                if is_root:
                    send_shared_tensors(op, models, grads=False)
                else:
                    recv_tensor = recv_shared_tensors(op, models, grads=False)
                    op.set_shared_tensor(recv_tensor, grads=False)
        

def synchronize_shared_weights_grads(models):
    for model in models:
        model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module)) 
        # two-phase to avoid deadlock
        # Phase 1: root: receive, sum up, send out
        #          workers: send
        for op in model.language_model.ops:
            if len(op.shared_weights_info) > 0:
                is_root = False
                for key in op.shared_weights_info:
                    if op.shared_weights_info[key]["root"]:
                        is_root = True                
                if is_root:
                    grads_dict = {}
                    recv_grads_dict = recv_shared_tensors(op, models, grads=True)
                    current_grads_dict = op.get_shared_tensor(grads=True)
                    for key in sorted(op.shared_weights_info):
                        # receive grads from all sync-ops.
                        recv_grads = recv_grads_dict[key]
                        # sum up the grads from all sync-ops and this op.
                        current_grads = current_grads_dict[key]
                        recv_grads.append(current_grads)
                        grads_dict[key] = [sum(recv_grads)]               
                    op.set_shared_tensor(grads_dict, grads=True)                    
                    # send sum of grads back to all the sync-ops.                  
                    send_shared_tensors(op, models, grads=True)                   
                else:
                    # send grads to root op. 
                    send_shared_tensors(op, models, grads=True)

        # Phase 2: workers: receive
        for op in model.language_model.ops:
            if len(op.shared_weights_info) > 0:
                is_root = False
                for key in op.shared_weights_info:
                    if op.shared_weights_info[key]["root"]:
                        is_root = True                  
                if not is_root:               
                    # recv sum of grads.
                    recv_grads = recv_shared_tensors(op, models, grads=True)
                    # update grads.
                    op.set_shared_tensor(recv_grads, grads=True)