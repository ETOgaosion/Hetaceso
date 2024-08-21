# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.parallel_state import (
    get_expert_model_parallel_group,
    get_tensor_and_expert_parallel_group,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_group,
)

from .utils import (
    split_tensor_along_last_dim,
    divide,
)

import numpy as np

#### Aceso:

def new_split(input_, ranks, dim):

    if dim == -1:
        dim = input_.dim() - 1

    dim_size = divide(input_.size()[dim], len(ranks))
    tensor_list = torch.split(input_, dim_size, dim=dim)
    tensor_list = tuple(chunk.contiguous() for chunk in tensor_list)   
    return tensor_list[torch.distributed.get_rank(get_group(ranks))].contiguous()

def new_all_gather(input_, ranks, dim):
    if dim == -1:
        dim = input_.dim() - 1
    
    if not input_.is_contiguous():
        input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in ranks]

    torch.distributed.all_gather(tensor_list, input_, group=get_group(ranks))
    torch.cuda.synchronize()

    # concat
    new_input_ = torch.cat(tensor_list, dim=dim).contiguous().requires_grad_()

    return new_input_    

def new_reduce(input_, ranks):
    """All-reduce the the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if len(ranks)==1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_group(ranks))
    torch.cuda.synchronize()

    return input_

def new_reduce_scatter(input_, ranks, dim):

    input_list = list(input_.chunk(len(ranks), dim))
    for idx, tensor in enumerate(input_list):
        if not tensor.is_contiguous():
            input_list[idx] = tensor.contiguous()
    new_input_ = torch.empty_like(input_list[0], requires_grad=True)
    torch.distributed.reduce_scatter(new_input_, input_list, group=get_group(ranks))
    torch.cuda.synchronize()
    return new_input_

def new_all_to_all(input_, ranks, src_dim, dst_dim):

    input_list = list(input_.chunk(len(ranks), dim=dst_dim))
    for idx, tensor in enumerate(input_list):
        if not tensor.is_contiguous():
            input_list[idx] = tensor.contiguous()
    new_input_list = [torch.empty_like(t) for t in input_list]
    torch.distributed.all_to_all(new_input_list, input_list, group=get_group(ranks))
    torch.cuda.synchronize()
    new_input_ = torch.concat(tuple(new_input_list), dim=src_dim).requires_grad_()

    return new_input_


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if get_tensor_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())

    return input_


def _split_along_last_dim(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_tensor_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output


def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_tensor_model_parallel_rank()
    dim_offset = rank * local_dim_size
    # return a view of input_
    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()

    return output


def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _reduce_scatter_along_last_dim(input_):
    """Reduce-scatter tensors on the last dimension."""
    num_dims = input_.dim()
    permute_order = (num_dims - 1,) + tuple(range(num_dims - 1))
    input_ = input_.permute(permute_order).contiguous()

    output = _reduce_scatter_along_first_dim(input_)

    permute_order = tuple(range(1, num_dims)) + (0,)
    output = output.permute(permute_order).contiguous()

    return output


def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )

    return output


def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_tensor_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert (
        dim_size[0] % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group()
    )
    return output


def _gather_along_first_dim_moe(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = get_tensor_and_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


def _reduce_scatter_along_first_dim_moe(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    group = get_tensor_and_expert_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0
    dim_size[0] = dim_size[0] // world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), group=group)
    return output


def _gather_along_first_dim_expert_parallel(input_):
    """Gather tensors and concatenate along the first dimension."""
    group = get_expert_model_parallel_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(output, input_.contiguous(), group=group)

    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_

    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)


class _GatherFromSequenceParallelRegionToMOE(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""  # TODO

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_first_dim_moe(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_first_dim_moe(grad_output)


class _ReduceScatterToSequenceParallelRegionFromMOE(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim_moe(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim_moe(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim_moe(grad_output)


class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce_scatter_along_last_dim(grad_output)


class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_last_dim(input_,)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_last_dim(grad_output)


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            # Equal split (all2all)
            output = torch.empty_like(input)
        else:
            # Unequal split (all2all-v)
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
        torch.distributed.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        return (
            None,
            _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------

## For Resharding

def transpose(mat: np.ndarray, dim0: int, dim1: int, get_reverse=False):
    """
    (from Zhiqi's codebase)
    put the dim0 and dim1 of the mat to the last two dims
    """
    ndims = len(mat.shape)
    axes = list(range(ndims))
    assert dim0 < ndims and dim1 < ndims, "dim0 or dim1 out of index"
    axes.pop(max(dim0, dim1))
    axes.pop(min(dim0, dim1))
    axes += [dim0, dim1]

    if get_reverse:
        reverse_axes = []
        for original_index in range(ndims):
            for new_index in axes:
                if axes[new_index] == original_index:
                    reverse_axes.append(new_index)
        return np.transpose(mat, axes), reverse_axes
    else:
        return np.transpose(mat, axes)

def identical_spec(input_spec, required_spec):
    identical = True 
    ## this is used in T5, to pass encoder_output.
    if len(input_spec) == 0 and len(required_spec) == 0:
        return identical

    if input_spec["R"] != required_spec["R"]:
        identical = False
    if input_spec["V"] != required_spec["V"]:
        identical = False    
    for dim_index in range(len(input_spec["dims"])):
        if input_spec["dims"][dim_index] != required_spec["dims"][dim_index]:
            identical = False
    
    return identical

def tensor_adapter_handler(input_dev_mat, init_output_dev_mat, inc_dim, dec_dim, inc_to_size, dec_to_size):
    trans_in_dev_mat = transpose(input_dev_mat, inc_dim, dec_dim)
    trans_out_dev_mat, reverse_axes = transpose(init_output_dev_mat, inc_dim, dec_dim, get_reverse=True)

    for index_r in range(len(trans_in_dev_mat)): 
        for index_v in range(len(trans_in_dev_mat[index_r])): 
            for index_d in range(len(trans_in_dev_mat[index_r][index_v])):
                tmp_arrays = np.hsplit(trans_in_dev_mat[index_r][index_v][index_d], dec_to_size)
                tmp_arrays = [tmp_arrays[i].reshape(inc_to_size, 1) for i in range(len(tmp_arrays))]
                new_mat = np.hstack(tmp_arrays)
                trans_out_dev_mat[index_r][index_v][index_d] = new_mat
    output_dev_mat = trans_out_dev_mat.transpose(reverse_axes)   

    return trans_in_dev_mat, output_dev_mat


def copy_to_tensor_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def reduce_from_tensor_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_, tensor_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)


def gather_from_sequence_parallel_region_to_moe(input_):
    return _GatherFromSequenceParallelRegionToMOE.apply(input_)


def reduce_scatter_to_sequence_parallel_region_from_moe(input_):
    return _ReduceScatterToSequenceParallelRegionFromMOE.apply(input_)


def all_gather_last_dim_from_tensor_parallel_region(input_):
    return _AllGatherFromTensorParallelRegion.apply(input_)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_):
    return _ReduceScatterToTensorParallelRegion.apply(input_)


def all_to_all(group, input_, output_split_sizes_=None, input_split_sizes_=None):
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes_)


def all_to_all_sp2hp(input_):
    world_size = get_tensor_model_parallel_world_size()
    tp_group = get_tensor_model_parallel_group()
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(tp_group, concat_tensor)
    return output


def all_to_all_hp2sp(input_):
    world_size = get_tensor_model_parallel_world_size()
    input_ = input_.reshape(-1, input_.shape[-1])
    tp_group = get_tensor_model_parallel_group()
    input_exchanged = all_to_all(tp_group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped, split_size_or_sections=input_reshaped.shape[0] // world_size, dim=0
    )
    output = torch.cat(split_tensors, dim=-1)
    return output
