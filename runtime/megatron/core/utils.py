# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Utility functions used throughout Megatron core"""
import math
import operator
from functools import reduce

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from dataclasses import dataclass, field
from torch.nn.parallel import DistributedDataParallel as torchDDP
@dataclass
class OpConfig:
    name: str
    prev_name: str
    input_tensors_info: dict = field(default_factory=dict)
    output_tensors_info: dict = field(default_factory=dict)
    input_extra_tensors_info: dict = field(default_factory=dict)
    output_extra_tensors_info: dict = field(default_factory=dict)    
    shared_weights_info: dict = field(default_factory=dict)

snapmap = dict()

def debug_mem_report(log_name, path=None, return_string=False):
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''
    
    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        string = ""
        # print('Storage on %s' %(mem_type))
        string += 'Storage on %s\n' %(mem_type)
        # print('-'*LEN)
        string += '-'*LEN + "\n"
        total_numel = 0
        total_mem = 0
        visited_data = []
        string_large = ""
        string_small = ""
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            # print('%s\t\t%s\t\t%.2f' % (element_type, size, mem) )
            if mem > 1:
                string_large += '%s\t\t%s\t\t%.2f\t\t%d\n' % (element_type, size, mem, data_ptr)
            else:
                string_small += '%s\t\t%s\t\t%.2f\t\t%d\n' % (element_type, size, mem, data_ptr)
        # print('-'*LEN)
        string += string_large
        # string += "\n" + string_small
        string += '-'*LEN + "\n"
        # print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        string += 'Total Tensors: %d \tUsed Memory Space: %.2f MBytes\n' % (total_numel, total_mem)
        # print('-'*LEN)
        string += '-'*LEN + "\n"
        return string

    string = ""
    LEN = 65
    string += f"================================== rank {torch.distributed.get_rank()} {log_name} ==================================\n"
    objects = gc.get_objects()
    string += '%s\t%s\t\t\t%s\n' %('Element type', 'Size', 'Used MEM(MBytes)')
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    string += _mem_report(cuda_tensors, 'GPU')
    string += '='*LEN + "\n"
    if path:
        with open(path, "a+") as f:
            f.write(string+"\n")   
    else:
        if return_string:
            return string
        else:
            print(string + "\n")        

def unwrap_model(model, module_instances=(torchDDP)):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


def calc_params_l2_norm(model):
    """Calculate l2 norm of parameters """
    args = get_args()
    if not isinstance(model, list):
        model = [model]
    # Remove duplicate params.
    params_data = []
    for model_ in model:
        for param in model_.parameters():
            is_not_shared = param_is_not_shared(param)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if is_not_shared and is_not_tp_duplicate:
                if args.bf16:
                    params_data.append(param.data.float())
                else:
                    params_data.append(param.data)
    # Calculate norm
    dummy_overflow_buf = torch.cuda.IntTensor([0])
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        dummy_overflow_buf,
        [params_data],
        False # no per-parameter norm
    )
    norm_2 = norm * norm
    # Sum across all model-parallel GPUs.
    torch.distributed.all_reduce(norm_2,
                                 op=torch.distributed.ReduceOp.SUM,
                                 group=mpu.get_model_parallel_group())
    return norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def report_memory(name, get_list=False):
    """Simple GPU memory report."""

    mega_bytes = 1024.0 * 1024.0
    allocated = torch.cuda.memory_allocated() / mega_bytes
    max_allocated = torch.cuda.max_memory_allocated() / mega_bytes
    reserved = torch.cuda.memory_reserved() / mega_bytes
    max_reserved = torch.cuda.max_memory_reserved() / mega_bytes

    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(allocated)
    string += ' | max allocated: {}'.format(max_allocated)
    string += ' | reserved: {}'.format(reserved)
    string += ' | max reserved: {}'.format(max_reserved)

    if get_list:
        mem_to_csv = [["allocated", "max_allocated", "reserved", "max_reserved"], 
            [f"{allocated:.2f}", f"{max_allocated:.2f}", f"{reserved:.2f}", f"{max_reserved:.2f}"]]
        return string, mem_to_csv

    return string

def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def get_attr_wrapped_model(model, attr, allow_none=True, return_model_obj=False):
    """Get an attribute from a wrapped model.
    If return_model_obj is true, return the object that has the 'attr' attribute;
    otherwise, return the attribute directly."""
    if isinstance(model, list):
        raise RuntimeError("_get_attr_wrapped_model given a list of models")

    if allow_none:

        def condition(model, attr):
            return not hasattr(model, attr)

    else:

        def condition(model, attr):
            return getattr(model, attr, None) is None

    while condition(model, attr):
        if not hasattr(model, "module"):
            raise RuntimeError(f"_get_attr_wrapped_model couldn't find attribute {attr}")

        model = model.module

    if return_model_obj:
        return model
    return getattr(model, attr)


def get_model_type(model):
    return get_attr_wrapped_model(model, 'model_type')


def get_model_config(model):
    return get_attr_wrapped_model(model, 'config', allow_none=False)


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len, dtype=dtype, device=torch.cuda.current_device(), requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def _kernel_make_viewless_tensor(inp, requires_grad):
    '''Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    '''
    out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad,)
    out.data = inp.data
    return out


class MakeViewlessTensor(torch.autograd.Function):
    '''
    Autograd function to make a viewless tensor.

    This function should be used in cases where the computation graph needs
    to be propagated, but we only want a viewless tensor (e.g.,
    ParallelTransformer's hidden_states). Call this function by passing
    'keep_graph = True' to 'make_viewless_tensor()'.
    '''

    @staticmethod
    def forward(ctx, inp, requires_grad):
        return _kernel_make_viewless_tensor(inp, requires_grad)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def make_viewless_tensor(inp, requires_grad, keep_graph):
    '''
    Entry-point for creating viewless tensors.

    This method should be used, rather than calling 'MakeViewlessTensor'
    or '_kernel_make_viewless_tensor' directly. This method acts as a
    switch for determining if an autograd function or a regular method
    should be used to create the tensor.
    '''

    # return tensor as-is, if not a 'view'
    if inp._base is None:
        return inp

    # create viewless tensor
    if keep_graph:
        return MakeViewlessTensor.apply(inp, requires_grad)
    else:
        return _kernel_make_viewless_tensor(inp, requires_grad)


def assert_viewless_tensor(tensor, extra_msg=None):
    '''Assert that a tensor is not a view (i.e., its '._base' field is
    not set).'''
    if isinstance(tensor, list):
        [assert_viewless_tensor(t) for t in tensor]
        return tensor
    if not isinstance(tensor, torch.Tensor):
        return tensor
    assert tensor._base is None, (
        "Ensure tensor._base is None before setting tensor.data or storing "
        "tensor to memory buffer. Otherwise, a memory leak will occur (and "
        "likely accumulate over iterations). %s"
    ) % extra_msg
    return tensor


def safely_set_viewless_tensor_data(tensor, new_data_tensor):
    '''Safely set tensor's '.data' field.

    Check first that the tensor is viewless (i.e., '._base' not set). If not,
    raise an exception.
    '''
    assert_viewless_tensor(
        tensor,
        extra_msg="FYI, tensor._base has shape %s, and new_data_tensor has shape %s."
        % ("--" if tensor._base is None else tensor._base.shape, new_data_tensor.shape),
    )
    tensor.data = new_data_tensor


def init_method_normal(sigma):
    """Init method based on N(0, sigma)."""

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method_normal(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def make_tp_sharded_tensor_for_checkpoint(
    tensor, key, tp_axis=0, replica_id=None, prepend_offsets=(), **kwargs
):
    """ Helper for instantiating a ShardedTensor where the `tp_axis` dimension is sharded across TP group.

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """

    prepend_axis_num = len(prepend_offsets)

    if replica_id is None:
        replica_id = (0, 0, parallel_state.get_data_parallel_rank(with_context_parallel=True))

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        (
            tp_axis + prepend_axis_num,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_tensor_model_parallel_world_size(),
        ),
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )


def make_sharded_tensor_for_checkpoint(tensor, key, prepend_offsets=(), replica_id=None, **kwargs):
    """ Helper for instantiating a non-sharded ShardedTensor (replicated across TP and DP group).

    Optionally, can provide offsets which prepend new dimensions to the tensor.
    """

    prepend_axis_num = len(prepend_offsets)

    if replica_id is None:
        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

    return ShardedTensor.from_rank_offsets(
        key,
        tensor,
        *prepend_offsets,
        replica_id=replica_id,
        prepend_axis_num=prepend_axis_num,
        **kwargs,
    )


def prepare_input_tensors_for_wgrad_compute(grad_output, all_gathered_input):

    # Doing gather + slicing during the NeMo forward pass can make this tensor
    # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
    # clones it if it's not contiguous:
    # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
    grad_output = grad_output.contiguous()
    # Convert the tensor shapes to 2D for execution compatibility
    if grad_output.dim() == 3:
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        all_gathered_input = all_gathered_input.view(
            all_gathered_input.shape[0] * all_gathered_input.shape[1], all_gathered_input.shape[2]
        )

    return grad_output, all_gathered_input


def drain_embedding_wgrad_compute(config, embedding_activation_buffer, grad_output_buffer, weight):
    """ Helper for performing embedding wgrad GEMM's during the pipeline drain phase, pipelines the AllGather and GEMM's.

    Should only be used when pipeline model parallelism and gradient accumulation fusion are enabled.
    """

    assert len(embedding_activation_buffer) == len(
        grad_output_buffer
    ), "Length of activation and gradient buffers need to be equal!"

    import fused_weight_gradient_mlp_cuda

    from megatron.core.parallel_state import (
        get_global_memory_buffer,
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_world_size,
    )

    input = embedding_activation_buffer.pop(0)
    world_size = get_tensor_model_parallel_world_size()
    dim_size = list(input.size())
    dim_size[0] = dim_size[0] * world_size

    all_gathered_input = [None, None]
    if config.sequence_parallel:
        all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu_0")
        handle = torch.distributed._all_gather_base(
            all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=False
        )

        all_gathered_input[0] = all_gather_buffer
        all_gather_buffer = None
    else:
        all_gathered_input[0] = input

    input = None

    def wgrad_compute(all_gathered_input, grad_output, weight):

        grad_output, all_gathered_input = prepare_input_tensors_for_wgrad_compute(
            grad_output, all_gathered_input
        )

        if config.gradient_accumulation_fusion:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    all_gathered_input, grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    all_gathered_input, grad_output, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")

    # We have all_gathered_input list acting as a double buffer here,
    # since we are pipelining the AllGather and GEMM,one buffer all gathers
    # the input while the other buffer reads from it for the GEMM. We use i
    # and (i+1) for indexing to enable this double buffering.
    for i in range(len(embedding_activation_buffer)):
        input = embedding_activation_buffer.pop(0)
        if config.sequence_parallel:
            name = "mpu_" + str((i + 1) % 2)
            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, name)
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            all_gathered_input[(i + 1) % 2] = all_gather_buffer
            all_gather_buffer = None
        else:
            all_gathered_input[(i + 1) % 2] = input

        grad_output = grad_output_buffer.pop(0)
        wgrad_compute(all_gathered_input[i % 2], grad_output, weight)
        input, all_gathered_input[i % 2], grad_output = None, None, None

        if config.sequence_parallel:
            handle.wait()

    grad_output = grad_output_buffer.pop(0)
    wgrad_compute(all_gathered_input[1], grad_output, weight)
    input, all_gathered_input[1], grad_output = None, None, None
