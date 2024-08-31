# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""

import random
import os
import time

import numpy as np
import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from datetime import timedelta

from megatron.legacy import fused_kernels
from megatron.training import get_adlr_autoresume
from megatron.training import get_args
from megatron.training import get_tensorboard_writer
from megatron.core import mpu, tensor_parallel
from megatron.training.arguments import parse_args, validate_args
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.legacy.model.transformer import bias_dropout_add_fused_train
from megatron.legacy.model.fused_bias_gelu import bias_gelu

from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.legacy.model import Float16Module
from megatron.core.pipeline_parallel.p2p_communication import (
    send_shared_tensors,
    recv_shared_tensors,
)

from megatron.training.utils import unwrap_model

ENABLE_WEIGHT_SHARE = os.environ.get("ENABLE_WEIGHT_SHARE", '1') == '1'

def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        assert torch.cuda.is_available(), "Megatron requires CUDA."

    # Parse arguments
    args = parse_args(extra_args_provider, ignore_unknown_args)

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        assert args.load is not None, "--use-checkpoints-args requires --load argument"
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)


    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        # TODO is this still a necessary option?
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
           _initialize_tp_communicators()

        # No continuation function
        return None


def _compile_dependencies():

    args = get_args()

    # =========================
    # Compile dataset C++ code.
    # =========================
    # TODO: move this to ninja
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling dataset index builder ...")
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            ">>> done with dataset index builder. Compilation time: {:.3f} "
            "seconds".format(time.time() - start_time),
            flush=True,
        )

    # ==================
    # Load fused kernels
    # ==================

    # [NOT SURE] seems like no need to check
    # # Custom kernel constraints check.
    # seq_len = args.seq_length
    # attn_batch_size = (
    #     args.num_attention_heads / args.tensor_model_parallel_size
    # ) * args.micro_batch_size
    # # Constraints on sequence length and attn_batch_size to enable warp based
    # # optimization and upper triangular optimization (for causal mask)
    # custom_kernel_constraint = (
    #     seq_len > 16
    #     and seq_len <= 16384
    #     and seq_len % 4 == 0
    #     and attn_batch_size % 4 == 0
    # )
    # # Print a warning.
    # if not (
    #     (args.fp16 or args.bf16)
    #     and custom_kernel_constraint
    #     and args.masked_softmax_fusion
    # ):
    #     if args.rank == 0:
    #         print(
    #             "WARNING: constraints for invoking optimized"
    #             " fused softmax kernel are not met. We default"
    #             " back to unfused kernel invocations.",
    #             flush=True,
    #         )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        fused_kernels.load(args)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(args)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )

def _initialize_tp_communicators():
    """ initializing the communicators with user buffers for high-performance tensor-model-parallel 
        communication overlap """

    try:
       import yaml

       import transformer_engine
       from transformer_engine.pytorch import module as te_module

    except ImportError:
       raise RuntimeError("Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
             "'transformer_engine' packages") 

    args = get_args()

    if args.tp_comm_overlap_cfg is not None:
       with open(args.tp_comm_overlap_cfg,"r") as stream:    
          ub_cfgs = yaml.safe_load(stream)
    else:
       ub_cfgs = {}

    input_shape = [(args.seq_length * args.micro_batch_size) // args.context_parallel_size , args.hidden_size]

    #We create a MPI process group, which is needed to bootstrap the pipelined 
    #tensor-model-parallel communication overlap
    torch.distributed.new_group(backend='mpi')

    te_module.base.initialize_ub(shape = input_shape, tp_size = args.tensor_model_parallel_size, 
                                 use_fp8 = (args.fp8 is not None) , ub_cfgs = ub_cfgs,)

def _initialize_distributed():
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():

        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:

        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            device = args.rank % device_count
            if args.local_rank is not None:
                assert (
                    args.local_rank == device
                ), "expected local-rank to be the same as rank % device-count."
            else:
                args.local_rank = device
            torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            if args.flexpipe:
                mpu.initialize_model_parallel_flexpipe(
                    args.num_ops_in_each_stage, 
                    args.virtual_pipeline_model_parallel_size, 
                    args.model_parallel_size_of_each_op,
                    args.data_parallel_size_of_each_op,
                    args.micro_batch_size 
                )
            else:
                raise NotImplementedError("Only FlexPipe is supported for now")
                mpu.initialize_model_parallel(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                    context_parallel_size=args.context_parallel_size,
                    expert_model_parallel_size=args.expert_model_parallel_size,
                    distributed_timeout_minutes=args.distributed_timeout_minutes,
                    nccl_communicator_config_path=args.nccl_communicator_config_path,
                )
            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )


def _init_autoresume():
    """Set autoresume start time."""
    autoresume = get_adlr_autoresume()
    if autoresume:
        torch.distributed.barrier()
        autoresume.init()
        torch.distributed.barrier()


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed))


def write_args_to_tensorboard():
    """Write arguments to tensorboard."""
    args = get_args()
    writer = get_tensorboard_writer()
    if writer:
        for arg in vars(args):
            writer.add_text(arg, str(getattr(args, arg)), global_step=args.iteration)


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    if (TORCH_MAJOR > 1) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)

    _warmup_jit_function()


def _warmup_jit_function():
    """Compilie JIT functions before the main training steps"""
    args = get_args()
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup fused bias+gelu
    bias = torch.rand(
        args.ffn_hidden_size // args.tensor_model_parallel_size,
        dtype=dtype,
        device="cuda",
    )
    input = torch.rand(
        (
            args.seq_length,
            args.micro_batch_size,
            args.ffn_hidden_size // args.tensor_model_parallel_size,
        ),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, input.requires_grad = bias_grad, input_grad
        for _ in range(5):
            output = bias_gelu(bias, input)
    del bias, input, output

    # Warmup fused bias+dropout+add
    if args.sequence_parallel:
        seq_length = args.seq_length // mpu.get_tensor_model_parallel_world_size()
    else:
        seq_length = args.seq_length
    input = torch.rand(
        (seq_length, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    residual = torch.rand(
        (seq_length, args.micro_batch_size, args.hidden_size),
        dtype=dtype,
        device="cuda",
    )
    bias = torch.rand((args.hidden_size), dtype=dtype, device="cuda").expand_as(
        residual
    )
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip(
        [False, True], [True, True], [True, True]
    ):
        input.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train(input, bias, residual, dropout_rate)
    del bias, input, residual, output
    torch.cuda.empty_cache()

def initialize_weights_sharing(models):
    if ENABLE_WEIGHT_SHARE:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
        virtual_pipeline_rank = mpu.get_virtual_pipeline_model_parallel_rank()    
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
    if ENABLE_WEIGHT_SHARE:
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