# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""

import os
import warnings
from datetime import timedelta
from typing import Optional
import itertools
import torch
import torch.distributed

from megatron.core.utils import ensure_divisibility

from .utils import GlobalMemoryBuffer


# Intra-layer model parallel group that the current rank belongs to.
_TENSOR_MODEL_PARALLEL_GROUP = None
# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra- and pipeline) that the current rank belongs to.
# TOCHECK: find where this is used
_MODEL_PARALLEL_GROUP = None
# Embedding group.
# TOCHECK: find where this is used
_EMBEDDING_GROUP = None
# Position embedding group.
# TOCHECK: find where this is used
_POSITION_EMBEDDING_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
# tensor model parallel group and data parallel group combined
# used for fp8 and moe training
# TOCHECK: find where this is used
_TENSOR_AND_DATA_PARALLEL_GROUP = None
# Expert parallel group that the current rank belongs to.
# TOCHECK: no ep, so no need for this
_EXPERT_MODEL_PARALLEL_GROUP = None
_TENSOR_AND_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP = None
_DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = None


_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
# TOCHECK: find where this is used
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

# These values enable us to change the mpu sizes on the fly.
_MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_MPU_TENSOR_MODEL_PARALLEL_RANK = None
_MPU_PIPELINE_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None

# A list of ranks that have a copy of the embedding.
# TOCHECK: find where this is used
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
# TOCHECK: find where this is used
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
# TOCHECK: find where this is used
_PIPELINE_GLOBAL_RANKS = None

# A list of global ranks for each data parallel group to ease calculation of the source
# rank when broadcasting weights from src to all other data parallel ranks
# TOCHECK: find where this is used
_DATA_PARALLEL_GLOBAL_RANKS = None

# Context parallel group that the current rank belongs to
_CONTEXT_PARALLEL_GROUP = None
# A list of global ranks for each context parallel group to ease calculation of the
# destination rank when exchanging KV/dKV between context parallel_ranks
_CONTEXT_PARALLEL_GLOBAL_RANKS = None

# Data parallel group information with context parallel combined.
_DATA_PARALLEL_GROUP_WITH_CP = None
_DATA_PARALLEL_GROUP_WITH_CP_GLOO = None
_DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = None

# combined parallel group of TP, DP, and CP used for fp8
_TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None

# Memory buffers to avoid dynamic memory allocation
# TOCHECK: find where this is used
_GLOBAL_MEMORY_BUFFER = None

# MOE logging
_MOE_AUX_LOSSES_LOGGING_TRACKER = {}

# For FlexPipe
_NUM_OPS_IN_EACH_STAGE_LIST =None
_OPS_START_INDEX_LIST = None
_OPS_END_INDEX_LIST = None

_CHILD_RANKS = None
_PARENT_RANKS = None

_FLEXPIPE_PREV_RANKS = None
_FLEXPIPE_NEXT_RANKS = None

_VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK = None

_RANKS_IN_EACH_PIPELINE_STAGE = None

_BWD_SEND_INFO = None
_FWD_RECV_INFO = None
_FWD_SEND_INFO = None
_BWD_RECV_INFO = None

all_groups = {}
_TP_SIZE_PER_OP = None
_DP_SIZE_PER_OP = None

# Currently not support resharding
_RESHARDING_GROUP = None
_RESHARDING_RANK = None
_RESHARDING_DIM = None
_OP_RESHARDING_RANKS = []

_TENSOR_MODEL_PARALLEL_RANKS = None
_DATA_PARALLEL_RANKS = None

def get_nccl_options(pg_name, nccl_comm_cfgs):
    """Set the NCCL process group options.

    Args:
        pg_name (str): process group name
        nccl_comm_cfgs (dict): nccl communicator configurations

    When an option (e.g., max_ctas) is not found in the config, use the NCCL default setting.
    """
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None



def initialize_model_parallel_flexpipe(num_ops_in_each_stage: list[int],
                                       num_layers: int,
                                       num_gpus: list[int],
                                       virtual_pipeline_model_parallel_size: int,
                                       tensor_parallel_size_of_each_op:list[list[int]],
                                       data_parallel_size_of_each_op: list[list[int]], 
                                       micro_batch_size: int
                                       ): 
    """
    Initialize model data parallel groups for FlexPipe.
    Generate _DATA_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP, _TENSOR_MODEL_PARALLEL_GROUP, _PIPELINE_MODEL_PARALLEL_GROUP in this function.
    Because FlexPipe supports different tensor model parallelism size at each pipeline stage,
    this function is quite different from original Megatron.
    """

    num_ops_in_each_stage = num_ops_in_each_stage
    virtual_pipeline_model_parallel_size_ = virtual_pipeline_model_parallel_size

    global _TP_SIZE_PER_OP, _DP_SIZE_PER_OP
    _TP_SIZE_PER_OP = []
    _DP_SIZE_PER_OP = [] 

    for i in range(len(tensor_parallel_size_of_each_op)):
        _TP_SIZE_PER_OP += tensor_parallel_size_of_each_op[i]  
    for i in range(len(data_parallel_size_of_each_op)):
        _DP_SIZE_PER_OP += data_parallel_size_of_each_op[i] 


    if torch.distributed.get_rank() == 0:
        print('> initializing FlexPipe...')

    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size_    

    global _NUM_OPS_IN_EACH_STAGE_LIST
    _NUM_OPS_IN_EACH_STAGE_LIST = list(map(int, num_ops_in_each_stage))

    global _OPS_START_INDEX_LIST
    global _OPS_END_INDEX_LIST
    start_index = 0
    start_index_list = []
    end_index_list = []
    for i in range(len(_NUM_OPS_IN_EACH_STAGE_LIST)):
        start_index_list.append(start_index)
        start_index += _NUM_OPS_IN_EACH_STAGE_LIST[i]
        end_index_list.append(start_index)
    _OPS_START_INDEX_LIST = start_index_list
    _OPS_END_INDEX_LIST = end_index_list

    # Build the data-parallel groups.
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    pipeline_model_parallel_size = len(_NUM_OPS_IN_EACH_STAGE_LIST)
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pipeline_model_parallel_size

    global _DATA_PARALLEL_GROUP, _DATA_PARALLEL_GROUP_GLOO, _DATA_PARALLEL_RANKS
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'   
    _DATA_PARALLEL_GROUP = []
    _DATA_PARALLEL_GROUP_GLOO = []
    # the DP ranks of each op in this PP stage
    _DATA_PARALLEL_RANKS = []
    
    for i in range(pipeline_model_parallel_size):
        start_rank = 0
        for ii in range(0, i):
            STAGE_TP_SIZE = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            STAGE_DP_SIZE = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            start_rank += STAGE_TP_SIZE * STAGE_DP_SIZE
        end_rank = start_rank + _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]] * _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]]
        
        for op_index in range(_OPS_START_INDEX_LIST[i], _OPS_END_INDEX_LIST[i]):
            OP_TP_SIZE = _TP_SIZE_PER_OP[op_index]
            OP_DP_SIZE = _DP_SIZE_PER_OP[op_index]
            for j in range(OP_TP_SIZE):
                ranks = range(start_rank + j, end_rank, OP_TP_SIZE)
                group = get_group(ranks)
                if rank in ranks:
                    _DATA_PARALLEL_GROUP.append(group) 
                    _DATA_PARALLEL_GROUP_GLOO.append(group)
                    _DATA_PARALLEL_RANKS.append(ranks)

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP, _TENSOR_MODEL_PARALLEL_RANKS
    assert _TENSOR_MODEL_PARALLEL_GROUP is None, \
        'tensor model parallel group is already initialized'
    _TENSOR_MODEL_PARALLEL_GROUP = []
    _TENSOR_MODEL_PARALLEL_RANKS = []

    for i in range(pipeline_model_parallel_size):
        start_rank = 0
        for ii in range(i):
            STAGE_TP_SIZE = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            STAGE_DP_SIZE = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[ii]]
            start_rank += STAGE_TP_SIZE * STAGE_DP_SIZE
        for op_index in range(_OPS_START_INDEX_LIST[i], _OPS_END_INDEX_LIST[i]):
            OP_TP_SIZE = _TP_SIZE_PER_OP[op_index]
            OP_DP_SIZE = _DP_SIZE_PER_OP[op_index]
            for j in range(OP_DP_SIZE):
                ranks = range(start_rank + j * OP_TP_SIZE, start_rank + (j+1) * OP_TP_SIZE)
                group = get_group(ranks)
                if rank in ranks:
                    _TENSOR_MODEL_PARALLEL_GROUP.append(group)
                    _TENSOR_MODEL_PARALLEL_RANKS.append(ranks)

    # Build the pipeline model-parallel groups
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    ranks_in_each_pipe_stage = []
    start_rank = 0
    for i in range(pipeline_model_parallel_size):
        STAGE_TP_SIZE = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]]
        STAGE_DP_SIZE = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[i]]
        end_rank = start_rank + STAGE_TP_SIZE * STAGE_DP_SIZE  
        ranks = [j for j in range(start_rank, end_rank)]
        if rank in ranks:
            _MPU_PIPELINE_MODEL_PARALLEL_RANK = i
        ranks_in_each_pipe_stage.append(ranks)
        start_rank = end_rank
    
    # Now only support 1 pipeline, so only first and last layer has embedding and post process op
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    ranks = range(0, pipeline_model_parallel_size)
    # Setup embedding group (to exchange gradients between
    # first and last stages).
    if len(ranks) > 1:
        embedding_ranks = [ranks[0], ranks[-1]]
        position_embedding_ranks = [ranks[0]]
    else:
        embedding_ranks = ranks
        position_embedding_ranks = ranks

    group = get_group(embedding_ranks)
    if rank in embedding_ranks:
        _EMBEDDING_GROUP = group
    _EMBEDDING_GLOBAL_RANKS = embedding_ranks

    group = get_group(position_embedding_ranks)
    if rank in position_embedding_ranks:
        _POSITION_EMBEDDING_GROUP = group
    _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # store child ranks and parent ranks for each rank
    child_ranks = [[] for _ in range(world_size)]
    parent_ranks = [[] for _ in range(world_size)]

    stage_start_rank = 0
    for i in range(pipeline_model_parallel_size):
        if i != (pipeline_model_parallel_size -1):
            next_i = i + 1
        else:
            next_i = 0    
        tp_size = _TP_SIZE_PER_OP[_OPS_END_INDEX_LIST[i]-1]
        dp_size = _DP_SIZE_PER_OP[_OPS_END_INDEX_LIST[i]-1]
        tp_size_next = _TP_SIZE_PER_OP[_OPS_START_INDEX_LIST[next_i]]
        dp_size_next = _DP_SIZE_PER_OP[_OPS_START_INDEX_LIST[next_i]]

        for j in range(len(ranks_in_each_pipe_stage[i])):
            current_rank = ranks_in_each_pipe_stage[i][j]
            dp_id = j // tp_size
            tp_id = j % tp_size

            next_dp_id = [dp_id]
            next_tp_id = [tp_id]

            if tp_size_next > tp_size:
                ensure_divisibility(tp_size_next, tp_size)
                ratio = tp_size_next // tp_size
                next_tp_id = range(tp_id * ratio, (tp_id + 1)*ratio)
            if tp_size_next < tp_size:
                ensure_divisibility(tp_size, tp_size_next)
                ratio = tp_size // tp_size_next
                next_tp_id = [tp_id // ratio]
            if dp_size_next > dp_size:
                ensure_divisibility(dp_size_next, dp_size)
                ratio = dp_size_next // dp_size
                next_dp_id = range(dp_id * ratio, (dp_id + 1)*ratio)
            if dp_size_next < dp_size:
                ensure_divisibility(dp_size, dp_size_next)
                ratio = dp_size // dp_size_next
                next_dp_id = [dp_id // ratio]

            child_rank_list = []
            if next_i != 0:
                next_stage_start_index = stage_start_rank + len(ranks_in_each_pipe_stage[i])
            else:
                next_stage_start_index = 0
            for _dp_id in next_dp_id:
                for _tp_id in next_tp_id:
                    child_rank_list.append(next_stage_start_index + _dp_id * tp_size_next + _tp_id)
            child_ranks[current_rank] = child_rank_list
        
        stage_start_rank += len(ranks_in_each_pipe_stage[i])

    for i in range(pipeline_model_parallel_size):
        for j in range(len(ranks_in_each_pipe_stage[i])):
            current_rank = ranks_in_each_pipe_stage[i][j]
            for child_rank in child_ranks[current_rank]:
                parent_ranks[child_rank].append(current_rank)

    global _CHILD_RANKS
    global _PARENT_RANKS

    _CHILD_RANKS = child_ranks
    _PARENT_RANKS = parent_ranks

    global _FLEXPIPE_PREV_RANKS
    global _FLEXPIPE_NEXT_RANKS

    _FLEXPIPE_PREV_RANKS = parent_ranks[rank]
    _FLEXPIPE_NEXT_RANKS = child_ranks[rank]

    global _RANKS_IN_EACH_PIPELINE_STAGE
    _RANKS_IN_EACH_PIPELINE_STAGE = ranks_in_each_pipe_stage

    global _OP_RESHARDING_RANKS
    _OP_RESHARDING_RANKS = [None for _ in range(sum(_NUM_OPS_IN_EACH_STAGE_LIST))]

    ## fix: workaround for the group issue:
    if world_size >= 2:
        for i in range(0, world_size, 2):
            ranks = range(i, i+2)
            get_group(ranks)

    if world_size >= 4:
        for i in range(0, world_size, 4):
            ranks = range(i, i+4)
            get_group(ranks)    

    print(f'[DEBUG]|rank {torch.distributed.get_rank()}| \
    pipeline_rank= {get_pipeline_model_parallel_rank()} | \
    tp_size= {get_tensor_model_parallel_world_size()} | \
    tp_rank={get_tensor_model_parallel_rank()} | \
    tp_src_rank={get_tensor_model_parallel_src_rank()} | \
    dp_size= {get_data_parallel_world_size()} | \
    parent ranks={get_stage_comm_recv_ranks()} | \
    child ranks = {get_stage_comm_send_ranks()} | \
    micro_batch_size = {micro_batch_size}\n')

    print(f'[DEBUG]|rank {torch.distributed.get_rank()}| \
    MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE: {_MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE} | \
    DATA_PARALLEL_RANKS: {_DATA_PARALLEL_RANKS} | \
    MPU_PIPELINE_MODEL_PARALLEL_RANK: {_MPU_PIPELINE_MODEL_PARALLEL_RANK} | \
    CHILD_RANKS: {_CHILD_RANKS} | \
    PARENT_RANKS: {_PARENT_RANKS} | \
    FLEXPIPE_PREV_RANKS: {_FLEXPIPE_PREV_RANKS} | \
    FLEXPIPE_NEXT_RANKS: {_FLEXPIPE_NEXT_RANKS} | \
    RANKS_IN_EACH_PIPELINE_STAGE: {_RANKS_IN_EACH_PIPELINE_STAGE}| \
    OP_RESHARDING_RANKS: {_OP_RESHARDING_RANKS} | \
    EMBEDDING_GLOBAL_RANKS: {_EMBEDDING_GLOBAL_RANKS} | \
    POSITION_EMBEDDING_GLOBAL_RANKS: {_POSITION_EMBEDDING_GLOBAL_RANKS}')
    
    _set_global_memory_buffer()

# For reference of other parallel groups
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
) -> None:
    """Initialize model data parallel groups.

    Args:
        tensor_model_parallel_size (int, default = 1):
            The number of GPUs to split individual tensors across.

        pipeline_model_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            Transformer layers across. For example, if
            tensor_model_parallel_size is 4 and
            pipeline_model_parallel_size is 2, the model will be split
            into 2 groups of 4 GPUs.

        virtual_pipeline_model_parallel_size (int, optional):
            The number of stages that each pipeline group will have,
            interleaving as necessary. If None, no interleaving is
            performed. For example, if tensor_model_parallel_size is 1,
            pipeline_model_parallel_size is 4,
            virtual_pipeline_model_parallel_size is 2, and there are
            16 transformer layers in the model, the model will be
            split into 8 stages with two layers each and each GPU
            would get 2 stages as such (layer number starting with 1):

            GPU 0: [1, 2] [9, 10]
            GPU 1: [3, 4] [11, 12]
            GPU 2: [5, 6] [13, 14]
            GPU 3: [7, 8] [15, 16]

        pipeline_model_parallel_split_rank (int, optional):
            For models with both an encoder and decoder, the rank in
            pipeline to switch between encoder and decoder (i.e. the
            first rank of the decoder). This allows the user to set
            the pipeline parallel size of the encoder and decoder
            independently. For example, if
            pipeline_model_parallel_size is 8 and
            pipeline_model_parallel_split_rank is 3, then ranks 0-2
            will be the encoder and ranks 3-7 will be the decoder.

        use_sharp (bool, default = False):
            Set the use of SHARP for the collective communications of
            data-parallel process groups. When `True`, run barrier
            within each data-parallel process group, which specifies
            the SHARP application target groups.

        context_parallel_size (int, default = 1):
            The number of tensor parallel GPU groups to split the
            network input sequence length across. Compute of attention
            module requires tokens of full sequence length, so GPUs
            in a context parallel group need to communicate with each
            other to exchange information of other sequence chunks.
            Each GPU and its counterparts in other tensor parallel
            groups compose a context parallel group.

            For example, assume we have 8 GPUs, if tensor model parallel
            size is 4 and context parallel size is 2, the network input
            will be split into two sequence chunks, which are processed
            by 2 different groups of 4 GPUs. One chunk is processed by
            GPU0-3, the other chunk is processed by GPU4-7. Four groups
            are build to do context parallel communications: [GPU0, GPU4],
            [GPU1, GPU5], [GPU2, GPU6], and [GPU3, GPU7].

            Context parallelism partitions sequence length, so it has no
            impact on weights, which means weights are duplicated among
            GPUs in a context parallel group. Hence, weight gradients
            all-reduce is required in backward. For simplicity, we piggyback
            GPUs of context parallelism on data parallel group for
            weight gradient all-reduce.

        nccl_communicator_config_path (str, default = None):
            Path to the yaml file of NCCL communicator configurations.
            `min_ctas`, `max_ctas`, and `cga_cluster_size` can be set
            for each communicator.

        distributed_timeout_minutes (int, default = 30): Timeout, in
            minutes,for operations executed against distributed
            process groups. See PyTorch documentation at
            https://pytorch.org/docs/stable/distributed.html for
            caveats.

    Let's say we have a total of 16 GPUs denoted by g0 ... g15 and we
    use 2 GPUs to parallelize the model tensor, and 4 GPUs to parallelize
    the model pipeline. The present function will
    create 8 tensor model-parallel groups, 4 pipeline model-parallel groups
    and 8 data-parallel groups as:
        8 data_parallel groups:
            [g0, g2], [g1, g3], [g4, g6], [g5, g7], [g8, g10], [g9, g11], [g12, g14], [g13, g15]
        8 tensor model-parallel groups:
            [g0, g1], [g2, g3], [g4, g5], [g6, g7], [g8, g9], [g10, g11], [g12, g13], [g14, g15]
        4 pipeline model-parallel groups:
            [g0, g4, g8, g12], [g1, g5, g9, g13], [g2, g6, g10, g14], [g3, g7, g11, g15]
    Note that for efficiency, the caller should make sure adjacent ranks
    are on the same DGX box. For example if we are using 2 DGX-1 boxes
    with a total of 16 GPUs, rank 0 to 7 belong to the first box and
    ranks 8 to 15 belong to the second box.

    """
    # Get world size and rank. Ensure some consistencies.
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            )

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP
    assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'
    all_data_parallel_group_ranks_with_cp = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(context_parallel_size * tensor_model_parallel_size):
            ranks = range(
                start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
            )
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
            if rank in ranks:
                _DATA_PARALLEL_GROUP = group
                _DATA_PARALLEL_GROUP_GLOO = group_gloo
                _DATA_PARALLEL_GLOBAL_RANKS = ranks
        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
            group_with_cp = torch.distributed.new_group(
                ranks_with_cp, timeout=timeout, pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs)
            )
            group_with_cp_gloo = torch.distributed.new_group(
                ranks_with_cp, timeout=timeout, backend="gloo"
            )
            if rank in ranks_with_cp:
                _DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                _DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Apply SHARP to DP process groups
    if use_sharp:
        if rank == 0:
            print(
                "The number of process groups to use SHARP with depends on the type "
                "of the network switch. Nvidia QM1 switch supports SAHRP up to 8 "
                "process groups and QM2 supports up to 256 process groups. We apply "
                "SHARP to the communications of the data-parallel domain. If the "
                "number of data-parallel process groups is larger than the max "
                "process groups that the network switch supports, the communication "
                "will fall back to non-SHARP operators. To enable SHARP, "
                "`#SBATCH_NETWORK=sharp` should be set in the sbatch script."
            )
        torch.distributed.barrier(
            group=get_data_parallel_group(with_context_parallel=True),
            device_ids=[torch.cuda.current_device()],
        )
        # Set `NCCL_COLLNET_ENABLE=0` to restrict SHARP application to DP process groups
        os.environ["NCCL_COLLNET_ENABLE"] = "0"

    # Build the context-parallel groups.
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    assert _CONTEXT_PARALLEL_GROUP is None, 'context parallel group is already initialized'
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                i * num_pipeline_model_parallel_groups
                + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                i * num_pipeline_model_parallel_groups
                + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('cp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _CONTEXT_PARALLEL_GROUP = group
                    _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # Build the model-parallel groups.
    global _MODEL_PARALLEL_GROUP
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group is already initialized'
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [
            data_parallel_group_ranks_with_cp[i]
            for data_parallel_group_ranks_with_cp in all_data_parallel_group_ranks_with_cp
        ]
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    global _TENSOR_MODEL_PARALLEL_GROUP
    assert (
        _TENSOR_MODEL_PARALLEL_GROUP is None
    ), 'tensor model parallel group is already initialized'
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ), 'pipeline model parallel group is already initialized'
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    assert _EMBEDDING_GROUP is None, 'embedding group is already initialized'
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    assert _POSITION_EMBEDDING_GROUP is None, 'position embedding group is already initialized'
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('pp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            _EMBEDDING_GROUP = group
        if rank in ranks:
            _EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            _POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            _POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    assert (
        _TENSOR_AND_DATA_PARALLEL_GROUP is None
    ), 'Tensor + data parallel group is already initialized'
    tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
    num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
    for i in range(num_tensor_and_data_groups_with_cp):
        start_rank = i * tensor_and_data_group_size_with_cp
        end_rank = start_rank + tensor_and_data_group_size_with_cp
        ranks = range(start_rank, end_rank)
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

        for j in range(context_parallel_size):
            ranks = []
            for k in range(data_parallel_size):
                start_rank = (
                    i * tensor_and_data_group_size_with_cp
                    + j * tensor_model_parallel_size
                    + k * tensor_model_parallel_size * context_parallel_size
                )
                end_rank = start_rank + tensor_model_parallel_size
                ranks = ranks + list(range(start_rank, end_rank))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_dp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    global _EXPERT_MODEL_PARALLEL_GROUP
    assert _EXPERT_MODEL_PARALLEL_GROUP is None, 'Expert parallel group is already initialized'
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'Tensor + expert parallel group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'Data modulo expert group is already initialized'
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
    tensor_and_data_group_size: int = tensor_model_parallel_size * data_parallel_size
    num_tensor_and_data_groups: int = world_size // tensor_and_data_group_size
    tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
    num_expert_groups: int = data_parallel_size // expert_model_parallel_size
    for i in range(num_tensor_and_data_groups):
        for j in range(num_expert_groups):
            # TPxEP Group
            start_rank = i * tensor_and_data_group_size + j * tensor_and_expert_group_size
            end_rank = i * tensor_and_data_group_size + (j + 1) * tensor_and_expert_group_size
            ranks = range(start_rank, end_rank)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
            )
            if rank in ranks:
                _TENSOR_AND_EXPERT_PARALLEL_GROUP = group
            for k in range(tensor_model_parallel_size * context_parallel_size):
                ranks = range(
                    start_rank + k, end_rank, tensor_model_parallel_size * context_parallel_size
                )
                group = torch.distributed.new_group(
                    ranks, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    _EXPERT_MODEL_PARALLEL_GROUP = group

    for i in range(num_tensor_and_data_groups):
        start_rank = i * tensor_and_data_group_size
        end_rank = (i + 1) * tensor_and_data_group_size
        for j in range(tensor_and_expert_group_size):
            ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
            )
            group_gloo = torch.distributed.new_group(ranks, backend="gloo")
            if rank in ranks:
                _DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    _set_global_memory_buffer()

def is_initialized():
    """Useful for code segments that may be accessed with or without mpu initialization"""
    return _DATA_PARALLEL_GROUP is not None


def is_unitialized() -> bool:
    """Check if parallel state has been initialized

    Deprecated. Use is_initialized instead.

    """
    warnings.warn(
        "is_unitialized is deprecated, use is_initialized instead", DeprecationWarning,
    )
    return not is_initialized()


def model_parallel_is_initialized():
    """Check if model and data parallel groups are initialized."""
    if (
        _TENSOR_MODEL_PARALLEL_GROUP is None
        or _PIPELINE_MODEL_PARALLEL_GROUP is None
        or _DATA_PARALLEL_GROUP is None
    ):
        return False
    return True


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is None, 'model parallel group shall not be used'
    return _MODEL_PARALLEL_GROUP

# /// *** modify
# def get_tensor_model_parallel_group(check_initialized=True):
#     """Get the tensor model parallel group the caller rank belongs to."""
#     if check_initialized:
#         assert (
#             _TENSOR_MODEL_PARALLEL_GROUP is not None
#         ), 'tensor model parallel group is not initialized'
    
#     return _TENSOR_MODEL_PARALLEL_GROUP

def get_tensor_model_parallel_group(op_index=None, check_initialized=True):
    """Get the tensor model parallel group the caller rank belongs to."""
    if check_initialized:
        assert (
            _TENSOR_MODEL_PARALLEL_GROUP is not None
        ), 'tensor model parallel group is not initialized'
    if op_index is None:
        op_index = _OPS_START_INDEX_LIST[get_pipeline_model_parallel_rank()]
    return get_tensor_model_parallel_group_via_op_index(op_index)
    



def get_pipeline_model_parallel_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    assert (
        _PIPELINE_MODEL_PARALLEL_GROUP is not None
    ), 'pipeline_model parallel group is not initialized'
    return _PIPELINE_MODEL_PARALLEL_GROUP


# def get_data_parallel_group(with_context_parallel=False):
#     """Get the data parallel group the caller rank belongs to."""
#     if with_context_parallel:
#         assert (
#             _DATA_PARALLEL_GROUP_WITH_CP is not None
#         ), 'data parallel group with context parallel combined is not initialized'
#         return _DATA_PARALLEL_GROUP_WITH_CP
#     else:
#         assert _DATA_PARALLEL_GROUP is not None, 'data parallel group is not initialized'
#         return _DATA_PARALLEL_GROUP
# /// *** modify

def get_data_parallel_group(op_index=None, with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    with_context_parallel = False
    # print('context parallel not implemented, with_context_parallel is set to False')
    assert with_context_parallel == False, 'context parallel not implemented'
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    if op_index is None:
        op_index = _OPS_START_INDEX_LIST[get_pipeline_model_parallel_rank()]
    return get_data_parallel_group_via_op_index(op_index)  



def get_data_parallel_group_gloo(with_context_parallel=False):
    """Get the data parallel group-gloo the caller rank belongs to."""
    with_context_parallel = False
    # print('context parallel not implemented, with_context_parallel is set to False')
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GROUP_WITH_CP_GLOO is not None
        ), 'data parallel group-gloo with context parallel combined is not initialized'
        return _DATA_PARALLEL_GROUP_WITH_CP_GLOO
    else:
        assert _DATA_PARALLEL_GROUP_GLOO is not None, 'data parallel group-gloo is not initialized'
        return _DATA_PARALLEL_GROUP_GLOO


def get_context_parallel_group(check_initialized=True):
    """Get the context parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None, 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GROUP


def get_context_parallel_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    if check_initialized:
        assert (
            _CONTEXT_PARALLEL_GLOBAL_RANKS is not None
        ), 'context parallel group is not initialized'
    return _CONTEXT_PARALLEL_GLOBAL_RANKS


def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    assert _EMBEDDING_GROUP is not None, 'embedding group is not initialized'
    return _EMBEDDING_GROUP


def get_position_embedding_group():
    """Get the position embedding group the caller rank belongs to."""
    assert _POSITION_EMBEDDING_GROUP is not None, 'position embedding group is not initialized'
    return _POSITION_EMBEDDING_GROUP


def get_amax_reduction_group(with_context_parallel=False):
    """Get the FP8 amax reduction group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'FP8 amax reduction group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_tensor_and_data_parallel_group(with_context_parallel=False):
    """Get the tensor and data parallel group the caller rank belongs to."""
    if with_context_parallel:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    else:
        assert (
            _TENSOR_AND_DATA_PARALLEL_GROUP is not None
        ), 'tensor and data parallel group is not initialized'
        return _TENSOR_AND_DATA_PARALLEL_GROUP


def get_expert_model_parallel_group():
    assert (
        _EXPERT_MODEL_PARALLEL_GROUP is None
    ), 'expert model parallel group is not supported'
    return _EXPERT_MODEL_PARALLEL_GROUP


def get_tensor_and_expert_parallel_group():
    assert (
        _TENSOR_AND_EXPERT_PARALLEL_GROUP is None
    ), 'tensor and expert parallel group is not supported'
    return _TENSOR_AND_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP is None
    ), 'data modulo expert parallel group is not supported'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP


def get_data_modulo_expert_parallel_group_gloo():
    assert (
        _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO is None
    ), 'data modulo expert parallel group-gloo is not supported'
    return _DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO


def set_expert_model_parallel_world_size(world_size):
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_tensor_model_parallel_world_size(world_size):
    """Set the tensor model parallel size"""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def set_virtual_pipeline_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_tensor_model_parallel_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_pipeline_model_parallel_group())


def set_expert_model_parallel_rank(rank):
    """Set expert model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def set_tensor_model_parallel_rank(rank):
    """Set tensor model parallel rank."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = rank


def set_pipeline_model_parallel_split_rank(rank):
    """Set pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = rank


def get_tensor_model_parallel_rank():
    """Return my rank for the tensor model parallel group."""
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    if _MPU_TENSOR_MODEL_PARALLEL_RANK is not None:
        return _MPU_TENSOR_MODEL_PARALLEL_RANK
    return torch.distributed.get_rank(group=get_tensor_model_parallel_group())

# /// *** modify
# def get_pipeline_model_parallel_rank():
#     """Return my rank for the pipeline model parallel group."""
#     global _MPU_PIPELINE_MODEL_PARALLEL_RANK
#     if _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None:
#         return _MPU_PIPELINE_MODEL_PARALLEL_RANK
#     return torch.distributed.get_rank(group=get_pipeline_model_parallel_group())
def get_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    assert _MPU_PIPELINE_MODEL_PARALLEL_RANK is not None
    return _MPU_PIPELINE_MODEL_PARALLEL_RANK

def get_pipeline_model_parallel_split_rank():
    """Return pipeline model parallel split rank."""
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    return _PIPELINE_MODEL_PARALLEL_SPLIT_RANK


def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if (
            get_virtual_pipeline_model_parallel_world_size() is not None
            and get_virtual_pipeline_model_parallel_rank() != 0
        ):
            return False
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        virtual_pipeline_model_parallel_world_size = (
            get_virtual_pipeline_model_parallel_world_size()
        )
        if virtual_pipeline_model_parallel_world_size is not None and get_virtual_pipeline_model_parallel_rank() != (
            virtual_pipeline_model_parallel_world_size - 1
        ):
            return False
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)


def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    return False
    rank = torch.distributed.get_rank()
    global _EMBEDDING_GLOBAL_RANKS
    if ignore_virtual:
        return rank in _EMBEDDING_GLOBAL_RANKS
    if rank in _EMBEDDING_GLOBAL_RANKS:
        if rank == _EMBEDDING_GLOBAL_RANKS[0]:
            return is_pipeline_first_stage(ignore_virtual=False)
        elif rank == _EMBEDDING_GLOBAL_RANKS[-1]:
            return is_pipeline_last_stage(ignore_virtual=False)
        else:
            return True
    return False


def is_rank_in_position_embedding_group():
    """Return true if current rank is in position embedding group, False otherwise."""
    return False
    rank = torch.distributed.get_rank()
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    return rank in _POSITION_EMBEDDING_GLOBAL_RANKS


def is_pipeline_stage_before_split(rank=None):
    """Return True if pipeline stage executes encoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank < _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_after_split(rank=None):
    """Return True if pipeline stage executes decoder block for a model
    with both encoder and decoder."""
    if get_pipeline_model_parallel_world_size() == 1:
        return True
    if rank is None:
        rank = get_pipeline_model_parallel_rank()
    global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
    if _PIPELINE_MODEL_PARALLEL_SPLIT_RANK is None:
        return True
    if rank >= _PIPELINE_MODEL_PARALLEL_SPLIT_RANK:
        return True
    return False


def is_pipeline_stage_at_split():
    """Return true if pipeline stage executes decoder block and next
    stage executes encoder block for a model with both encoder and
    decoder."""
    rank = get_pipeline_model_parallel_rank()
    return is_pipeline_stage_before_split(rank) and is_pipeline_stage_after_split(rank + 1)


def get_virtual_pipeline_model_parallel_rank():
    """Return the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK


def set_virtual_pipeline_model_parallel_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = rank


def get_virtual_pipeline_model_parallel_world_size():
    """Return the virtual pipeline-parallel world size."""
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    return _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the tensor model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_tensor_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_src_rank(with_context_parallel=False):
    """Calculate the global rank corresponding to the first local rank
    in the data parallel group."""
    with_context_parallel = False
    print("context parallel is not supported, force to use with_context_parallel=False")
    if with_context_parallel:
        assert (
            _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP is not None
        ), "Data parallel group with context parallel combined is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS_WITH_CP[0]
    else:
        assert _DATA_PARALLEL_GLOBAL_RANKS is not None, "Data parallel group is not initialized"
        return _DATA_PARALLEL_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_first_rank():
    """Return the global rank of the first process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    return _PIPELINE_GLOBAL_RANKS[0]


def get_pipeline_model_parallel_last_rank():
    """Return the global rank of the last process in the pipeline for the
    current tensor parallel group"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    last_rank_local = get_pipeline_model_parallel_world_size() - 1
    return _PIPELINE_GLOBAL_RANKS[last_rank_local]


def get_pipeline_model_parallel_next_rank():
    """Return the global rank that follows the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline + 1) % world_size]


def get_pipeline_model_parallel_prev_rank():
    """Return the global rank that preceeds the caller in the pipeline"""
    assert _PIPELINE_GLOBAL_RANKS is not None, "Pipeline parallel group is not initialized"
    rank_in_pipeline = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return _PIPELINE_GLOBAL_RANKS[(rank_in_pipeline - 1) % world_size]

# /// *** modify
# def get_data_parallel_world_size(with_context_parallel=False):
#     """Return world size for the data parallel group."""
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         return torch.distributed.get_world_size(
#             group=get_data_parallel_group(with_context_parallel=with_context_parallel)
#         )
#     else:
#         return 0
def get_data_parallel_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group(with_context_parallel=with_context_parallel))

def get_data_parallel_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(
            group=get_data_parallel_group(with_context_parallel=with_context_parallel)
        )
    else:
        return 0


def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    else:
        return 0


def get_context_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    else:
        return 0


def get_expert_model_parallel_world_size():
    """Return world size for the expert model parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_world_size
    else:
        return 0


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    if _MPU_EXPERT_MODEL_PARALLEL_RANK:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=get_tensor_and_expert_parallel_group()
        )
        return tensor_and_expert_parallel_rank // get_tensor_model_parallel_world_size()
    else:
        return 0


def get_data_modulo_expert_parallel_rank():
    """Return my rank for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_data_modulo_expert_parallel_group())
    else:
        return 0


def _set_global_memory_buffer():
    """Initialize global buffer"""
    global _GLOBAL_MEMORY_BUFFER
    assert _GLOBAL_MEMORY_BUFFER is None, 'global memory buffer is already initialized'
    _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    assert _GLOBAL_MEMORY_BUFFER is not None, 'global memory buffer is not initialized'
    return _GLOBAL_MEMORY_BUFFER


def destroy_global_memory_buffer():
    """Sets the global memory buffer to None"""
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _TENSOR_MODEL_PARALLEL_GROUP
    _TENSOR_MODEL_PARALLEL_GROUP = None
    global _PIPELINE_MODEL_PARALLEL_GROUP
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP_WITH_CP
    _DATA_PARALLEL_GROUP_WITH_CP = None
    global _CONTEXT_PARALLEL_GROUP
    _CONTEXT_PARALLEL_GROUP = None
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    _CONTEXT_PARALLEL_GLOBAL_RANKS = None
    global _EMBEDDING_GROUP
    _EMBEDDING_GROUP = None
    global _POSITION_EMBEDDING_GROUP
    _POSITION_EMBEDDING_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP
    _TENSOR_AND_DATA_PARALLEL_GROUP = None
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = None
    global _EXPERT_MODEL_PARALLEL_GROUP
    _EXPERT_MODEL_PARALLEL_GROUP = None
    global _TENSOR_AND_EXPERT_PARALLEL_GROUP
    _TENSOR_AND_EXPERT_PARALLEL_GROUP = None
    global _DATA_MODULO_EXPERT_PARALLEL_GROUP
    _DATA_MODULO_EXPERT_PARALLEL_GROUP = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
    global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE
    _MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_TENSOR_MODEL_PARALLEL_RANK
    _MPU_TENSOR_MODEL_PARALLEL_RANK = None
    global _MPU_PIPELINE_MODEL_PARALLEL_RANK
    _MPU_PIPELINE_MODEL_PARALLEL_RANK = None
    global _GLOBAL_MEMORY_BUFFER
    _GLOBAL_MEMORY_BUFFER = None
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = None

def get_stage_comm_recv_ranks():
    assert _FLEXPIPE_PREV_RANKS is not None, \
        "_FLEXPIPE_PREV_RANKS is not initialized"
    return _FLEXPIPE_PREV_RANKS

def get_stage_comm_send_ranks():
    assert _FLEXPIPE_NEXT_RANKS is not None, \
        "_FLEXPIPE_NEXT_RANKS is not initialized"
    return _FLEXPIPE_NEXT_RANKS

def get_op_start_index(rank_in_pipeline, model_chunk_id=0):
    assert _OPS_START_INDEX_LIST is not None, \
        "_OPS_START_INDEX_LIST is not initialized"
    num_pipeline_stages = len(_NUM_OPS_IN_EACH_STAGE_LIST)
    return _OPS_START_INDEX_LIST[rank_in_pipeline + model_chunk_id * num_pipeline_stages]

def get_op_end_index(rank_in_pipeline, model_chunk_id=0):
    assert _OPS_END_INDEX_LIST is not None, \
        "_OPS_END_INDEX_LIST is not initialized"
    num_pipeline_stages = len(_NUM_OPS_IN_EACH_STAGE_LIST)
    return _OPS_END_INDEX_LIST[rank_in_pipeline + model_chunk_id * num_pipeline_stages]

def get_num_ops_list():
    assert _NUM_OPS_IN_EACH_STAGE_LIST is not None, \
        "_NUM_OPS_IN_EACH_STAGE_LIST is not initialized"
    return _NUM_OPS_IN_EACH_STAGE_LIST

def set_virtual_pipeline_next_forward_model_rank(model_chunk_id):
    global _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK = model_chunk_id

def set_virtual_pipeline_next_backward_model_rank(model_chunk_id):
    global _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK = model_chunk_id

def get_virtual_pipeline_next_forward_model_rank():
    if _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK is not None:
        return _VIRTUAL_PIPELINE_NEXT_FORWARD_MODEL_PARALLEL_RANK
    else:
        return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK

def get_virtual_pipeline_next_backward_model_rank():
    if _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK is not None:
        return _VIRTUAL_PIPELINE_NEXT_BACKWARD_MODEL_PARALLEL_RANK
    else:
        return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK

def set_virtual_pipeline_backward_model_parallel_rank(model_chunk_id):
    global _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK
    _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK = model_chunk_id

def get_virtual_pipeline_backward_model_parallel_rank():
    if _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK is not None:
        return _VIRTUAL_PIPELINE_BACKWARD_MODEL_PARALLEL_RANK
    else:
        return _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK

def get_pipeline_rank_via_op_index(op_index):
    global _NUM_OPS_IN_EACH_STAGE_LIST
    sum = 0
    for i in range(len(_NUM_OPS_IN_EACH_STAGE_LIST)):
        sum += _NUM_OPS_IN_EACH_STAGE_LIST[i]
        if sum > op_index:
            return  i % len(_NUM_OPS_IN_EACH_STAGE_LIST)

def get_ranks_via_pipeline_stage(pipeline_stage):
    return _RANKS_IN_EACH_PIPELINE_STAGE[pipeline_stage]

def get_next_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    if is_pipeline_last_stage():
        return 0
    else:
        return get_pipeline_model_parallel_rank() + 1

def get_prev_pipeline_model_parallel_rank():
    """Return my rank for the pipeline model parallel group."""
    if is_pipeline_first_stage():
        return get_pipeline_model_parallel_world_size() - 1
    else:
        return get_pipeline_model_parallel_rank() - 1

def set_comm_info(bwd_send_info, fwd_recv_info, fwd_send_info, bwd_recv_info):
    global _BWD_SEND_INFO, _FWD_RECV_INFO, _FWD_SEND_INFO, _BWD_RECV_INFO
    _BWD_SEND_INFO = bwd_send_info
    _FWD_RECV_INFO = fwd_recv_info
    _FWD_SEND_INFO = fwd_send_info
    _BWD_RECV_INFO = bwd_recv_info

def get_recv_info(forward):
    global _FWD_RECV_INFO, _BWD_RECV_INFO
    if forward:
        return _FWD_RECV_INFO
    else:
        return _BWD_RECV_INFO

def get_send_info(forward):
    global _FWD_SEND_INFO, _BWD_SEND_INFO
    if forward:
        return _FWD_SEND_INFO
    else:
        return _BWD_SEND_INFO

def bitmap(ranks):
    """
    (from Zhiqi's codebase)
    map the rank list to the bit map string
    """
    bits = '0' * torch.distributed.get_world_size()
    for rank in ranks:
        if rank >= len(bits):
            raise ValueError("rank {} out of range ({})".format(rank, len(bits)))
        bits = bits[0:rank] + '1' + bits[rank+1:]
    return bits

def get_group(ranks):
    group_bits = bitmap(ranks)
    if group_bits not in all_groups: 
        all_groups[group_bits] = torch.distributed.new_group(list(ranks))       

    return all_groups[group_bits]

def get_group_gloo(ranks):
    return torch.distributed.new_group(ranks, backend="gloo")

def get_op_tp_size(op_index):
    return _TP_SIZE_PER_OP[op_index]

def get_op_dp_size(op_index):
    assert op_index < len(_DP_SIZE_PER_OP), f"op index {op_index} out of range({len(_DP_SIZE_PER_OP)})."
    return _DP_SIZE_PER_OP[op_index]

'''
Currently not support resharding
'''
def set_resharding_group(devices):
    global _RESHARDING_GROUP
    _RESHARDING_GROUP = devices 

def get_resharding_group():
    global _RESHARDING_GROUP
    assert _RESHARDING_GROUP is not None
    return _RESHARDING_GROUP

def set_resharding_rank(rank):
    global _RESHARDING_RANK
    _RESHARDING_RANK = rank 

def get_resharding_rank():
    global _RESHARDING_RANK
    assert _RESHARDING_RANK is not None
    return _RESHARDING_RANK

def set_resharding_dim(dim):
    global _RESHARDING_DIM
    _RESHARDING_DIM = dim 

def get_resharding_dim():
    global _RESHARDING_DIM
    assert _RESHARDING_DIM is not None
    return _RESHARDING_DIM

def set_op_resharding_ranks(op_index, ranks):
    _OP_RESHARDING_RANKS[op_index] = ranks 

def get_op_resharding_ranks(op_index):
    return _OP_RESHARDING_RANKS[op_index]

def get_data_parallel_group_via_op_index(op_index):
    assert _DATA_PARALLEL_GROUP is not None, \
        'data parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]
    return _DATA_PARALLEL_GROUP[op_index - start_op_index]

def get_data_parallel_ranks():
    assert _DATA_PARALLEL_RANKS is not None, \
        'data parallel group is not initialized'
    return _DATA_PARALLEL_RANKS[0]  

def get_data_parallel_ranks_via_op_index(op_index):
    assert _DATA_PARALLEL_RANKS is not None, \
        'data parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]
    return _DATA_PARALLEL_RANKS[op_index - start_op_index]

def get_tensor_model_parallel_group_via_op_index(op_index):
    assert _TENSOR_MODEL_PARALLEL_GROUP is not None, \
        'tensor model parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]
    return _TENSOR_MODEL_PARALLEL_GROUP[op_index - start_op_index]

def get_tensor_model_parallel_ranks_via_op_index(op_index):
    assert _TENSOR_MODEL_PARALLEL_RANKS is not None, \
        'tensor model parallel group is not initialized'
    pp_stage = get_pipeline_model_parallel_rank()
    start_op_index = _OPS_START_INDEX_LIST[pp_stage]
    return _TENSOR_MODEL_PARALLEL_RANKS[op_index - start_op_index]

def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses,
                                 group=get_data_parallel_group())
    averaged_losses = averaged_losses / \
        torch.distributed.get_world_size(group=get_data_parallel_group())

    return averaged_losses