
import os
import warnings
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed


def initialize_model_parallel_flexpipe(num_ops_in_each_stage: int,   #[3, 2]
                                       virtual_pipeline_model_parallel_size: int, # 
                                       model_parallel_size_of_each_op:list[int], # [[2, 2, 2], [1, 1]]
                                       data_parallel_size_of_each_op: list[int], # [[1, 1, 1], [2, 2]]
                                       micro_batch_size: int #1
                                       ): 
    """
    Initialize model data parallel groups for FlexPipe.
    Generate _DATA_PARALLEL_GROUP, _MODEL_PARALLEL_GROUP, _TENSOR_MODEL_PARALLEL_GROUP, _PIPELINE_MODEL_PARALLEL_GROUP in this function.
    Because FlexPipe supports different tensor model parallelism size at each pipeline stage,
    this function is quite different from original Megatron.
    """
    # printDebug(1, num_ops_in_each_stage, model_parallel_size_of_each_op, data_parallel_size_of_each_op, micro_batch_size)

    num_ops_in_each_stage = num_ops_in_each_stage
    virtual_pipeline_model_parallel_size_ = virtual_pipeline_model_parallel_size

    global _TP_SIZE_PER_OP, _DP_SIZE_PER_OP
    _TP_SIZE_PER_OP = []
    _DP_SIZE_PER_OP = [] 

    


    for tp_item, dp_item, in zip(model_parallel_size_of_each_op, data_parallel_size_of_each_op):
        _TP_SIZE_PER_OP += tp_item  #[2, 2, 2, 1, 1]
        _DP_SIZE_PER_OP += dp_item  #[1, 1, 1, 2, 2]

    # for i in range(len(model_parallel_size_of_each_op)):
    #     _TP_SIZE_PER_OP += model_parallel_size_of_each_op[i]  
    # for i in range(len(data_parallel_size_of_each_op)):
    #     _DP_SIZE_PER_OP += data_parallel_size_of_each_op[i]  


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
    _NUM_OPS_IN_EACH_STAGE_LIST = list(map(int, num_ops_in_each_stage)) #[3,2]

####################################################
    global _OPS_START_INDEX_LIST
    global _OPS_END_INDEX_LIST
    start_index = 0
    start_index_list = []
    end_index_list = []
    ######???
    for i in range(len(_NUM_OPS_IN_EACH_STAGE_LIST)):
        start_index_list.append(start_index)
        start_index += _NUM_OPS_IN_EACH_STAGE_LIST[i]
        end_index_list.append(start_index)
    _OPS_START_INDEX_LIST = start_index_list  #每个stage 的开始 [0, 3]
    _OPS_END_INDEX_LIST = end_index_list      # 每个stage 的结束 [3, 5]
#######################################################

    global _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
    pipeline_model_parallel_size = len(_NUM_OPS_IN_EACH_STAGE_LIST) 
    _MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pipeline_model_parallel_size #2

    global _DATA_PARALLEL_GROUP, _DATA_PARALLEL_GROUP_GLOO, _DATA_PARALLEL_RANKS
    assert _DATA_PARALLEL_GROUP is None, \
        'data parallel group is already initialized'    

    _DATA_PARALLEL_GROUP = []
    _DATA_PARALLEL_GROUP_GLOO = []
    _DATA_PARALLEL_RANKS = []
   

    # for i in range(pipeline_model_parallel_size):
    #     start_rank = 0
    #     for j in range(0, i):
    #         STAGE_TP_SIZE = _TP_SIZE_PER_OP


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
                group = get_group(ranks)#!!!
                if rank in ranks:
                    _DATA_PARALLEL_GROUP.append(group) 
                    _DATA_PARALLEL_GROUP_GLOO.append(group)
                    _DATA_PARALLEL_RANKS.append(ranks)
    
    printDebug(_DATA_PARALLEL_GROUP, _DATA_PARALLEL_GROUP_GLOO, _DATA_PARALLEL_RANKS)

    # start_rank, end_rank = [2, 4]
    # start_rank, end_rank = [4, 6]

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
    
    _set_global_memory_buffer()