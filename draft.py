def initialize_model_parallel_flexpipe(num_ops_in_each_stage: int,   #[3, 2]
                                       virtual_pipeline_model_parallel_size: int, # 
                                       model_parallel_size_of_each_op:list[int], # [[2, 2, 2], [1, 1]]
                                       data_parallel_size_of_each_op: list[int], # [[1, 1, 1], [2, 2]]
                                       micro_batch_size: int #1
                                       ): 
    
    _TP_SIZE_PER_OP = [] #每个OP的TP和DP大小
    _DP_SIZE_PER_OP = []

    world_size = torch.distributed.get_world_size() #GPU数量
    rank = torch.distributed.get_rank() #当前Rank

    _NUM_OPS_IN_EACH_STAGE_LIST = list(map(int, num_ops_in_each_stage)) #[3,2]

    _OPS_START_INDEX_LIST  #[0, 3]  #第一个stage 和最后一个stage的rank
    _OPS_END_INDEX_LIST     #[3, 5] 