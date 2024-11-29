'''
    Load arguments from an Aceso json file.
'''

import json

def load_json_args(json_file, args):
    with open(json_file) as f:
        config_dict = json.load(f)
        args.num_layers = config_dict["num_layers"]
        args.num_stages = config_dict["num_stages"]
        args.num_gpus = config_dict["num_gpus"]
        args.num_ops_in_each_stage = config_dict["num_ops_in_each_stage"]
        args.tensor_parallel_size_of_each_stage = config_dict["tensor_parallel_size_of_each_stage"]
        args.data_parallel_size_of_each_stage = config_dict["data_parallel_size_of_each_stage"]
        args.context_parallel_size_of_each_stage = config_dict["context_parallel_size_of_each_stage"]
        args.ulysses_context_parallel_size_of_each_stage = config_dict["ulysses_context_parallel_size_of_each_stage"]
        args.ring_context_parallel_size_of_each_stage = config_dict["ring_context_parallel_size_of_each_stage"]
        args.data_parallel_split_of_each_stage = config_dict["data_parallel_split_of_each_stage"]
        args.ulysses_context_parallel_split_of_each_stage = config_dict["ulysses_context_parallel_split_of_each_stage"]
        if config_dict.get("seq_length") is not None:
            args.seq_length = config_dict["seq_length"]
        if config_dict.get("max_position_embeddings") is not None:
            args.max_position_embeddings = config_dict["max_position_embeddings"]
        if config_dict.get("num_attention_head") is not None:
            args.num_attention_heads = config_dict["num_attention_head"]
        if config_dict.get("hidden_size") is not None:
            args.hidden_size = config_dict["hidden_size"]
        if config_dict.get("global_batch_size") is not None:
            args.global_batch_size = config_dict["global_batch_size"]
        if config_dict.get("micro_batch_size") is not None:
            args.micro_batch_size = config_dict["micro_batch_size"]
        if config_dict.get("model_size") is not None:
            args.model_size = config_dict["model_size"]
            if args.model_size == '350M':
                # args.num_layers = 24
                args.num_attention_heads = 16
                args.hidden_size = 1024
            elif args.model_size == '1_3B':
                # args.num_layers = 24
                args.num_attention_heads = 24
                args.hidden_size = 2048
            elif args.model_size == '2_7B':
                # args.num_layers = 32
                args.num_attention_heads = 32
                args.hidden_size = 2560
            elif args.model_size == '6_7B':
                # args.num_layers = 32
                args.num_attention_heads = 32
                args.hidden_size = 4096
            elif args.model_size == '13B':
                # args.num_layers = 40
                args.num_attention_heads = 40
                args.hidden_size = 5140
        if config_dict.get("model_name") is not None:
            args.model_name = config_dict["model_name"]
    return args

def validate_json_args(args):

    assert (
        len(args.num_gpus)
        == len(args.num_ops_in_each_stage)
        == len(args.tensor_parallel_size_of_each_stage)
        == len(args.data_parallel_size_of_each_stage)
        == len(args.context_parallel_size_of_each_stage)
        == len(args.ulysses_context_parallel_size_of_each_stage)
        == len(args.ring_context_parallel_size_of_each_stage)
        == len(args.data_parallel_split_of_each_stage)
        == len(args.ulysses_context_parallel_split_of_each_stage)
    ), f"Number of pipeline stages is the same"

    for i in range(len(args.num_gpus)):
        assert (
            args.num_gpus[i]
            == args.tensor_parallel_size_of_each_stage[i]
            * args.data_parallel_size_of_each_stage[i]
            * args.context_parallel_size_of_each_stage[i]
        ), f"GPUs in stage{i} not equal to TP * DP * CP"
        assert (
            args.context_parallel_size_of_each_stage[i]
            == args.ulysses_context_parallel_size_of_each_stage[i]
            * args.ring_context_parallel_size_of_each_stage[i]
        ), f"CP size of stage {i} not equal to UCP * RCP"
        assert (
            len(args.data_parallel_split_of_each_stage[i])
            == args.data_parallel_size_of_each_stage[i]
        ), f"DP split of stage {i} not equal to DP size"
        assert args.micro_batch_size == sum(
            args.data_parallel_split_of_each_stage[i]
        ), f"Data split by DP of stage {i} not equal to mbs"
        if (args.nproc_per_node < args.tensor_parallel_size_of_each_stage[i] * args.ulysses_context_parallel_size_of_each_stage[i]):
            print(f"[Warning] It's a common practice that TP and UCP happen within a node")
        assert args.tensor_parallel_size_of_each_stage[i] == len(args.ulysses_context_parallel_split_of_each_stage[i]), f'TP size of stage {i} not equal to UCP split size [0]'
        rsp_size = args.ring_context_parallel_size_of_each_stage[i]
        for k in range(len(args.ulysses_context_parallel_split_of_each_stage[i])):
            assert args.ring_context_parallel_size_of_each_stage[i] * args.data_parallel_size_of_each_stage[i] == len(args.ulysses_context_parallel_split_of_each_stage[i][k]), f"Ring CP * DP size of stage {i} not equal to UCP split size [1]"
            for j in range(len(args.ulysses_context_parallel_split_of_each_stage[i][k]) // rsp_size):
                assert args.seq_length == args.ulysses_context_parallel_size_of_each_stage[i] * sum(
                    args.ulysses_context_parallel_split_of_each_stage[i][k][j * rsp_size : (j + 1) * rsp_size]
                ), f"Sequence split by Ulysses CP of stage {i} not equal to sequence length, {j} ring cp group, {args.seq_length}, {sum(args.ulysses_context_parallel_split_of_each_stage[i][k][j * rsp_size : (j + 1) * rsp_size])}"
        if args.transformer_impl != 'transformer_engine' and args.context_parallel_size_of_each_stage[i] != 1:
            raise ValueError(f"Only transformer_engine supports context parallelism > 1")
        sum_ops = 0
        for i in range(len(args.num_gpus)):
            sum_ops += args.num_ops_in_each_stage[i]
        assert sum_ops == args.num_layers * 2 + 2, f"num_ops_in_each_stage should be equal to num_layers + preprocess and postprocess"

