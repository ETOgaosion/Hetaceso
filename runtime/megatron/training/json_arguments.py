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
        args.flex_recompute_activations = config_dict["flex_recompute_activations"]
        args.resharding_stages = config_dict["resharding_stages"]
        args.num_ops_in_each_stage = config_dict["num_ops_in_each_stage"]
        # args.tensor_parallel_size_of_each_op = config_dict["tensor_parallel_size_of_each_op"]
        # args.data_parallel_size_of_each_op = config_dict["data_parallel_size_of_each_op"]
        args.recompute_ops = config_dict["recompute_ops"]
        args.algo_of_each_op = config_dict["algo_of_each_op"]
        args.tensor_parallel_size_of_each_stage = config_dict["tensor_parallel_size_of_each_stage"]
        args.data_parallel_size_of_each_stage = config_dict["data_parallel_size_of_each_stage"]
        args.context_parallel_size_of_each_stage = config_dict["context_parallel_size_of_each_stage"]
        args.data_parallel_split_of_each_stage = config_dict["data_parallel_split_of_each_stage"]
        args.context_parallel_split_of_each_stage = config_dict["context_parallel_split_of_each_stage"]
    return args

def validate_json_args(args):

    assert (
        len(args.num_gpus)
        == len(args.num_ops_in_each_stage)
        == len(args.tensor_parallel_size_of_each_stage)
        == len(args.data_parallel_size_of_each_stage)
        == len(args.context_parallel_size_of_each_stage)
    ), f"Number of pipeline stages is the same"

    for i in range(len(args.num_gpus)):
        assert (
            args.num_gpus[i]
            == args.tensor_parallel_size_of_each_stage[i]
            * args.data_parallel_size_of_each_stage[i]
            * args.context_parallel_size_of_each_stage[i]
        ), f"GPUs in stage{i} not equal to TP * DP * PP"
        assert (
            len(args.data_parallel_split_of_each_stage[i])
            == args.data_parallel_size_of_each_stage[i]
        ), f"DP split of stage {i} not equal to DP size"
        assert args.micro_batch_size == sum(
            args.data_parallel_split_of_each_stage[i]
        ), f"Data split by DP of stage {i} not equal to mbs"
        assert (
            len(args.context_parallel_split_of_each_stage[i])
            == args.context_parallel_size_of_each_stage[i]
        ), f"CP split of stage {i} not equal to CP size"
        assert args.seq_length == sum(
            args.context_parallel_split_of_each_stage[i]
        ), f"Sequence split by CP of stage {i} not equal to sequence length"

