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
        args.num_ops_in_each_stage = config_dict["num_ops_in_each_stage"]
        args.recompute_ops = config_dict["recompute_ops"]
        args.tensor_parallel_size_of_each_stage = config_dict["tensor_parallel_size_of_each_stage"]
        args.data_parallel_size_of_each_stage = config_dict["data_parallel_size_of_each_stage"]
        args.context_parallel_size_of_each_stage = config_dict["context_parallel_size_of_each_stage"]
        args.ulysses_context_parallel_size_of_each_stage = config_dict["ulysses_context_parallel_size_of_each_stage"]
        args.ring_context_parallel_size_of_each_stage = config_dict["ring_context_parallel_size_of_each_stage"]
        args.data_parallel_split_of_each_stage = config_dict["data_parallel_split_of_each_stage"]
        args.ring_context_parallel_split_of_each_stage = config_dict["ring_context_parallel_split_of_each_stage"]
        args.ulysses_context_parallel_split_of_each_stage = config_dict["ulysses_context_parallel_split_of_each_stage"]
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
        == len(args.ring_context_parallel_split_of_each_stage)
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
        assert (
            len(args.ring_context_parallel_split_of_each_stage[i])
            == args.ring_context_parallel_size_of_each_stage[i]
        ), f"Ring CP split of stage {i} not equal to Ring CP size"
        if (args.nproc_per_node < args.tensor_parallel_size_of_each_stage[i] * args.ulysses_context_parallel_size_of_each_stage[i]):
            print(f"[Warning] It's a common practice that TP and UCP happen within a node")
        assert args.seq_length == sum(
            args.ring_context_parallel_split_of_each_stage[i]
        ), f"Sequence split by CP of stage {i} not equal to sequence length"
        assert (
            len(args.ulysses_context_parallel_split_of_each_stage[i])
            == args.ring_context_parallel_size_of_each_stage[i]
        ), f"Ulysses CP split of stage {i} not equal to Ring CP size, [notice] Ulysses CP evenly devide the Ring CP size, so just need 1 value to represent ulysses cp size in each stage"
        for j in range(len(args.ring_context_parallel_split_of_each_stage[i])):
            assert (
                args.ring_context_parallel_split_of_each_stage[i][j]
                == args.ulysses_context_parallel_split_of_each_stage[i][j] * args.ulysses_context_parallel_size_of_each_stage[i]
            ), f"Ring CP split of stage {i} not equal to Ulysses CP split"
        if args.transformer_impl != 'transformer_engine' and args.context_parallel_size_of_each_stage[i] != 1:
            raise ValueError(f"Only transformer_engine supports context parallelism > 1")
        sum_ops = 0
        for i in range(len(args.num_gpus)):
            sum_ops += args.num_ops_in_each_stage[i]
        assert sum_ops == args.num_layers * 2 + 2, f"num_ops_in_each_stage should be equal to num_layers + preprocess and postprocess"

