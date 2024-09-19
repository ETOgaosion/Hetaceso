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
        args.tensor_parallel_size_of_each_op = config_dict["tensor_parallel_size_of_each_op"]
        args.data_parallel_size_of_each_op = config_dict["data_parallel_size_of_each_op"]
        args.recompute_ops = config_dict["recompute_ops"]
        args.algo_of_each_op = config_dict["algo_of_each_op"]
    return args