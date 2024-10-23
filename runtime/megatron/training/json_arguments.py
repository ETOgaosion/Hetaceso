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

def validate_json_args(args):
    # len(num_gpus) must be equal to num_stages
    assert len(args.num_gpus) == args.num_stages, f"num_gpus should have the same length as num_stages: {len(args.num_gpus)} {args.num_stages}"
    
    assert len(args.flex_recompute_activations) == args.num_stages, f"flex_recompute_activations should have the same length as num_stages: {len(args.flex_recompute_activations)} {args.num_stages}"
    
    assert len(args.resharding_stages) == args.num_stages, f"resharding_stages should have the same length as num_stages: {len(args.resharding_stages)} {args.num_stages}"
    
    assert len(args.num_ops_in_each_stage) == args.num_stages, f"num_ops_in_each_stage should have the same length as num_stages: {len(args.num_ops_in_each_stage)} {args.num_stages}"
    
    num_ops_total = 0
    for num_ops in args.num_ops_in_each_stage:
        num_ops_total += num_ops
    assert num_ops_total == args.num_layers + 2, f"num_ops_in_each_stage should sum to num_layers + 2: {num_ops_total} {args.num_layers}"
    
    # tp and dp in group must be same
    for tp_list in args.tensor_parallel_size_of_each_op:
        base_tp = None
        for tp in tp_list:
            assert tp > 0, f"tensor_parallel_size_of_each_op should be positive: {tp}"
            if base_tp is None:
                base_tp = tp
            else:
                assert tp == base_tp, f"tensor_parallel_size_of_each_op should be the same for all ops in the same stage: {tp} {base_tp}"
    for dp_list in args.data_parallel_size_of_each_op:
        base_dp = None
        for dp in dp_list:
            assert dp > 0, f"data_parallel_size_of_each_op should be positive: {dp}"
            if base_dp is None:
                base_dp = dp
            else:
                assert dp == base_dp, f"data_parallel_size_of_each_op should be the same for all ops in the same stage: {dp} {base_dp}"

    # for each op, dp * tp must equal to num_gpus
    for i in range(args.num_stages):
        tp = args.tensor_parallel_size_of_each_op[i][0]
        dp = args.data_parallel_size_of_each_op[i][0]
        assert tp * dp == args.num_gpus[i], f"tensor_parallel_size_of_each_op * data_parallel_size_of_each_op should be equal to num_gpus: {tp} {dp} {args.num_gpus[i]}"
        assert len(args.tensor_parallel_size_of_each_op[i]) == args.num_ops_in_each_stage[i], f"tensor_parallel_size_of_each_op should have the same length as num_ops_in_each_stage: {len(args.tensor_parallel_size_of_each_op[i])} {args.num_ops_in_each_stage[i]}"
        assert len(args.data_parallel_size_of_each_op[i]) == args.num_ops_in_each_stage[i], f"data_parallel_size_of_each_op should have the same length as num_ops_in_each_stage: {len(args.data_parallel_size_of_each_op[i])} {args.num_ops_in_each_stage[i]}"
        assert len(args.recompute_ops[i]) == args.num_ops_in_each_stage[i], f"recompute_ops should have the same length as num_ops_in_each_stage: {len(args.recompute_ops[i])} {args.num_ops_in_each_stage[i]}"
        assert len(args.algo_of_each_op[i]) == args.num_ops_in_each_stage[i], f"algo_of_each_op should have the same length as num_ops_in_each_stage: {len(args.algo_of_each_op[i])} {args.num_ops_in_each_stage[i]}"