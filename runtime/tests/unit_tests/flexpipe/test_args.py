from megatron.training.arguments import parse_args

import unittest
from unittest.mock import patch
import sys

class TestFlexpipeArgs(unittest.TestCase):

    @patch(
        "sys.argv",
        [
            "script_name",
            "--flexpipe-config", 
            "./gpt_1_3B_4stages.json" 
        ],
    )
    def test_flexpipe_args(self):
        args = parse_args()
        print(f"checkpoint_activations: {args.checkpoint_activations}")
        print(f"resharding_stages: {args.resharding_stages}")
        print(f"num_ops_in_each_stage: {args.num_ops_in_each_stage}")
        print(f"num_gpus: {args.num_gpus}")
        print(f"tensor_parallel_size_of_each_op: {args.tensor_parallel_size_of_each_op}")
        print(f"data_parallel_size_of_each_op: {args.data_parallel_size_of_each_op}")
        print(f"recompute_ops: {args.recompute_ops}")
        print(f"algo_of_each_op: {args.algo_of_each_op}")


if __name__ == '__main__':
    unittest.main()

