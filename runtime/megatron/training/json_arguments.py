'''
    Load arguments from an Aceso json file.
'''

import json

def load_json_args(json_file, args):
    with open(json_file) as f:
        config_dict = json.load(f)
        args.model_name = config_dict["model_name"]
        args.global_batch_size = config_dict["global_batch_size"]
        args.micro_batch_size = config_dict["micro_batch_size"]
        args.num_layers = config_dict["num_layers"]

        if args.model_name in ["gpt"]:
            args.num_attention_heads = config_dict["num_attention_heads"]
            args.hidden_size = config_dict["hidden_size"]
            args.max_position_embeddings = config_dict["max_position_embeddings"]
            args.seq_length = config_dict["seq_length"]
        elif args.model_name in ["resnet"]:
            args.in_channels = config_dict["in_channels"]
            args.width_factor = config_dict["width_factor"]
        elif args.model_name in ["t5"]:
            args.encoder_seq_length = config_dict["encoder_seq_length"]
            args.decoder_seq_length = config_dict["decoder_seq_length"]
            args.seq_length = config_dict["encoder_seq_length"]
            args.max_position_embeddings = config_dict["max_position_embeddings"]
            args.num_attention_heads = config_dict["num_attention_heads"]
            args.kv_channels = config_dict["kv_channels"]
            args.hidden_size = config_dict["hidden_size"]
            args.ffn_hidden_size = config_dict["ffn_hidden_size"]
                
        args.num_ops_in_each_stage = config_dict["num_ops_in_each_stage"]
        args.num_gpus = config_dict["num_gpus"]
        args.num_stages = config_dict["num_stages"]
        args.algo_of_each_op = config_dict["algo_of_each_op"]
        args.tensor_parallel_size_of_each_op = config_dict["tensor_parallel_size_of_each_op"]
        args.data_parallel_size_of_each_op = config_dict["data_parallel_size_of_each_op"]
        args.recompute_ops = config_dict["recompute_ops"]
        args.resharding_stages = config_dict["resharding_stages"]
        args.flex_recompute_activations = config_dict["flex_recompute_activations"]
    return args