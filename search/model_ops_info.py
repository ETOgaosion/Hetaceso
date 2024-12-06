# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
op_list = None
full_op_list = None
tunable_op_list = None

def get_op_list(args):
    global op_list
    if op_list is None:
        op_list = []
        model_size = args.model_size
        src_data_file = args.profiled_gpt_path + args.model_name + f"_{model_size}_mbs1_seqlen2048_tp1.csv"
        with open(src_data_file) as f:
            src_data = csv.reader(f)
            line_index = 0
            for row in src_data:
                line_index += 1
                if line_index > 1:
                    op_list.append(row[0])   
    return op_list   

def get_full_op_list(args):
    global op_list, full_op_list
    if op_list is None:
        op_list = get_op_list(args)
    if full_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            head_ops = [op_list[0]]
            decoder_layer = op_list[1:3]
            tail_ops = op_list[3:]
            full_op_list = head_ops + decoder_layer * args.num_layers + tail_ops
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return full_op_list

def get_tunable_op_list(args):
    """
    Tunable operators are the oprators which support tensor parallelims, we can tune the tensor-parallelism size.
    """
    global tunable_op_list, op_list
    if tunable_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            tunable_op_list = ["dec-embedding", "dec-self-attention", "dec-mlp", "dec-post-process"]      
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return tunable_op_list

def get_op_spec(op_name, tp_size, dp_size):
    if op_name in ["dec-embedding", "dec-self-attention", "dec-mlp", "dec-post-process"]:
        return {"dims": [1, dp_size, tp_size]}
    else:
        raise RuntimeError(f"op_name {op_name} not supported yet.")



    
