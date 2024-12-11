# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import math
import os

import sys
sys.path.append("../runtime")
from megatron.training.theoretical_memory_usage import report_theoretical_memory

from model_ops_info import get_op_spec, get_op_list
from aceso_utils import *

args = parse_args()

op_list = get_op_list(args)

global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, collective_time
global reserved_fwd, reserved_bwd
global inter_band, intra_band

global num_ops_stage, num_gpu_list, total_mbs, tp_size_list, cp_size_list, usp_size_list, rsp_size_list, dp_size_list, rsp_split_list, dp_split_list

node_rank = args.node_rank
global tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
global cur_tp, cur_cp, cur_usp, cur_rsp, cur_dp
global cur_mbs, cur_seqlen

def calculate_node_rank():
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_mbs, cur_seqlen
    
    global num_ops_stage, num_gpu_list, total_mbs, tp_size_list, cp_size_list, usp_size_list, rsp_size_list, dp_size_list, rsp_split_list, dp_split_list
    global cur_tp, cur_cp, cur_usp, cur_rsp, cur_dp
    
    num_ops_stage, num_gpu_list, total_mbs, tp_size_list, cp_size_list, usp_size_list, rsp_size_list, dp_size_list, rsp_split_list, dp_split_list = config_details(config)
    
    sum_nodes = 0
    for i in range(len(num_gpu_list)):
        if sum_nodes + num_gpu_list[i] > node_rank:
            pp_rank = i
            break
        sum_nodes += num_gpu_list[i]
    node_inner_rank = node_rank - sum_nodes
    tp_rank = node_inner_rank % tp_size_list[pp_rank]
    cp_rank = node_inner_rank // tp_size_list[pp_rank] % cp_size_list[pp_rank]
    usp_rank = node_inner_rank // tp_size_list[pp_rank] % usp_size_list[pp_rank]
    rsp_rank = node_inner_rank // tp_size_list[pp_rank] // usp_size_list[pp_rank] % rsp_size_list[pp_rank]
    dp_rank = node_inner_rank // tp_size_list[pp_rank] // usp_size_list[pp_rank] // rsp_size_list[pp_rank] % dp_size_list[pp_rank]
    cur_mbs = dp_split_list[pp_rank][dp_rank]
    cur_seqlen = rsp_split_list[pp_rank][rsp_rank] // usp_size_list[pp_rank]
    
    cur_tp, cur_cp, cur_usp, cur_rsp, cur_dp = tp_size_list[pp_rank], cp_size_list[pp_rank], usp_size_list[pp_rank], rsp_size_list[pp_rank], dp_size_list[pp_rank]

def read_profiled(
    model_name, model_size, gpt_path, dist_p2p_path, local_p2p_path, local_comm_path
):    
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd
    
    global num_ops_stage, num_gpu_list, total_mbs, tp_size_list, cp_size_list, usp_size_list, rsp_size_list, dp_size_list, rsp_split_list, dp_split_list
    
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_tp, cur_cp, cur_usp, cur_rsp, cur_dp
    global cur_mbs, cur_seqlen

    unique_config_list = []
    unique_config_map = {}
    comm_num_gpus_list_map = {"tp": [], "usp": [], "rsp": [], "dp": []}
    comm_num_gpus_map_map = {"tp": {}, "usp": {}, "rsp": {}, "dp": {}}
    
    unique_config_map[(cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp)] = node_rank
    unique_config_list.append((cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp))

    compute_fwd_time = {}
    compute_bwd_time = {}
    input_size = {}
    output_size = {}
    weights = {}
    activations = {}
    reserved_fwd = {}
    reserved_bwd = {}

    global op_list

    ## T5 22B and 11B share same op.
    if model_name == "t5" and model_size == "22B":
        model_size = "11B"

    for op_name in op_list:
        compute_fwd_time[op_name] = {}
        compute_bwd_time[op_name] = {}
        input_size[op_name] = {}
        output_size[op_name] = {}
        weights[op_name] = {}
        activations[op_name] = {}

        reserved_fwd[op_name] = {}
        reserved_bwd[op_name] = {}
        
        for mbs, seqlen, tp, usp, rsp, dp in unique_config_list:
            if mbs not in compute_fwd_time[op_name]:
                compute_fwd_time[op_name][mbs] = {}
                compute_bwd_time[op_name][mbs] = {}
                input_size[op_name][mbs] = {}
                output_size[op_name][mbs] = {}
                weights[op_name][mbs] = {}
                activations[op_name][mbs] = {}

                reserved_fwd[op_name][mbs] = {}
                reserved_bwd[op_name][mbs] = {}
            if seqlen not in compute_fwd_time[op_name][mbs]:
                compute_fwd_time[op_name][mbs][seqlen] = {}
                compute_bwd_time[op_name][mbs][seqlen] = {}
                input_size[op_name][mbs][seqlen] = {}
                output_size[op_name][mbs][seqlen] = {}
                weights[op_name][mbs][seqlen] = {}
                activations[op_name][mbs][seqlen] = {}

                reserved_fwd[op_name][mbs][seqlen] = {}
                reserved_bwd[op_name][mbs][seqlen] = {}
            if tp not in compute_fwd_time[op_name][mbs][seqlen]:
                compute_fwd_time[op_name][mbs][seqlen][tp] = 1000000
                compute_bwd_time[op_name][mbs][seqlen][tp] = 1000000
                input_size[op_name][mbs][seqlen][tp] = 1000000
                output_size[op_name][mbs][seqlen][tp] = 1000000
                weights[op_name][mbs][seqlen][tp] = 1000000
                activations[op_name][mbs][seqlen][tp] = 1000000

                reserved_fwd[op_name][mbs][seqlen][tp] = 1000000
                reserved_bwd[op_name][mbs][seqlen][tp] = 1000000
            if tp not in comm_num_gpus_map_map["tp"]:
                comm_num_gpus_list_map["tp"].append(tp)
                comm_num_gpus_map_map["tp"][tp] = 1
            if usp not in comm_num_gpus_map_map["usp"]:
                comm_num_gpus_list_map["usp"].append(usp)
                comm_num_gpus_map_map["usp"][usp] = 1
            if rsp not in comm_num_gpus_map_map["rsp"]:
                comm_num_gpus_list_map["rsp"].append(rsp)
                comm_num_gpus_map_map["rsp"][rsp] = 1
            if dp not in comm_num_gpus_map_map["dp"]:
                comm_num_gpus_list_map["dp"].append(dp)
                comm_num_gpus_map_map["dp"][dp] = 1

    for mbs, seqlen, tp, usp, rsp, dp in unique_config_list:
        src_data_file = (
            gpt_path + model_name + f"_{model_size}_mbs{mbs}_seqlen{seqlen}_tp{tp}.csv"
        )
        print(src_data_file)
        try:
            with open(src_data_file) as f:
                src_data = csv.reader(f)
                line_index = 0
                for row in src_data:
                    line_index += 1
                    if line_index > 1:
                        op_name = row[0]
                        compute_fwd_time[op_name][mbs][seqlen][tp] = float(
                            row[1]
                        )
                        compute_bwd_time[op_name][mbs][seqlen][tp] = float(
                            row[2]
                        )
                        input_size[op_name][mbs][seqlen][tp] = float(row[3])
                        output_size[op_name][mbs][seqlen][tp] = float(row[4])
                        weights[op_name][mbs][seqlen][tp] = float(row[5])
                        activations[op_name][mbs][seqlen][tp] = float(row[6])

                        if args.consider_reserved_space:
                            reserved_fwd[op_name][mbs][seqlen][tp] = float(
                                row[7]
                            )
                            reserved_bwd[op_name][mbs][seqlen][tp] = float(
                                row[8]
                            )
        except:
            print(
                f"file ({src_data_file}) not exist, or the file is not formatted as expected."
            )
    
    '''
    Communications in Megatron:
    
    - tensor parallel: 4 all-reduce (self-attention 2, mlp 2)
    - ulysses context parallel: an inevitable all-to-all communication
    - ring context parallel: all ring rank need p2p communication, like all-reduce but can possibly overlap with computation
    - data parallel: all ranks need all-reduce gradients
    '''
    global collective_time
    collective_time = {"all_reduce": {}, "all_gather": {}, "reduce_scatter": {}, "all_to_all": {}}
    comm_prim_map = {"tp": ["all_reduce"], "usp": ["all_to_all"], "rsp": ["all_reduce"], "dp": ["all_reduce"]}
    for parallel in comm_prim_map.keys():
        for prim in comm_prim_map[parallel]:
            for num_gpus in comm_num_gpus_list_map[parallel]:
                if num_gpus < 2:
                    continue
                if num_gpus not in collective_time[prim]:
                    collective_time[prim][num_gpus] = {}
                else:
                    continue
                src_data_file = (
                    local_comm_path
                    + f"prim_{model_name}_{model_size}_{prim}_{num_gpus}gpus.csv"
                )
                with open(src_data_file) as f:
                    src_data = csv.reader(f)
                    line_index = 0
                    for row in src_data:
                        line_index += 1
                        if line_index > 1:
                            data_size = row[0]
                            collective_time[prim][num_gpus][data_size] = float(row[1])

    global inter_band, intra_band
    inter_band_file = dist_p2p_path + "p2p_inter_node.csv"
    intra_band_file = local_p2p_path + "p2p_intra_node.csv"
    try:
        with open(intra_band_file) as f:
            src_data = csv.reader(f)
            for idx, row in enumerate(src_data):
                if idx == 1:
                    intra_band = [float(row[i]) for i in range(len(row))]
    except:
        print(f"intra-node bandwidth file is not found.")
    try:
        with open(inter_band_file) as f:
            src_data = csv.reader(f)
            for idx, row in enumerate(src_data):
                if idx == 1:
                    inter_band = [float(row[i]) for i in range(len(row))]
    except:
        print(
            f"inter-node bandwidth file is not found, using intra-node bandwidth instead."
        )
        inter_band = intra_band

    return len(op_list)


def identical_spec(input_spec, required_spec):
    identical = True
    if input_spec is None or required_spec is None:
        return identical

    for dim_index in range(len(input_spec["dims"])):
        if input_spec["dims"][dim_index] != required_spec["dims"][dim_index]:
            identical = False

    return identical

def intra_node_band(data_size):
    global intra_band
    if data_size > 0:
        index = int(math.log(data_size, 2))
        if index >= 1:
            index -= 1
        if index >= len(intra_band):
            return intra_band[-1] * 0.001
        else:
            return intra_band[index] * 0.001
    else:
        return 1


def inter_node_band(data_size):
    global inter_band
    if data_size > 0:
        index = int(math.log(data_size, 2))
        if index >= 1:
            index -= 1
        if index >= len(inter_band):
            return inter_band[-1] * 0.001
        else:
            return inter_band[index] * 0.001
    else:
        return 1

def get_time_v3(ops, mbs, tp, cp, usp, rsp, dp, rsp_split, dp_split, in_cross_node, out_cross_node):
    if len(ops) == 0:
        return 0, 0, 0, 0, 0
    global compute_fwd_time, compute_bwd_time, input_size, output_size
    fwd_comp, bwd_comp, in_comm, out_comm, tp_comm, usp_comm, rsp_comm, dp_comm = 0, 0, 0, 0, 0, 0, 0, 0
    
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_mbs, cur_seqlen
    global cur_tp, cur_cp, cur_usp, cur_rsp, cur_dp
    
    '''
    TP communication refer to https://www.cnblogs.com/rossiXYZ/p/15871062.html
    DP communication refer to https://www.cnblogs.com/rossiXYZ/p/15868988.html
    CP according https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html
    '''

    for i in range(len(ops)):
        op_name = ops[i]
        fwd_comp += compute_fwd_time[op_name][cur_mbs][cur_seqlen][tp]
        bwd_comp += compute_bwd_time[op_name][cur_mbs][cur_seqlen][tp]
        cur_op_input_size = str(int(input_size[op_name][cur_mbs][cur_seqlen][tp]))
        cur_op_output_size = str(int(output_size[op_name][cur_mbs][cur_seqlen][tp]))
        if op_name == "dec-embedding":
            '''
            Embedding layer need all-reduce output 
            runtime/megatron/core/tensor_parallel/layers.py: 228, VocabParallelEmbedding::forward
            '''
            if tp > 1:
                assert cur_op_output_size in collective_time["all_reduce"][tp], f'{op_name} {cur_op_output_size}'
                tp_comm += collective_time["all_reduce"][tp][cur_op_output_size]
        elif op_name == "dec-post-process":
            '''
            TP: In theory like above
            '''
            if tp > 1:
                assert cur_op_input_size in collective_time["all_reduce"][tp], f'{op_name} {cur_op_input_size}'
                tp_comm += collective_time["all_reduce"][tp][cur_op_input_size]
            '''
            DP: Need to allreduce gradients
            - Grad Buffer Async and Overlappable: runtime/megatron/core/distributed/param_and_grad_buffer.py: 140, Bucket::start_gradient_sync
            - Model Grad not overlappable: runtime/megatron/core/distributed/finalize_model_grads.py
            '''
            if dp > 1:
                assert cur_op_input_size in collective_time["all_reduce"][dp], f'{op_name} {cur_op_input_size}'
                dp_comm += collective_time["all_reduce"][dp][cur_op_input_size]
        elif op_name == "dec-self-attention":
            '''
            Self attention
            - QKV need 3 ColumnParallelLinear layers, thus forward: 1 all-gather, backward: 1 all-reduce
            runtime/megatron/core/tensor_parallel/layers.py: 826, ColumnParallelLinear::forward
            - Dropout need 1 RowParallelLinear layer, thus forward: 1 all-reduce, backward: 1 all-gather
            '''
            if tp > 1:
                assert cur_op_output_size in collective_time["all_gather"][tp], f'{op_name} {cur_op_output_size}'
                assert cur_op_output_size in collective_time["all_reduce"][tp], f'{op_name} {cur_op_output_size}'
                tp_comm += (collective_time["all_gather"][tp][cur_op_output_size] + collective_time["all_reduce"][tp][cur_op_output_size]) * 4
            '''
            CP:
            - USP: In TE's implementation, USP QKV communication can overlap with each other, thus only need to consider 1 all-to-all
            - RSP: In most case rsp can overlap with calculation
            '''
            if usp > 1:
                assert cur_op_output_size in collective_time["all_to_all"][usp], f'{op_name} {cur_op_output_size}'
                cp_comm += collective_time["all_to_all"][usp][str(output_size[op_name][cur_mbs][cur_seqlen][tp])]
        elif op_name == "dec-mlp":
            '''
            MLP
            MLP need 1 ColumnParallelLinear, 1 RowParallelLinear
            '''
            if tp > 1:
                assert cur_op_output_size in collective_time["all_gather"][tp], f'{op_name} {cur_op_output_size}'
                assert cur_op_output_size in collective_time["all_reduce"][tp], f'{op_name} {cur_op_output_size}'
                tp_comm += (collective_time["all_gather"][tp][cur_op_output_size] + collective_time["all_reduce"][tp][cur_op_output_size]) * 2
        else:
            raise RuntimeError(f"unknown op_name {op_name}")

    input_comm_size = input_size[ops[0]][cur_mbs][cur_seqlen][tp]
    output_comm_size = output_size[ops[-1]][cur_mbs][cur_seqlen][tp]

    if in_cross_node:
        in_comm = input_comm_size / inter_node_band(input_comm_size)
    else:
        in_comm = input_comm_size / intra_node_band(input_comm_size)

    if out_cross_node:
        out_comm = output_comm_size / inter_node_band(output_comm_size)
    else:
        out_comm = output_comm_size / intra_node_band(output_comm_size)

    return fwd_comp, bwd_comp, in_comm, out_comm, tp_comm, usp_comm, rsp_comm, dp_comm


def get_memory_v3(ops, mbs, tp, cp, usp, rsp, dp, rsp_split, dp_split):
    global input_size, output_size, weights
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_mbs, cur_seqlen
    
    inputs = input_size[ops[0]][cur_mbs][cur_seqlen][tp]
    _activations = 0
    _weights = 0
    for i in range(len(ops)):
        # TODO: Be more precisely
        if args.consider_shared_space and ops[i] == "dec-self-attention":
            _activations += activations[ops[i]][cur_mbs][cur_seqlen][tp] * 1.5
        else:
            _activations += activations[ops[i]][cur_mbs][cur_seqlen][tp]
        _weights += weights[ops[i]][cur_mbs][cur_seqlen][tp]

    return _weights, inputs, _activations


def get_activations_v3(ops, mbs, tp, cp, usp, rsp, dp, rsp_split, dp_split):
    if len(ops) <= 1:
        return 0
    
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    cur_mbs = dp_split[dp_rank]
    cur_seqlen = rsp_split[rsp_rank] // usp

    global activations
    saved_activations = 0
    for i in range(len(ops) - 1):
        # TODO: Be more precisely
        if args.consider_shared_space and ops[i] == "dec-self-attention":
            saved_activations += (
                activations[ops[i]][cur_mbs][cur_seqlen][tp] * 1.5
            )
        else:
            saved_activations += activations[ops[i]][cur_mbs][cur_seqlen][tp]

    return saved_activations


def get_peak_activations(ops, mbs, tp, cp, usp, rsp, dp, rsp_split, dp_split):
    if len(ops) <= 1:
        return 0
    
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_mbs, cur_seqlen

    global activations
    saved_activations = 0
    saved_activations_list = [0]

    for i in range(len(ops) - 1):
        # TODO: Be more precisely
        if args.consider_shared_space and ops[i] == "dec-self-attention":
            saved_activations += (
                activations[ops[i]][cur_mbs][cur_seqlen][tp] * 1.5
            )
            saved_activations_list.append(saved_activations)
            saved_activations = 0
        else:
            if saved_activations > 0:
                saved_activations_list.append(saved_activations)
                saved_activations = 0

    return max(saved_activations_list)


def get_reserved_memory(ops, mbs, tp, cp, usp, rsp, dp, rsp_split, dp_split, memory_weights):
    global reserved_fwd, reserved_bwd
    current_reserved_fwd = 0
    current_reserved_bwd = 0
    
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    cur_mbs = dp_split[dp_rank]
    cur_seqlen = rsp_split[rsp_rank] // usp
    for i in range(len(ops) - 1):
        if reserved_fwd[ops[i]][cur_mbs][cur_seqlen][tp] > current_reserved_fwd:
            current_reserved_fwd = reserved_fwd[ops[i]][cur_mbs][cur_seqlen][tp]
        if reserved_bwd[ops[i]][cur_mbs][cur_seqlen][tp] > current_reserved_bwd:
            current_reserved_bwd = reserved_bwd[ops[i]][cur_mbs][cur_seqlen][tp]

    max_collective = 0

    if args.memory_pred_type == "MAX":
        return (
            max(current_reserved_fwd + current_reserved_bwd, memory_weights)
            + max_collective
        )
    elif args.memory_pred_type == "MIN":
        return max(
            current_reserved_fwd, current_reserved_bwd, memory_weights, max_collective
        )
    else:
        raise RuntimeError(f"unknown args.memory_pred_type {args.memory_pred_type}")


def get_activation_size(op_name, mbs, tp, cp, usp, rsp, dp, rsp_split, dp_split):
    global activations
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_mbs, cur_seqlen
    
    return activations[op_name][cur_mbs][cur_seqlen][tp]


def predict_stage_time(
    ops,
    tp_size,
    cp_size,
    usp_size,
    rsp_size,
    dp_size,
    rsp_split,
    dp_split,
    delta=False,
):
    in_cross_node = False
    out_cross_node = False
    
    global total_mbs

    fwd_comp, bwd_comp, in_comm, out_comm, tp_comm, usp_comm, rsp_comm, dp_comm = get_time_v3(
        ops, total_mbs, tp_size, cp_size, usp_size, rsp_size, dp_size, rsp_split, dp_split, in_cross_node, out_cross_node
    )
    sum_time = fwd_comp + bwd_comp + in_comm + out_comm + tp_comm + usp_comm + rsp_comm + dp_comm

    return sum_time / 1000


def predict_stage_memory(
    ops,
    tp_size,
    cp_size,
    usp_size,
    rsp_size,
    dp_size,
    rsp_split,
    dp_split,
    num_stages_behind,
    breakdown=False,
):
    global total_mbs

    memory_weights, inputs, activations = get_memory_v3(
        ops, total_mbs, tp_size, cp_size, usp_size, rsp_size, dp_size, rsp_split, dp_split
    )
    memory_gradients = memory_weights
    memory_main_params = memory_weights * args.memory_main_params
    memory_optimizer = memory_weights * args.memory_optimizer

    saved_activations = get_activations_v3(
        ops, total_mbs, tp_size, cp_size, usp_size, rsp_size, dp_size, rsp_split, dp_split
    )
    peak_activations = get_peak_activations(
        ops, total_mbs, tp_size, cp_size, usp_size, rsp_size, dp_size, rsp_split, dp_split
    )

    if args.consider_reserved_space:
        memory_reserved = get_reserved_memory(
        ops, total_mbs, tp_size, cp_size, usp_size, rsp_size, dp_size, rsp_split, dp_split, memory_weights
        )
    else:
        memory_reserved = 0

    memory_activations = (inputs + activations - saved_activations) * (
        num_stages_behind
    )
    print(f'inputs: {inputs}, activations: {activations}, saved_activations: {saved_activations}')
    memory_peak = inputs + activations - saved_activations + peak_activations

    memory_weights += memory_main_params
    memory_sum = (
        memory_weights
        + memory_gradients
        + memory_optimizer
        + memory_activations
        + memory_peak
        + memory_reserved
    )

    if breakdown:
        return (
            memory_weights,
            memory_gradients,
            memory_optimizer,
            memory_activations,
            memory_peak,
            memory_reserved,
        )
    else:
        return memory_sum


def predict_time_breakdown(config, print_time=False, print_memory=False):
    global total_mbs, cur_mbs
    
    base_batch_size = config.micro_bs
    global_batch_size = config.global_bs
    num_batches = global_batch_size // base_batch_size

    _time_list = []
    memory_list = []
    compute_time_list = []
    efficiency_list = []
    gpu_time_list = []
    breakdown_ideal_time_per_gpu_list = []

    breakdown_pure_comp_time_list = []
    breakdown_pure_eff_loss_time_list = []

    memory_result_strings = []
    megatron_memory_result_strings = []
    time_result_strings = []
    
    theoretical_memory_lists = []

    num_gpus_till_now = 0
    for i in range(config.num_stages):
        stage = config.stages[i]
        ops = stage.ops
        num_gpus = stage.num_gpus
        tp_size = stage.tp_size
        cp_size = stage.cp_size
        usp_size = stage.usp_size
        rsp_size = stage.rsp_size
        dp_size = stage.dp_size
        rsp_split = stage.rsp_split
        dp_split = stage.dp_split
        num_stages_behind = stage.num_stages_behind

        in_cross_node = (
            num_gpus_till_now % args.num_gpus_per_node
        ) == 0 and num_gpus_till_now > 0
        num_gpus_till_now += num_gpus
        out_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0

        ## compute actual time of each stage
        fwd_comp, bwd_comp, in_comm, out_comm, tp_comm, usp_comm, rsp_comm, dp_comm = get_time_v3(
            ops, total_mbs, tp_size, cp_size, usp_size, rsp_size, dp_size, rsp_split, dp_split, in_cross_node, out_cross_node
        )
        sum_time = (fwd_comp + bwd_comp + in_comm + out_comm + tp_comm + usp_comm + rsp_comm + dp_comm) / 1000
        _time_list.append(sum_time)
        compute_time_list.append((fwd_comp + bwd_comp) / 1000)
        gpu_time_list.append(sum_time * num_gpus)

        if print_time:
            time_result_strings.append(
                "[stage {}], {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, ".format(
                    i,
                    fwd_comp / 1000 * num_batches,
                    (bwd_comp) / 1000 * num_batches,
                    in_comm / 1000 * num_batches,
                    out_comm / 1000 * num_batches,
                    tp_comm / 1000 * num_batches,
                    usp_comm / 1000 * num_batches,
                    rsp_comm / 1000 * num_batches,
                    dp_comm / 1000 * num_batches,
                )
            )

        ## compute ideal time of each stage
        _tp_size = 1
        _cp_size = 1
        _usp_size = 1
        _rsp_size = 1
        _dp_size = 1
        _rsp_split = [config.total_seqlen]
        _dp_split = [config.micro_bs]
        _fwd_comp, _bwd_comp, _in_comm, _out_comm, _tp_comm, _usp_comm, _rsp_comm, _dp_comm = get_time_v3(
            ops, total_mbs, _tp_size, _cp_size, _usp_size, rsp_size, _dp_size, _rsp_split, _dp_split, in_cross_node, out_cross_node
        )
        ideal_time = (_fwd_comp + _bwd_comp + _in_comm + _out_comm + _tp_comm + _usp_comm + _rsp_comm + _dp_comm) / 1000

        ## calculate time breakdown at sum of GPUs
        eff_loss_time = (fwd_comp + bwd_comp) - (_fwd_comp + _bwd_comp) / num_gpus

        ## calculate time breakdown per GPU
        breakdown_ideal_time_per_gpu_list.append(
            ((_fwd_comp + _bwd_comp) / num_gpus) / 1000
        )
        breakdown_pure_eff_loss_time_list.append(eff_loss_time / 1000)

        ## compute memory
        (
            memory_weights,
            memory_gradients,
            memory_optimizer,
            memory_activations,
            memory_peak,
            memory_reserved,
        ) = predict_stage_memory(
            ops,
            tp_size,
            cp_size,
            usp_size,
            rsp_size,
            dp_size,
            rsp_split,
            dp_split,
            num_stages_behind,
            breakdown=True,
        )
        memory_sum = (
            memory_weights
            + memory_gradients
            + memory_optimizer
            + memory_activations
            + memory_peak
            + memory_reserved
        )
        memory_list.append(memory_sum)
        
        args.data_parallel_size = dp_size
        args.micro_batch_size = cur_mbs
        weight_and_optimizer_memory, activation_memory, total_memory = report_theoretical_memory(args, cur_mbs)
        theoretical_memory_lists.append([weight_and_optimizer_memory, activation_memory, total_memory])

        if print_memory:
            memory_result_strings.append(
                f"[stage {i}] memory = {memory_sum:.2f} MB. weights = {memory_weights:.0f}, gradients = {memory_gradients:.0f}, optimizer = {memory_optimizer:.0f}, activations = {memory_activations:.0f}, peak += {memory_peak:.0f}, memory_reserved = {memory_reserved:.0f}"
            )
            megatron_memory_result_strings.append(
                f'[stage {i} reference] total_memory = {total_memory}, weight_and_optimizer_memory = {weight_and_optimizer_memory}, activation_memory = {activation_memory}'
            )

        efficiency_list.append(ideal_time / (sum_time * num_gpus))

    sum_stage_time = sum(_time_list)
    time_list = []
    max_time = 0
    bottleneck = 0
    for i in range(config.num_stages):
        time_stage = _time_list[i] * (num_batches - 1) + sum_stage_time
        time_list.append(time_stage)
        if print_time:
            time_result_strings[i] += f"{time_stage:.2f}"
            if time_stage > max_time:
                max_time = time_stage
                bottleneck = i
    if print_time:
        time_result_strings[bottleneck] = " * " + time_result_strings[bottleneck]
        print("overall time = {:.2f} ms".format(max_time))
        print(
            "stage, fwd_comp, bwd_comp, in_comm(+reshard), out_comm(+reshard), tp_comm, usp_comm, rsp_comm, dp_comm, reshard, sum(us)"
        )
        for i in range(config.num_stages):
            print(time_result_strings[i])

    config.time_list = time_list
    config.memory_list = memory_list
    config.compute_time_list = compute_time_list
    config.total_gpu_time = sum(gpu_time_list) * (num_batches - 1)
    config.breakdown_ideal_time_per_gpu = breakdown_ideal_time_per_gpu_list
    config.breakdown_eff_loss_time_per_gpu = breakdown_pure_eff_loss_time_list

    max_time = max(time_list)
    max_mem = args.memory_limit
    efficient_time_list = []
    for i in range(config.num_stages):
        used_time = time_list[i]
        used_memory = memory_list[i]
        idle_time = 0
        # TODO: not understand
        idle_time = (max_time - used_time) / 2
        efficient_time_list.append(
            idle_time * config.stages[i].num_gpus * efficiency_list[i]
        )
    config.efficient_time_list = efficient_time_list

    if print_memory:
        max_memory = 0
        bottleneck = 0
        for i in range(config.num_stages):
            if (memory_list[i]) > max_memory:
                max_memory = memory_list[i]
                bottleneck = i
        memory_result_strings[bottleneck] = " * " + memory_result_strings[bottleneck]
        megatron_memory_result_strings[bottleneck] = " * " + megatron_memory_result_strings[bottleneck]
        print("\nmax allocated memory = {:.2f} MB".format(max_memory))
        for i in range(config.num_stages):
            print(memory_result_strings[i])
            print(megatron_memory_result_strings[i])
        print(" ")

    return


def get_reserved_memory_list(config):
    reserved_mem_list = []
    if config is not None:
        base_batch_size = config.micro_bs
        for i in range(config.num_stages):
            stage = config.stages[i]
            ops = stage.ops
            tp_size = stage.tp_size
            cp_size = stage.cp_size
            usp_size = stage.usp_size
            rsp_size = stage.rsp_size
            dp_size = stage.dp_size
            rsp_split = stage.rsp_split
            dp_split = stage.dp_split
            num_stages_behind = stage.num_stages_behind

            _, _, _, _, _, reserved_mem = predict_stage_memory(
                ops,
                tp_size,
                cp_size,
                usp_size,
                rsp_size,
                dp_size,
                rsp_split,
                dp_split,
                num_stages_behind,
                breakdown=True,
            )
            
            reserved_mem_list.append(reserved_mem)
    return reserved_mem_list


######## recomputation-related functions #########

stage_memory_set = {}
stage_memory_visit = 0
stage_memory_hit = 0


def predict_stage_memory_helper(
    config,
    stage_index,
    ops=None,
    tp_size=None,
    cp_size=None,
    usp_size=None,
    rsp_size=None,
    dp_size=None,
    rsp_split=None,
    dp_split=None,
    num_stages_behind=None,
):
    global stage_memory_visit, stage_memory_hit, stage_memory_set
    if ops is None:
        ops = config.stages[stage_index].ops
        tp_size = config.stages[stage_index].tp_size
        cp_size = config.stages[stage_index].cp_size
        usp_size = config.stages[stage_index].usp_size
        rsp_size = config.stages[stage_index].rsp_size
        dp_size = config.stages[stage_index].dp_size
        rsp_split = config.stages[stage_index].rsp_split
        dp_split = config.stages[stage_index].dp_split
        num_stages_behind = config.stages[stage_index].num_stages_behind

    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_mbs, cur_seqlen

    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}cp{cp_size}usp{usp_size}rsp{rsp_size}dp{dp_size}seqlen{cur_seqlen}bs{cur_mbs}stage{num_stages_behind}"
    stage_memory_visit += 1
    if stage_memory_set.get(config_str) is not None:
        stage_memory_hit += 1
        return stage_memory_set[config_str]

    pred_memory = predict_stage_memory(
        ops,
        tp_size,
        cp_size,
        usp_size,
        rsp_size,
        dp_size,
        rsp_split,
        dp_split,
        num_stages_behind,
    )
    stage_memory_set[config_str] = pred_memory

    return pred_memory


stage_time_set = {}
stage_time_visit = 0
stage_time_hit = 0


def predict_stage_time_helper(config, stage_index):
    global stage_time_visit, stage_time_hit, stage_time_set
    ops = config.stages[stage_index].ops
    tp_size = config.stages[stage_index].tp_size
    cp_size = config.stages[stage_index].cp_size
    usp_size = config.stages[stage_index].usp_size
    rsp_size = config.stages[stage_index].rsp_size
    dp_size = config.stages[stage_index].dp_size
    rsp_split = config.stages[stage_index].rsp_split
    dp_split = config.stages[stage_index].dp_split
    
    global node_rank, tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank
    global cur_mbs, cur_seqlen

    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}cp{cp_size}usp{usp_size}rsp{rsp_size}dp{dp_size}seqlen{cur_seqlen}bs{cur_mbs}"
    stage_time_visit += 1
    if stage_time_set.get(config_str) is not None:
        stage_time_hit += 1
        return stage_time_set[config_str]

    pred_time = predict_stage_time(
        ops, tp_size, cp_size, usp_size, rsp_size, dp_size, rsp_split, dp_split
    )
    stage_time_set[config_str] = pred_time

    return pred_time

if __name__ == "__main__":
    config, config_dict = read_config_from_json(args, return_config_dict=True)
    args = config_to_args(config, config_dict, args)
    calculate_node_rank()
    read_profiled(
        config_dict["model_name"], config_dict["model_size"], args.profiled_gpt_path, args.profiled_dist_p2p_path, args.profiled_local_p2p_path, args.profiled_local_comm_path
    )
    predict_time_breakdown(config, print_time=True, print_memory=True)
    if args.save_to_csv is not None:
        save_config_info_to_csv(
            config, get_reserved_memory_list(config), args.save_to_csv
        )

    # print(f"---- testing model ----")
    # from hetaceso_cost_model import HetacesoPerfModel

    # test_perf_model = HetacesoPerfModel(
    #     config,
    #     args.node_rank,
    #     cur_mbs,
    #     cur_seqlen,
    #     args.profiled_gpt_path,
    #     args.profiled_local_p2p_path,
    #     args.profiled_local_comm_path,
    #     args.profiled_dist_p2p_path,
    #     config_dict["model_name"],
    #     config_dict["model_size"],
    #     args.num_gpus_per_node,
    #     args.dist_optimizer,
    #     "1000Mbps"
    # )
    # test_perf_model.predict_config_performance(config, print_detail=True)
    # print(
    #     f"time list = {list(map(int, config.time_list))}\nmemory list = {list(map(int, config.memory_list))}"
    # )