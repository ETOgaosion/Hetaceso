# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import math
import os
from model_ops_info import get_op_spec, get_op_list
from aceso_utils import *

args = parse_args()

op_list = get_op_list(args)

math_log_2 = {
    1: int(math.log(1, 2)),
    2: int(math.log(2, 2)),
    4: int(math.log(4, 2)),
    8: int(math.log(8, 2)),
    16: int(math.log(16, 2)),
}
global_mbs_index = None

global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, collective_time
global reserved_fwd, reserved_bwd
global inter_band, intra_band

global max_tp_size


def get_mbs_index(mbs):
    global global_mbs_index
    assert global_mbs_index is not None
    return global_mbs_index[mbs]


def read_profiled(
    model_name, model_size, gpt_path, dist_p2p_path, local_p2p_path, local_comm_path
):
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index

    mbs_list = args.micro_batch_size
    global_mbs_index = {}
    for i in range(len(mbs_list)):
        global_mbs_index[mbs_list[i]] = i
        
    global max_tp_size

    if (model_name == "gpt" and model_size == "350M") or (
        model_name == "t5" and model_size == "220M"
    ):
        max_tp_size = min(args.max_tp, 4)
    else:
        max_tp_size = min(args.max_tp, 8)

    tp_size_list = []
    tp = 1
    while tp <= max_tp_size:
        tp_size_list.append(tp)
        tp *= 2
    comm_num_gpus_list = tp_size_list[1:]

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
        compute_fwd_time[op_name] = []
        compute_bwd_time[op_name] = []
        input_size[op_name] = []
        output_size[op_name] = []
        weights[op_name] = []
        activations[op_name] = []

        reserved_fwd[op_name] = []
        reserved_bwd[op_name] = []

        for i in range(len(mbs_list)):
            compute_fwd_time[op_name].append([])
            compute_bwd_time[op_name].append([])
            input_size[op_name].append([])
            output_size[op_name].append([])
            weights[op_name].append([])
            activations[op_name].append([])

            reserved_fwd[op_name].append([])
            reserved_bwd[op_name].append([])

            for j in range(len(tp_size_list)):
                compute_fwd_time[op_name][i].append(1000000)
                compute_bwd_time[op_name][i].append(1000000)
                input_size[op_name][i].append(1000000)
                output_size[op_name][i].append(1000000)
                weights[op_name][i].append(1000000)
                activations[op_name][i].append(1000000)

                reserved_fwd[op_name][i].append(1000000)
                reserved_bwd[op_name][i].append(1000000)

    for mbs in mbs_list:
        for tp in tp_size_list:
            mbs_index = get_mbs_index(mbs)
            tp_index = int(math.log(tp, 2))
            if model_name == "scale-layer":
                src_data_file = gpt_path + f"gpt_scale-layer_mbs{mbs}_tp{tp}.csv"
            else:
                src_data_file = (
                    gpt_path + model_name + f"_{model_size}_mbs{mbs}_tp{tp}.csv"
                )
            try:
                with open(src_data_file) as f:
                    src_data = csv.reader(f)
                    line_index = 0
                    for row in src_data:
                        line_index += 1
                        if line_index > 1:
                            op_name = row[0]
                            compute_fwd_time[op_name][mbs_index][tp_index] = float(
                                row[1]
                            )
                            compute_bwd_time[op_name][mbs_index][tp_index] = float(
                                row[2]
                            )
                            input_size[op_name][mbs_index][tp_index] = float(row[3])
                            output_size[op_name][mbs_index][tp_index] = float(row[4])
                            weights[op_name][mbs_index][tp_index] = float(row[5])
                            activations[op_name][mbs_index][tp_index] = float(row[6])

                            if args.consider_reserved_space:
                                reserved_fwd[op_name][mbs_index][tp_index] = float(
                                    row[7]
                                )
                                reserved_bwd[op_name][mbs_index][tp_index] = float(
                                    row[8]
                                )
            except:
                print(
                    f"file ({src_data_file}) not exist, or the file is not formatted as expected."
                )
    global collective_time
    collective_time = {}
    if model_name in ["gpt", "scale-layer"]:
        prim_list = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
    elif model_name in ["t5"]:
        prim_list = []
    elif model_name in ["resnet"]:
        prim_list = ["all_gather", "all_to_all"]
    for prim in prim_list:
        collective_time[prim] = {}
        for num_gpus in comm_num_gpus_list:
            collective_time[prim][num_gpus] = {}
            if model_name == "scale-layer":
                src_data_file = (
                    local_comm_path + f"prim_gpt_scale-layer_{prim}_{num_gpus}gpus.csv"
                )
            else:
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


def get_reshard_primitives(input_spec, required_spec):
    if identical_spec(input_spec, required_spec):
        return None, None, 0

    for src_dim_index in range(len(input_spec["dims"])):
        if input_spec["dims"][src_dim_index] > required_spec["dims"][src_dim_index]:
            ## D -> R, all-gather
            if input_spec["R"] < required_spec["R"]:
                assert (
                    input_spec["dims"][src_dim_index]
                    % required_spec["dims"][src_dim_index]
                    == 0
                )
                num_devices = (
                    input_spec["dims"][src_dim_index]
                    // required_spec["dims"][src_dim_index]
                )

                return "all_gather", "split", num_devices

            for dst_dim_index in range(len(input_spec["dims"])):
                ## D -> D, all-to-all
                if (
                    dst_dim_index != src_dim_index
                    and input_spec["dims"][dst_dim_index]
                    < required_spec["dims"][dst_dim_index]
                ):
                    assert (
                        input_spec["dims"][src_dim_index]
                        % required_spec["dims"][src_dim_index]
                        == 0
                    )
                    num_devices = (
                        input_spec["dims"][src_dim_index]
                        // required_spec["dims"][src_dim_index]
                    )

                    return "all_to_all", "all_to_all", num_devices


def get_reshard_time(prim, num_devices, data_size):
    if num_devices <= 1:
        return 0
    if prim in ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]:
        _data_size = "{:.0f}".format(float(data_size))
        if _data_size in collective_time[prim][num_devices]:
            return collective_time[prim][num_devices][_data_size]
        elif (
            "{:.0f}".format(float(data_size) - 1) in collective_time[prim][num_devices]
        ):
            return collective_time[prim][num_devices][
                "{:.0f}".format(float(data_size) - 1)
            ]
        else:
            return 100000
    elif prim in ["split"]:
        return 0
    else:
        return 100000


def get_reshard_memory(prim, num_devices, data_size):
    assert num_devices > 1
    if prim == "all_reduce":
        return data_size
    elif prim == "all_gather":
        return data_size * num_devices
    elif prim == "reduce_scatter":
        return data_size
    elif prim == "split":
        return data_size
    elif prim == "all_to_all":
        return data_size * num_devices


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

def get_stage_of_op(op_index):
    if op_index == 0:
        return 0
    else:
        return (op_index - 1) / 2

def get_time_v3(ops, mbs, tp, dp, in_cross_node, out_cross_node):
    if len(ops) == 0:
        return 0, 0, 0, 0, 0
    global compute_fwd_time, compute_bwd_time, input_size, output_size
    fwd_comp, bwd_comp, in_comm, out_comm, tp_comm = 0, 0, 0, 0, 0

    for i in range(len(ops)):
        op_name = ops[i]
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp, 2))
        fwd_comp += compute_fwd_time[op_name][mbs_index][tp_index]
        bwd_comp += compute_bwd_time[op_name][mbs_index][tp_index]
        if args.support_comm_predict:
            # TODO: Check correctness, QKV/dense/GEMM ops need reshard time
            for op_name_suffix in ["attention", "mlp"]:
                if op_name_suffix in op_name:
                    tp_comm += (
                        2
                        * get_reshard_time(
                            "all_reduce",
                            tp,
                            output_size[op_name][mbs_index][tp_index],
                        )
                        * 1000
                    )

    in_mbs_index = get_mbs_index(mbs[0])
    in_tp_index = int(math.log(tp, 2))
    out_mbs_index = get_mbs_index(mbs[-1])
    out_tp_index = int(math.log(tp, 2))
    input_comm_size = input_size[ops[0]][in_mbs_index][in_tp_index]
    output_comm_size = output_size[ops[-1]][out_mbs_index][out_tp_index]

    if in_cross_node:
        in_comm = input_comm_size / inter_node_band(input_comm_size)
    else:
        in_comm = input_comm_size / intra_node_band(input_comm_size)

    if out_cross_node:
        out_comm = output_comm_size / inter_node_band(output_comm_size)
    else:
        out_comm = output_comm_size / intra_node_band(output_comm_size)

    fwd_reshard = 0
    bwd_reshard = 0
    if args.resharding:
        for i in range(1, len(ops)):
            prev_spec = get_op_spec(ops[i - 1], tp, dp)
            current_spec = get_op_spec(ops[i], tp, dp)
            fwd_prim, bwd_prim, num_devices = get_reshard_primitives(
                prev_spec, current_spec
            )
            mbs_index = get_mbs_index(mbs[i])
            tp_index = int(math.log(tp, 2))
            if fwd_prim is not None:
                fwd_reshard += get_reshard_time(
                    fwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index]
                )
            if bwd_prim is not None:
                bwd_reshard += get_reshard_time(
                    bwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index]
                )

    in_comm += fwd_reshard * 1000
    out_comm += bwd_reshard * 1000

    tp_comm += fwd_reshard * 1000 + bwd_reshard * 1000
    return fwd_comp, bwd_comp, in_comm, out_comm, tp_comm


def get_memory_v3(ops, mbs, tp):
    global input_size, output_size, weights
    in_mbs_index = get_mbs_index(mbs[0])
    in_tp_index = int(math.log(tp, 2))
    inputs = input_size[ops[0]][in_mbs_index][in_tp_index]
    _activations = 0
    _weights = 0
    for i in range(len(ops)):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp, 2))
        # TODO: Be more precisely
        if args.consider_shared_space and ops[i] == "dec-self-attention":
            _activations += activations[ops[i]][mbs_index][tp_index] * 1.5
        else:
            _activations += activations[ops[i]][mbs_index][tp_index]
        _weights += weights[ops[i]][mbs_index][tp_index]

    return _weights, inputs, _activations


def get_activations_v3(ops, mbs, tp):

    if len(ops) <= 1:
        return 0

    global activations
    saved_activations = 0
    for i in range(len(ops) - 1):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp, 2))
        # TODO: Be more precisely
        if args.consider_shared_space and ops[i] == "dec-self-attention":
            saved_activations += (
                activations[ops[i]][mbs_index][tp_index] * 1.5
            )
        else:
            saved_activations += activations[ops[i]][mbs_index][tp_index]

    return saved_activations


def get_peak_activations(ops, mbs, tp):

    if len(ops) <= 1:
        return 0

    global activations
    saved_activations = 0
    saved_activations_list = [0]

    for i in range(len(ops) - 1):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp, 2))
        # TODO: Be more precisely
        if args.consider_shared_space and ops[i] == "dec-self-attention":
            saved_activations += (
                activations[ops[i]][mbs_index][tp_index] * 1.5
            )
            saved_activations_list.append(saved_activations)
            saved_activations = 0
        else:
            if saved_activations > 0:
                saved_activations_list.append(saved_activations)
                saved_activations = 0

    return max(saved_activations_list)


def get_reserved_memory(ops, mbs, tp, dp, memory_weights):
    global reserved_fwd, reserved_bwd
    current_reserved_fwd = 0
    current_reserved_bwd = 0
    for i in range(len(ops) - 1):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = int(math.log(tp, 2))
        if reserved_fwd[ops[i]][mbs_index][tp_index] > current_reserved_fwd:
            current_reserved_fwd = reserved_fwd[ops[i]][mbs_index][tp_index]
        if reserved_bwd[ops[i]][mbs_index][tp_index] > current_reserved_bwd:
            current_reserved_bwd = reserved_bwd[ops[i]][mbs_index][tp_index]

    max_collective = 0
    if args.consider_collective_memory:
        if args.resharding:
            for i in range(1, len(ops)):
                prev_spec = get_op_spec(
                    ops[i - 1], tp, dp, input_spec=False
                )
                current_spec = get_op_spec(
                    ops[i], tp, dp, input_spec=True
                )
                fwd_prim, bwd_prim, num_devices = get_reshard_primitives(
                    prev_spec, current_spec
                )
                mbs_index = get_mbs_index(mbs[i])
                tp_index = int(math.log(tp, 2))
                if fwd_prim is not None:
                    fwd_collective = get_reshard_memory(
                        fwd_prim,
                        num_devices,
                        input_size[ops[i]][mbs_index][tp_index],
                    )
                    if fwd_collective > max_collective:
                        max_collective = fwd_collective
                if bwd_prim is not None:
                    bwd_collective = get_reshard_memory(
                        bwd_prim,
                        num_devices,
                        input_size[ops[i]][mbs_index][tp_index],
                    )
                    if bwd_collective > max_collective:
                        max_collective = bwd_collective

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


def get_activation_size(op_name, mbs, tp):
    global activations
    mbs_index = get_mbs_index(mbs)
    tp_index = math_log_2[tp]
    return activations[op_name][mbs_index][tp_index]


def predict_stage_time(
    ops,
    tp_size,
    dp_size,
    base_batch_size,
    delta=False,
    on_the_right=False,
    decrease=True,
):
    in_cross_node = False
    out_cross_node = False
    mbs_list = [base_batch_size // dp_size for j in range(len(ops))]

    fwd_comp, bwd_comp, in_comm, out_comm, _ = get_time_v3(
        ops, mbs_list, tp_size, dp_size, in_cross_node, out_cross_node
    )
    if not delta:
        sum_time = fwd_comp + bwd_comp + in_comm + out_comm
    else:
        if on_the_right and decrease:
            sum_time = fwd_comp + bwd_comp - in_comm + out_comm
        elif not on_the_right and decrease:
            sum_time = fwd_comp + bwd_comp + in_comm - out_comm
        elif on_the_right and not decrease:
            sum_time = fwd_comp + bwd_comp + in_comm - out_comm
        elif not on_the_right and not decrease:
            sum_time = fwd_comp + bwd_comp - in_comm + out_comm

    return sum_time / 1000


def predict_stage_memory(
    ops,
    tp_size,
    dp_size,
    base_batch_size,
    num_stages_behind,
    breakdown=False,
):
    mbs_list = [base_batch_size // dp_size for j in range(len(ops))]

    memory_weights, inputs, activations = get_memory_v3(
        ops, mbs_list, tp_size
    )
    memory_gradients = memory_weights
    memory_main_params = memory_weights * args.memory_main_params
    memory_optimizer = memory_weights * args.memory_optimizer

    saved_activations = get_activations_v3(
        ops, mbs_list, tp_size
    )
    peak_activations = get_peak_activations(
        ops, mbs_list, tp_size
    )

    if args.consider_reserved_space:
        memory_reserved = get_reserved_memory(
            ops, mbs_list, tp_size, dp_size, memory_weights
        )
    else:
        memory_reserved = 0

    memory_activations = (inputs + activations - saved_activations) * (
        num_stages_behind
    )
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
    time_result_strings = []

    num_gpus_till_now = 0
    for i in range(config.num_stages):
        stage = config.stages[i]
        ops = stage.ops
        num_gpus = stage.num_gpus
        tp_size = stage.tp_size
        dp_size = stage.dp_size
        num_stages_behind = stage.num_stages_behind
        mbs_list = [base_batch_size // dp_size for j in range(len(ops))]

        in_cross_node = (
            num_gpus_till_now % args.num_gpus_per_node
        ) == 0 and num_gpus_till_now > 0
        num_gpus_till_now += num_gpus
        out_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0

        ## compute actual time of each stage
        fwd_comp, bwd_comp, in_comm, out_comm, tp_comm = get_time_v3(
            ops, mbs_list, tp_size, dp_size, in_cross_node, out_cross_node
        )
        sum_time = (fwd_comp + bwd_comp + in_comm + out_comm) / 1000
        _time_list.append(sum_time)
        compute_time_list.append((fwd_comp + bwd_comp) / 1000)
        gpu_time_list.append(sum_time * num_gpus)

        if print_time:
            time_result_strings.append(
                "[stage {}], {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, ".format(
                    i,
                    fwd_comp / 1000 * num_batches,
                    (bwd_comp) / 1000 * num_batches,
                    in_comm / 1000 * num_batches,
                    out_comm / 1000 * num_batches,
                    tp_comm / 1000 * num_batches,
                )
            )

        ## compute ideal time of each stage
        _mbs_list = [base_batch_size for _ in range(len(ops))]
        _tp_size = 1
        _dp_size = 1
        _fwd_comp, _bwd_comp, _in_comm, _out_comm, _tp_comm = get_time_v3(
            ops, _mbs_list, _tp_size, _dp_size, in_cross_node, out_cross_node
        )
        ideal_time = (_fwd_comp + _bwd_comp + _in_comm + _out_comm) / 1000

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
            dp_size,
            base_batch_size,
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

        if print_memory:
            memory_result_strings.append(
                f"[stage {i}] memory = {memory_sum:.2f} MB. weights = {memory_weights:.0f}, gradients = {memory_gradients:.0f}, optimizer = {memory_optimizer:.0f}, activations = {memory_activations:.0f}, peak += {memory_peak:.0f}, memory_reserved = {memory_reserved:.0f}"
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
            "stage, fwd_comp, bwd_comp, in_comm(+reshard), out_comm(+reshard), reshard, sum(us)"
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
        print("\nmax allocated memory = {:.2f} MB".format(max_memory))
        for i in range(config.num_stages):
            print(memory_result_strings[i])
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
            dp_size = stage.dp_size
            num_stages_behind = stage.num_stages_behind

            _, _, _, _, _, reserved_mem = predict_stage_memory(
                ops,
                tp_size,
                dp_size,
                base_batch_size,
                num_stages_behind,
                breakdown=True,
            )
            reserved_mem_list.append(reserved_mem)
    return reserved_mem_list


def predict_value_after_move(
    config,
    bottleneck,
    partner,
    num_ops_moved,
    metric,
    inc_gpus=False,
    dec_gpus=False,
    dim=None,
):
    base_batch_size = config.micro_bs
    ops = list(config.stages[bottleneck].ops)
    tp_size = list(config.stages[bottleneck].tp_size)
    dp_size = list(config.stages[bottleneck].dp_size)
    num_stages_behind = config.stages[bottleneck].num_stages_behind

    if num_ops_moved > 0:
        if bottleneck < partner:
            ops = ops[:-num_ops_moved]
            tp_size = tp_size
            dp_size = dp_size
        else:
            ops = ops[num_ops_moved:]
            tp_size = tp_size
            dp_size = dp_size

    if inc_gpus:
        if dim == "tp":
            tp_size *= 2
        elif dim == "dp":
            dp_size *= 2
    if dec_gpus:
        if dim == "tp":
            tp_size //= 2
        elif dim == "dp":
            dp_size //= 2

    if metric in ["time", "time_with_efficiency"]:
        pred_value = predict_stage_time(
            ops, tp_size, dp_size, base_batch_size
        )
    elif metric == "memory":
        pred_value = predict_stage_memory(
            ops,
            tp_size,
            dp_size,
            base_batch_size,
            num_stages_behind,
        )
    else:
        raise RuntimeError(f"metric {metric} not implemented.")
    return pred_value


######## recomputation-related functions #########

stage_memory_set = {}
stage_memory_visit = 0
stage_memory_hit = 0


def predict_stage_memory_helper(
    config,
    stage_index,
    ops=None,
    tp_size=None,
    dp_size=None,
    base_batch_size=None,
    num_stages_behind=None,
):
    global stage_memory_visit, stage_memory_hit, stage_memory_set
    if ops is None:
        ops = config.stages[stage_index].ops
        tp_size = config.stages[stage_index].tp_size
        dp_size = config.stages[stage_index].dp_size
        base_batch_size = config.stages[stage_index].base_bs
        num_stages_behind = config.stages[stage_index].num_stages_behind

    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}dp{dp_size}bs{base_batch_size}stage{num_stages_behind}"
    stage_memory_visit += 1
    if stage_memory_set.get(config_str) is not None:
        stage_memory_hit += 1
        return stage_memory_set[config_str]

    pred_memory = predict_stage_memory(
        ops,
        tp_size,
        dp_size,
        base_batch_size,
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
    dp_size = config.stages[stage_index].dp_size
    base_batch_size = config.micro_bs

    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}dp{dp_size}bs{base_batch_size}"
    stage_time_visit += 1
    if stage_time_set.get(config_str) is not None:
        stage_time_hit += 1
        return stage_time_set[config_str]

    pred_time = predict_stage_time(
        ops, tp_size, dp_size, base_batch_size
    )
    stage_time_set[config_str] = pred_time

    return pred_time

def wrap_predict_delta_time(
    config, longest_stage, shortest_stage, num_ops_moved, decrease=True
):
    base_batch_size = config.micro_bs
    ops = config.stages[longest_stage].ops
    tp_size = config.stages[longest_stage].tp_size
    dp_size = config.stages[longest_stage].dp_size

    num_ops = len(ops)
    if longest_stage < shortest_stage and decrease:
        ops = ops[num_ops - num_ops_moved :]
        tp_size = tp_size
        dp_size = dp_size
    elif longest_stage > shortest_stage and decrease:
        ops = ops[0:num_ops_moved]
        tp_size = tp_size
        dp_size = dp_size
    elif longest_stage < shortest_stage and not decrease:
        ops = ops[num_ops - num_ops_moved :]
        tp_size = config.stages[longest_stage + 1].tp_size
        dp_size = config.stages[longest_stage + 1].dp_size
    elif longest_stage > shortest_stage and not decrease:
        ops = ops[0:num_ops_moved]
        tp_size = config.stages[longest_stage - 1].tp_size
        dp_size = config.stages[longest_stage - 1].dp_size
    else:
        raise RuntimeError("")

    pred_delta_time = predict_stage_time(
        ops,
        tp_size,
        dp_size,
        base_batch_size,
        delta=True,
        on_the_right=longest_stage < shortest_stage,
        decrease=decrease,
    )
    return pred_delta_time


################################

if __name__ == "__main__":

    config, config_dict = read_config_from_json(args, return_config_dict=True)
    read_profiled(
        config_dict["model_name"], config_dict["model_size"], args.profiled_gpt_path, args.profiled_dist_p2p_path, args.profiled_local_p2p_path, args.profiled_local_comm_path
    )
    predict_time_breakdown(config, print_time=True, print_memory=True)
    if args.save_to_csv is not None:
        save_config_info_to_csv(
            config, get_reserved_memory_list(config), args.save_to_csv
        )

    print(f"---- testing model ----")
    from hetaceso_cost_model import HetacesoPerfModel

    test_perf_model = HetacesoPerfModel(
        args.profiled_gpt_path,
        args.profiled_local_p2p_path,
        args.profiled_dist_p2p_path,
        config_dict["model_name"],
        config_dict["model_size"],
        max_tp_size,
        args.micro_batch_size,
        args.num_gpus_per_node,
        args.dist_optimizer,
        "1000Mbps"
    )
    test_perf_model.predict_config_performance(config, print_detail=True)
    print(
        f"time list = {list(map(int, config.time_list))}\nmemory list = {list(map(int, config.memory_list))}"
    )