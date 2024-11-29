import math
import csv
from hetaceso_utils import (
    get_op_list,
)
import os

LOG_LEVEL = int(os.environ.get("LOG_LEVEL", "0"))


def read_profiled_results(
    gpt_path, local_p2p_path, dist_p2p_path, model_name, model_size, max_tp_size, mbs_index_dict
):

    tp_size_list = [2**i for i in range(int(math.log2(max_tp_size)) + 1)]

    compute_fwd_time = {}
    compute_bwd_time = {}
    input_size = {}
    output_size = {}
    weights = {}
    activations = {}
    reserved_fwd = {}
    reserved_bwd = {}

    if model_name == "gpt":
        op_list = get_op_list()
    else:
        raise RuntimeError(f"not implemented for model {model_name}")

    for op_name in op_list:
        compute_fwd_time[op_name] = []
        compute_bwd_time[op_name] = []
        input_size[op_name] = []
        output_size[op_name] = []
        weights[op_name] = []
        activations[op_name] = []

        reserved_fwd[op_name] = []
        reserved_bwd[op_name] = []

        for i in range(len(mbs_index_dict)):
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

    for mbs in mbs_index_dict:
        for tp in tp_size_list:
            mbs_index = mbs_index_dict[mbs]
            tp_index = int(math.log(tp, 2))
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

                            reserved_fwd[op_name][mbs_index][tp_index] = float(row[7])
                            reserved_bwd[op_name][mbs_index][tp_index] = float(row[8])
            except:
                print(
                    f"file ({src_data_file}) not exist, or the file is not formatted as expected."
                )

    inter_band = 0
    intra_band = 0
    inter_band_file = dist_p2p_path + "p2p_inter_node.csv"
    intra_band_file = local_p2p_path + "p2p_intra_node.csv"
    try:
        with open(intra_band_file) as f:
            src_data = csv.reader(f)
            for idx, row in enumerate(src_data):
                if idx == 1:
                    intra_band = [float(row[i]) * 0.001 for i in range(len(row))]
    except:
        print(f"intra-node bandwidth file is not found.")

    try:
        with open(inter_band_file) as f:
            src_data = csv.reader(f)
            for idx, row in enumerate(src_data):
                if idx == 1:
                    inter_band = [float(row[i]) * 0.001 for i in range(len(row))]
    except:
        print(
            f"inter-node bandwidth file is not found, using intra-node bandwidth instead.\n"
        )
        inter_band = intra_band

    return (
        compute_fwd_time,
        compute_bwd_time,
        input_size,
        output_size,
        weights,
        activations,
        reserved_fwd,
        reserved_bwd,
        inter_band,
        intra_band,
    )


def customize_inter_band(src_band, band_type):
    if band_type == "1000Mbps":
        return [0.125 * 0.001 for _ in range(len(src_band))]
    elif band_type == "10Gbps":
        return [1.25 * 0.001 for _ in range(len(src_band))]
    elif band_type == "100Gbps":
        return [12.5 * 0.001 for _ in range(len(src_band))]
    elif band_type == "200Gbps":
        return [25 * 0.001 for _ in range(len(src_band))]
    else:
        raise RuntimeError("inter-node-band is not specified.")


class HetacesoPerfModel:
    def __init__(
        self,
        gpt_path,
        local_p2p_path,
        dist_p2p_path,
        model_name,
        model_size,
        max_tp_size,
        mbs_list,
        num_gpus_per_node,
        dist_optimizer,
        inter_node_band: str=None,
    ):
        self.mbs_index = {}
        for i in range(len(mbs_list)):
            self.mbs_index[mbs_list[i]] = i

        ## read profiled results
        (
            self.compute_fwd_time,
            self.compute_bwd_time,
            self.input_size,
            self.output_size,
            self.weights,
            self.activations,
            self.reserved_fwd,
            self.reserved_bwd,
            self.inter_node_band,
            self.intra_node_band,
        ) = read_profiled_results(
            gpt_path, local_p2p_path, dist_p2p_path, model_name, model_size, max_tp_size, self.mbs_index
        )
        if inter_node_band is not None:
            self.inter_node_band = customize_inter_band(
                self.inter_node_band, inter_node_band
            )

        ## memory ratio used to calculate main_param and optimizer size
        self.memory_ratio_main_param = 2
        self.memory_ratio_optimizer = 4
        self.memory_ratio_gradient = 1

        ## memory predict type:
        ## MAX refers to predict the max reserved memory as (reserved_fwd + reserved_bwd)
        ## MIN refers to max(reserved_fwd, reserved_bwd)
        self.memory_predict_type = "MAX"

        ## hardware infomation
        self.num_gpus_per_node = num_gpus_per_node

        ## whether use distributed optimizer
        self.dist_optimizer = dist_optimizer

    ## return in MB/us
    def bandwidth(self, data_size, cross_node):
        assert data_size >= 0, f'invalid data_size {data_size}'

        if cross_node:
            band = self.inter_node_band
        else:
            band = self.intra_node_band

        if data_size > 0:
            index = int(math.log(data_size, 2))
            if index >= 1:
                index -= 1
            if index >= len(band):
                return band[-1]
            else:
                return band[index]
        else:
            return 1

    ## TODO: check if mbs is needed
    def get_weight_size(self, ops, mbs, tp):

        weight_size = 0
        for i in range(len(ops)):
            mbs_index = self.mbs_index[mbs[i]]
            tp_index = int(math.log(tp, 2))
            weight_size += self.weights[ops[i]][mbs_index][tp_index]

        return weight_size

    def get_weight_size_no_embed(self, ops, mbs, tp):

        weight_size = 0
        ignored_ops = ["encoder-embedding", "gpt-post-process"]
        for i in range(len(ops)):
            if ops[i] not in ignored_ops:
                mbs_index = self.mbs_index[mbs[i]]
                tp_index = int(math.log(tp, 2))
                weight_size += self.weights[ops[i]][mbs_index][tp_index]

        return weight_size

    def get_activation_size(self, ops, mbs, tp, num_stages_behind):
        in_mbs_index = self.mbs_index[mbs[0]]
        in_tp_index = int(math.log(tp, 2))
        inputs = self.input_size[ops[0]][in_mbs_index][in_tp_index]

        sum_activation_size = 0
        saved_size = 0
        saved_size_list = [0]
        for i in range(len(ops)):
            mbs_index = self.mbs_index[mbs[i]]
            tp_index = int(math.log(tp, 2))
            # TODO: Check Calculation
            current_activation_size = self.activations[ops[i]][mbs_index][tp_index]
            saved_size += current_activation_size
            sum_activation_size += current_activation_size

            if saved_size > 0:
                saved_size_list.append(saved_size)
                saved_size = 0

        peak_activation_size = max(saved_size_list)
        activation_size = (inputs + sum_activation_size) * (
            num_stages_behind + 1
        ) + peak_activation_size

        return activation_size

    def get_reserved_size(self, ops, mbs, tp, weight_size):

        reserved_fwd_size = 0
        reserved_bwd_size = 0
        for i in range(len(ops) - 1):
            mbs_index = self.mbs_index[mbs[i]]
            tp_index = int(math.log(tp, 2))
            if self.reserved_fwd[ops[i]][mbs_index][tp_index] > reserved_fwd_size:
                reserved_fwd_size = self.reserved_fwd[ops[i]][mbs_index][tp_index]
            if self.reserved_bwd[ops[i]][mbs_index][tp_index] > reserved_bwd_size:
                reserved_bwd_size = self.reserved_bwd[ops[i]][mbs_index][tp_index]

        if self.memory_predict_type == "MAX":
            return max(reserved_fwd_size + reserved_bwd_size, weight_size)
        elif self.memory_predict_type == "MIN":
            return max(reserved_fwd_size, reserved_bwd_size, weight_size)
        else:
            raise RuntimeError(
                f"unknown memory_predict_type {self.memory_predict_type}"
            )

    def get_op_time(self, ops, mbs, tp):

        fwd_comp_time, bwd_comp_time = 0, 0
        for i in range(len(ops)):
            op_name = ops[i]
            mbs_index = self.mbs_index[mbs[i]]
            tp_index = int(math.log(tp, 2))
            fwd_comp_time += self.compute_fwd_time[op_name][mbs_index][tp_index]
            bwd_comp_time += self.compute_bwd_time[op_name][mbs_index][tp_index]

        return fwd_comp_time, bwd_comp_time

    def get_p2p_comm_time(self, ops, mbs, tp, in_cross_node, out_cross_node):
        in_mbs_index = self.mbs_index[mbs[0]]
        in_tp_index = int(math.log(tp, 2))
        input_comm_size = self.input_size[ops[0]][in_mbs_index][in_tp_index]
        in_comm_time = input_comm_size / self.bandwidth(input_comm_size, in_cross_node)

        out_mbs_index = self.mbs_index[mbs[-1]]
        out_tp_index = int(math.log(tp, 2))
        output_comm_size = self.output_size[ops[-1]][out_mbs_index][out_tp_index]
        if output_comm_size < 0:
            output_comm_size = 0
        out_comm_time = output_comm_size / self.bandwidth(
            output_comm_size, out_cross_node
        )

        # return in_comm_time, out_comm_time
        return in_comm_time, out_comm_time

    # def predict_stage_memory(self, stage_info, micro_batch_size, print_detail=False, breakdown=False):
    def predict_stage_memory(
        self, config, stage_index, stage_info=None, print_detail=False, breakdown=False
    ):
        if stage_info is None:
            stage_info = config.stages[stage_index]
        ops = stage_info.ops
        if len(ops) == 0:
            return 0
        micro_batch_size = config.micro_bs
        tp_size = stage_info.tp_size
        dp_size = stage_info.dp_size
        mbs_list = [micro_batch_size // dp_size for j in range(len(ops))]
        num_stages_behind = stage_info.num_stages_behind

        weight_size = self.get_weight_size(ops, mbs_list, tp_size)
        weight_size_no_embedding = self.get_weight_size_no_embed(ops, mbs_list, tp_size)
        main_param_size = weight_size * self.memory_ratio_main_param
        gradient_size = weight_size * self.memory_ratio_gradient
        optimizer_size = weight_size * self.memory_ratio_optimizer
        if self.dist_optimizer:
            optimizer_size /= dp_size
        reserved_memory_size = self.get_reserved_size(
            ops, mbs_list, tp_size, weight_size
        )
        activation_size = self.get_activation_size(
            ops, mbs_list, tp_size, num_stages_behind
        )

        memory_sum = (
            weight_size
            + main_param_size
            + gradient_size
            + optimizer_size
            + reserved_memory_size
            + activation_size
        )
        if print_detail:
            print(
                f"MEMORY=[{memory_sum:.0f}]. weights = {weight_size:.0f}, main_params = {main_param_size:.0f}, gradients = {gradient_size:.0f}, optimizer = {optimizer_size:.0f}, activation = {activation_size:.0f}, reserved = {reserved_memory_size:.0f}"
            )

        if breakdown:
            return memory_sum, weight_size, weight_size_no_embedding
        else:
            return memory_sum

    def predict_stage_time(
        self,
        stage_info,
        micro_batch_size,
        num_micro_batches,
        in_cross_node,
        out_cross_node,
        print_detail=False,
    ):  # , ops, tp_size, dp_size, base_batch_size, delta=False, on_the_right=False, decrease=True, in_cross_node=False, out_cross_node=False):
        ops = stage_info.ops
        if len(ops) == 0:
            return 0
        dp_size = stage_info.dp_size
        tp_size = stage_info.tp_size
        mbs_list = [micro_batch_size // dp_size for j in range(len(ops))]

        ## all the time is in [us].
        fwd_comp_time, bwd_comp_time = self.get_op_time(ops, mbs_list, tp_size)
        in_comm_time, out_comm_time = self.get_p2p_comm_time(
            ops, mbs_list, tp_size, in_cross_node, out_cross_node
        )
        sum_time = fwd_comp_time + bwd_comp_time + in_comm_time + out_comm_time

        if print_detail:
            print(
                f"Time(ms)=[{sum_time/1000 * num_micro_batches:.2f}]. fwd_compute = {fwd_comp_time * num_micro_batches / 1000 :.2f}, bwd_compute = {bwd_comp_time * num_micro_batches / 1000 :.2f}, in_comm_time = {in_comm_time * num_micro_batches / 1000 :.2f}, out_comm_time = {out_comm_time * num_micro_batches / 1000 :.2f}"
            )

        ## return [ms]
        return (
            sum_time / 1000 * num_micro_batches,
            fwd_comp_time / 1000 * num_micro_batches,
            bwd_comp_time / 1000 * num_micro_batches,
        )

    def predict_config_performance(self, config, print_detail=False, print_log=False):
        micro_batch_size = config.micro_bs
        num_micro_batches = config.global_bs // micro_batch_size

        time_list = []
        fwd_time_list = []
        # bwd_time_list = []
        memory_list = []
        weight_size_list = []
        weight_size_no_embed_list = []

        num_gpus_till_now = 0
        for i in range(config.num_stages):
            stage_info = config.stages[i]

            in_cross_node = (
                num_gpus_till_now % self.num_gpus_per_node
            ) == 0 and num_gpus_till_now > 0
            num_gpus_till_now += stage_info.num_gpus
            out_cross_node = (num_gpus_till_now % self.num_gpus_per_node) == 0

            total_time, fwd_time, _ = self.predict_stage_time(
                stage_info,
                micro_batch_size,
                num_micro_batches,
                in_cross_node,
                out_cross_node,
                print_detail,
            )
            time_list.append(total_time)
            fwd_time_list.append(fwd_time)
            memory_sum, memory_weight, memory_weight_no_embed = (
                self.predict_stage_memory(
                    config, i, print_detail=print_detail, breakdown=True
                )
            )

            memory_list.append(memory_sum)
            weight_size_list.append(memory_weight)
            weight_size_no_embed_list.append(memory_weight_no_embed)

        warmup_cooldown_time = sum(time_list) / num_micro_batches
        for i in range(len(time_list)):
            time_list[i] = (time_list[i] / num_micro_batches) * (
                num_micro_batches - 1
            ) + warmup_cooldown_time
        config.time_list = time_list
        config.memory_list = memory_list
        config.weight_size_list = weight_size_list
        config.weight_size_no_embed_list = weight_size_no_embed_list

        config.fwd_time_list = fwd_time_list
