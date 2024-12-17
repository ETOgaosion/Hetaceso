# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import math
import os

import sys
sys.path.append("../runtime")
from megatron.training.theoretical_memory_usage import report_theoretical_memory

from model_ops_info import get_op_list, get_full_op_list
from hetaceso_utils import *

configs = {
    "tp": [1, 2, 1, 1, 2, 1, 1, 1],
    "usp": [1, 1, 2, 1, 2, 2, 1, 4],
    "rsp": [1, 1, 1, 2, 1, 2, 4, 1],
    "dp": [4, 2, 2, 2, 1, 1, 1, 1],
}

class HetacesoPerformanceModel:
    def __init__(self, args, machine_topo, config, config_dict):
        self.args = args
        self.machine_topo = machine_topo
        self.config = config
        self.config_dict = config_dict
        self.op_list = get_op_list(self.args)
        self.full_op_list = get_full_op_list(self.args)
        self.ops_in_each_stage = []
        start_op_idx = 0
        for i in range(self.config.num_stages):
            self.ops_in_each_stage.append(self.full_op_list[start_op_idx: start_op_idx + self.config.num_op_list[i]])
            start_op_idx += self.config.num_op_list[i]
        
        self.model_name = self.config.model_name
        self.model_size = self.config.model_size
        
        self.compute_fwd_time = {}
        self.compute_bwd_time = {}
        self.input_size = {}
        self.output_size = {}
        self.weights = {}
        self.activations = {}
        self.collective_time = {}
        
        self.reserved_fwd = {}
        self.reserved_bwd = {}
        
        self.inter_band = {}
        self.intra_band = []
        
        self.num_ops_stage, self.num_gpu_list, self.total_mbs, self.tp_size_list, self.cp_size_list, self.usp_size_list, self.rsp_size_list, self.dp_size_list, self.rsp_split_list, self.dp_split_list = config_details(self.config)
        
        total_gpus = 0
        for num_gpu in self.num_gpu_list:
            total_gpus += num_gpu
        self.total_gpus = total_gpus
        self.machine_idx_rank_map = {}
        for rank in range(self.total_gpus):
            self.machine_idx_rank_map[rank] = self.get_machine_idx_from_rank(rank)

        ## memory ratio used to calculate main_param and optimizer size
        self.memory_ratio_main_param = 2
        self.memory_ratio_optimizer = 4
        self.memory_ratio_gradient = 1

        ## memory predict type:
        ## MAX refers to predict the max reserved memory as (reserved_fwd + reserved_bwd)
        ## MIN refers to max(reserved_fwd, reserved_bwd)
        self.memory_predict_type = "MAX"

        ## whether use distributed optimizer
        self.dist_optimizer = args.dist_optimizer
        
    def calculate_node_rank(self, node_rank):
        sum_nodes = 0
        for i in range(len(self.num_gpu_list)):
            if sum_nodes + self.num_gpu_list[i] > node_rank:
                pp_rank = i
                break
            sum_nodes += self.num_gpu_list[i]
        node_inner_rank = node_rank - sum_nodes
        tp_rank = node_inner_rank % self.tp_size_list[pp_rank]
        cp_rank = node_inner_rank // self.tp_size_list[pp_rank] % self.cp_size_list[pp_rank]
        usp_rank = node_inner_rank // self.tp_size_list[pp_rank] % self.usp_size_list[pp_rank]
        rsp_rank = node_inner_rank // self.tp_size_list[pp_rank] // self.usp_size_list[pp_rank] % self.rsp_size_list[pp_rank]
        dp_rank = node_inner_rank // self.tp_size_list[pp_rank] // self.usp_size_list[pp_rank] // self.rsp_size_list[pp_rank] % self.dp_size_list[pp_rank]
        cur_mbs = self.dp_split_list[pp_rank][dp_rank]
        cur_seqlen = self.rsp_split_list[pp_rank][rsp_rank] // self.usp_size_list[pp_rank]
        
        cur_tp, cur_cp, cur_usp, cur_rsp, cur_dp = self.tp_size_list[pp_rank], self.cp_size_list[pp_rank], self.usp_size_list[pp_rank], self.rsp_size_list[pp_rank], self.dp_size_list[pp_rank]
        return tp_rank, cp_rank, usp_rank, rsp_rank, dp_rank, pp_rank, cur_mbs, cur_seqlen, cur_tp, cur_cp, cur_usp, cur_rsp, cur_dp
        

    def read_profiled(self, homogeneous=True):
        unique_config_list = []
        unique_config_map = {}
        comm_num_gpus_list_map = {"tp": [], "usp": [], "rsp": [], "dp": []}
        comm_num_gpus_map_map = {"tp": {}, "usp": {}, "rsp": {}, "dp": {}}
        
        total_gpus = self.total_gpus
        if homogeneous:
            total_gpus = 1
        for rank in range(total_gpus):
            _, _, _, _, _, _, cur_mbs, cur_seqlen, cur_tp, _, cur_usp, cur_rsp, cur_dp = self.calculate_node_rank(rank)
            if (cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp) not in unique_config_map:
                unique_config_map[(cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp)] = rank
                unique_config_list.append((rank, cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp))

        for op_name in self.op_list:
            self.compute_fwd_time[op_name] = {}
            self.compute_bwd_time[op_name] = {}
            self.input_size[op_name] = {}
            self.output_size[op_name] = {}
            self.weights[op_name] = {}
            self.activations[op_name] = {}

            self.reserved_fwd[op_name] = {}
            self.reserved_bwd[op_name] = {}
            
            for rank, mbs, seqlen, tp, usp, rsp, dp in unique_config_list:
                if mbs not in self.compute_fwd_time[op_name]:
                    self.compute_fwd_time[op_name][mbs] = {}
                    self.compute_bwd_time[op_name][mbs] = {}
                    self.input_size[op_name][mbs] = {}
                    self.output_size[op_name][mbs] = {}
                    self.weights[op_name][mbs] = {}
                    self.activations[op_name][mbs] = {}

                    self.reserved_fwd[op_name][mbs] = {}
                    self.reserved_bwd[op_name][mbs] = {}
                if seqlen not in self.compute_fwd_time[op_name][mbs]:
                    self.compute_fwd_time[op_name][mbs][seqlen] = {}
                    self.compute_bwd_time[op_name][mbs][seqlen] = {}
                    self.input_size[op_name][mbs][seqlen] = {}
                    self.output_size[op_name][mbs][seqlen] = {}
                    self.weights[op_name][mbs][seqlen] = {}
                    self.activations[op_name][mbs][seqlen] = {}

                    self.reserved_fwd[op_name][mbs][seqlen] = {}
                    self.reserved_bwd[op_name][mbs][seqlen] = {}
                if tp not in self.compute_fwd_time[op_name][mbs][seqlen]:
                    self.compute_fwd_time[op_name][mbs][seqlen][tp] = 1000000
                    self.compute_bwd_time[op_name][mbs][seqlen][tp] = 1000000
                    self.input_size[op_name][mbs][seqlen][tp] = 1000000
                    self.output_size[op_name][mbs][seqlen][tp] = 1000000
                    self.weights[op_name][mbs][seqlen][tp] = 1000000
                    self.activations[op_name][mbs][seqlen][tp] = 1000000

                    self.reserved_fwd[op_name][mbs][seqlen][tp] = 1000000
                    self.reserved_bwd[op_name][mbs][seqlen][tp] = 1000000
                if tp > 1:
                    if tp not in comm_num_gpus_map_map["tp"]:
                        comm_num_gpus_list_map["tp"].append(tp)
                        comm_num_gpus_map_map["tp"][tp] = [rank]
                    elif not homogeneous:
                        comm_num_gpus_map_map["tp"][tp].append(rank)
                if usp > 1:
                    if usp not in comm_num_gpus_map_map["usp"]:
                        comm_num_gpus_list_map["usp"].append(usp)
                        comm_num_gpus_map_map["usp"][usp] = [rank]
                    elif not homogeneous:
                        comm_num_gpus_map_map["usp"][usp].append(rank)
                if rsp > 1:
                    if rsp not in comm_num_gpus_map_map["rsp"]:
                        comm_num_gpus_list_map["rsp"].append(rsp)
                        comm_num_gpus_map_map["rsp"][rsp] = [rank]
                    elif not homogeneous:
                        comm_num_gpus_map_map["rsp"][rsp].append(rank)
                if dp > 1:
                    if dp not in comm_num_gpus_map_map["dp"]:
                        comm_num_gpus_list_map["dp"].append(dp)
                        comm_num_gpus_map_map["dp"][dp] = [rank]
                    elif not homogeneous:
                        comm_num_gpus_map_map["dp"][dp].append(rank)
        
        for rank, mbs, seqlen, tp, usp, rsp, dp in unique_config_list:
            src_data_file = f'{self.args.profiled_gpt_path}rank{rank}/{self.model_name}_{self.model_size}_mbs{mbs}_seqlen{seqlen}_tp{tp}.csv'
            print(src_data_file)
            try:
                with open(src_data_file) as f:
                    src_data = csv.reader(f)
                    line_index = 0
                    for row in src_data:
                        line_index += 1
                        if line_index > 1:
                            op_name = row[0]
                            self.compute_fwd_time[op_name][mbs][seqlen][tp] = float(
                                row[1]
                            )
                            self.compute_bwd_time[op_name][mbs][seqlen][tp] = float(
                                row[2]
                            )
                            self.input_size[op_name][mbs][seqlen][tp] = float(row[3])
                            self.output_size[op_name][mbs][seqlen][tp] = float(row[4])
                            self.weights[op_name][mbs][seqlen][tp] = float(row[5])
                            self.activations[op_name][mbs][seqlen][tp] = float(row[6])
                            self.reserved_fwd[op_name][mbs][seqlen][tp] = float(
                                row[7]
                            )
                            self.reserved_bwd[op_name][mbs][seqlen][tp] = float(
                                row[8]
                            )
            except:
                print(
                    f"file ({src_data_file}) not exist, or the file is not formatted as expected."
                )
        
        for op_name in self.compute_fwd_time:
            for mbs in self.compute_fwd_time[op_name]:
                for seqlen in self.compute_fwd_time[op_name][mbs]:
                    for tp in self.compute_fwd_time[op_name][mbs][seqlen]:
                        assert self.reserved_bwd[op_name][mbs][seqlen][tp] < 1000000, f'{self.model_size} {op_name} {mbs} {seqlen} {tp} is not valid'
        
        '''
        Communications in Megatron:
        
        - tensor parallel: 4 all-reduce (self-attention 2, mlp 2)
        - ulysses context parallel: an inevitable all-to-all communication
        - ring context parallel: all ring rank need p2p communication, like all-reduce but can possibly overlap with computation
        - data parallel: all ranks need all-reduce gradients
        '''
        self.collective_time = {"all_reduce": {}, "all_to_all": {}}
        # self.collective_time = {"all_reduce": {}, "all_gather": {}, "reduce_scatter": {}, "all_to_all": {}}
        comm_prim_map = {"tp": ["all_reduce", "all_gather", "reduce_scatter"], "usp": ["all_to_all"], "rsp": ["all_reduce"], "dp": ["all_reduce"]}
        for cfg_i in range(len(configs["tp"])):
            tp = configs["tp"][cfg_i]
            usp = configs["usp"][cfg_i]
            rsp = configs["rsp"][cfg_i]
            dp = configs["dp"][cfg_i]
            for prim in self.collective_time.keys():
                if (prim == "all_to_all" and usp > 1) or (prim == "all_reduce" and dp > 1):
                    if (tp, usp, rsp, dp) not in self.collective_time[prim]:
                        self.collective_time[prim][(tp, usp, rsp, dp)] = {}
                    if rank not in self.collective_time[prim][(tp, usp, rsp, dp)]:
                        self.collective_time[prim][(tp, usp, rsp, dp)][rank] = {}
                    src_data_file = f'{self.args.profiled_local_comm_path}rank{rank}/prim_{self.model_name}_{self.model_size}_tp{tp}_usp{usp}_rsp{rsp}_dp{dp}_{prim}.csv'
                    print(f'read {src_data_file}')
                    with open(src_data_file) as f:
                        src_data = csv.reader(f)
                        line_index = 0
                        for row in src_data:
                            line_index += 1
                            if line_index > 1:
                                data_size = row[0]
                                self.collective_time[prim][(tp, usp, rsp, dp)][rank][data_size]= float(row[1])

        for rank in range(total_gpus):
            self.intra_band_file = f'{self.args.profiled_local_p2p_path}rank{rank}/p2p_intra_node.csv'
            try:
                with open(self.intra_band_file) as f:
                    src_data = csv.reader(f)
                    for idx, row in enumerate(src_data):
                        if idx == 1:
                            self.intra_band.append([float(row[i]) for i in range(len(row))])
                        elif idx > 1:
                            break
            except:
                print(f"intra-node bandwidth file is not found.")
                
        for machine in range(self.machine_topo.num_machines - 1):
            for machine_other in range(machine + 1, self.machine_topo.num_machines):
                self.inter_band_file = f'{self.args.profiled_local_p2p_path}{machine}-{machine_other}/p2p_intra_node.csv'
                try:
                    with open(self.inter_band_file) as f:
                        src_data = csv.reader(f)
                        for idx, row in enumerate(src_data):
                            if idx == 1:
                                self.inter_band[(machine, machine_other)] = [float(row[i]) for i in range(len(row))]
                            elif idx > 1:
                                break
                except:
                    print(
                        f"inter-node bandwidth file is not found, using intra-node bandwidth instead."
                    )
                    self.inter_band = None

        return len(self.op_list)

    def intra_node_band(self, rank, data_size):
        if data_size > 0:
            index = int(math.log(data_size, 2))
            if index >= 1:
                index -= 1
            if index >= len(self.intra_band[rank]):
                return self.intra_band[rank][-1]
            else:
                return self.intra_band[rank][index]
        else:
            return 1

    def inter_node_band(self, data_size, cur_machine, other_machine):
        if data_size > 0:
            index = int(math.log(data_size, 2))
            if index >= 1:
                index -= 1
            if index >= len(self.inter_band):
                return self.inter_band[(cur_machine, other_machine)][-1]
            else:
                return self.inter_band[(cur_machine, other_machine)][index]
        else:
            return 1

    def get_comp_comm_time(self, rank, machine_idx, ops, cur_mbs, cur_seqlen, tp, usp, rsp, dp, in_cross_node, out_cross_node):
        if len(ops) == 0:
            return 0, 0, 0, 0, 0
        fwd_comp, bwd_comp, in_comm, out_comm, tp_comm, usp_comm, rsp_comm, dp_comm = 0, 0, 0, 0, 0, 0, 0, 0
        op_comp_time = {}
        for op in get_op_list(self.args):
            op_comp_time[op] = {"fwd": 0, "bwd": 0}
        
        '''
        TP communication refer to https://www.cnblogs.com/rossiXYZ/p/15871062.html
        DP communication refer to https://www.cnblogs.com/rossiXYZ/p/15868988.html
        CP according https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/context_parallel.html
        '''

        for i in range(len(ops)):
            op_name = ops[i]
            fwd_comp += self.compute_fwd_time[op_name][cur_mbs][cur_seqlen][tp]
            bwd_comp += self.compute_bwd_time[op_name][cur_mbs][cur_seqlen][tp]
            op_comp_time[op_name]["fwd"] += fwd_comp
            op_comp_time[op_name]["bwd"] += bwd_comp
            cur_op_input_size = str(int(self.input_size[op_name][cur_mbs][cur_seqlen][tp]))
            cur_op_output_size = str(int(self.output_size[op_name][cur_mbs][cur_seqlen][tp]))
            if op_name == "dec-embedding":
                '''
                Embedding layer need all-reduce output 
                runtime/megatron/core/tensor_parallel/layers.py: 228, VocabParallelEmbedding::forward
                '''
                pass
            elif op_name == "dec-post-process":
                '''
                TP: In theory like above
                '''
                '''
                DP: Need to allreduce gradients
                - Grad Buffer Async and Overlappable: runtime/megatron/core/distributed/param_and_grad_buffer.py: 140, Bucket::start_gradient_sync
                - Model Grad not overlappable: runtime/megatron/core/distributed/finalize_model_grads.py
                '''
                if dp > 1:
                    assert cur_op_input_size in self.collective_time["all_reduce"][(tp, usp, rsp, dp)][rank], f'{op_name} {cur_op_input_size}'
                    dp_comm += self.collective_time["all_reduce"][(tp, usp, rsp, dp)][rank][cur_op_input_size] * 2
            elif op_name == "dec-self-attention":
                '''
                Self attention
                - QKV need 3 ColumnParallelLinear layers, thus forward: 1 all-gather, backward: 1 all-reduce
                runtime/megatron/core/tensor_parallel/layers.py: 826, ColumnParallelLinear::forward
                - Dropout need 1 RowParallelLinear layer, thus forward: 1 all-reduce, backward: 1 all-gather
                '''
                '''
                CP:
                - USP: In TE's implementation, USP QKV communication can overlap with each other, thus only need to consider 1 all-to-all
                - RSP: In most case rsp can overlap with calculation
                '''
                if usp > 1:
                    assert cur_op_output_size in self.collective_time["all_to_all"][(tp, usp, rsp, dp)][rank], f'{op_name} {cur_op_output_size} {(tp, usp, rsp, dp)} {self.collective_time["all_to_all"][(tp, usp, rsp, dp)][rank]}'
                    usp_comm += self.collective_time["all_to_all"][(tp, usp, rsp, dp)][rank][cur_op_output_size] * 4
                if rsp > 1:
                    # ring KV, communication calculate where cannot overlap with computation
                    rsp_comm += (float(self.output_size[op_name][cur_mbs][cur_seqlen][tp]) / self.intra_node_band(rank, self.output_size[op_name][cur_mbs][cur_seqlen][tp]) - self.compute_fwd_time[op_name][cur_mbs][cur_seqlen][tp]) * 4
                    if rsp_comm < 0:
                        rsp_comm = 0
            elif op_name == "dec-mlp":
                '''
                MLP
                MLP need 1 ColumnParallelLinear, 1 RowParallelLinear
                '''
                pass
            else:
                raise RuntimeError(f"unknown op_name {op_name}")

        input_comm_size = self.input_size[ops[0]][cur_mbs][cur_seqlen][tp]
        output_comm_size = self.output_size[ops[-1]][cur_mbs][cur_seqlen][tp]

        if in_cross_node:
            in_comm = input_comm_size / self.inter_node_band(input_comm_size, machine_idx, (self.machine_topo.num_machines + machine_idx - 1) % self.machine_topo.num_machines)
        else:
            in_comm = input_comm_size / self.intra_node_band(rank, input_comm_size)

        if out_cross_node:
            out_comm = output_comm_size / self.inter_node_band(output_comm_size, rank, (machine_idx + 1) % self.machine_topo.num_machines)
        else:
            out_comm = output_comm_size / self.intra_node_band(rank, output_comm_size)

        return fwd_comp, bwd_comp, in_comm, out_comm, tp_comm, usp_comm, rsp_comm, dp_comm, op_comp_time


    ## TODO: check if mbs is needed
    def get_weight_size(self, ops, cur_mbs, cur_seqlen, tp, usp, rsp, dp):
        weight_size = 0
        for i in range(len(ops)):
            weight_size += self.weights[ops[i]][cur_mbs][cur_seqlen][tp]

        return weight_size

    def get_weight_size_no_embed(self, ops, cur_mbs, cur_seqlen, tp, usp, rsp, dp):
        weight_size = 0
        ignored_ops = ["dec-embedding", "dec-post-process"]
        for i in range(len(ops)):
            if ops[i] not in ignored_ops:
                weight_size += self.weights[ops[i]][cur_mbs][cur_seqlen][tp]

        return weight_size

    def get_activation_size(self, ops, cur_mbs, cur_seqlen, tp, usp, rsp, dp, num_stages_behind):
        inputs = self.input_size[ops[0]][cur_mbs][cur_seqlen][tp]

        sum_activation_size = 0
        saved_size = 0
        saved_size_list = [0]
        for i in range(len(ops)):
            # TODO: Check Calculation
            current_activation_size = self.activations[ops[i]][cur_mbs][cur_seqlen][tp]
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

    def get_reserved_size(self, ops, cur_mbs, cur_seqlen, tp, usp, rsp, dp, weight_size):

        self.reserved_fwd_size = 0
        self.reserved_bwd_size = 0
        for i in range(len(ops) - 1):
            if self.reserved_fwd[ops[i]][cur_mbs][cur_seqlen][tp] > self.reserved_fwd_size:
                self.reserved_fwd_size = self.reserved_fwd[ops[i]][cur_mbs][cur_seqlen][tp]
            if self.reserved_bwd[ops[i]][cur_mbs][cur_seqlen][tp] > self.reserved_bwd_size:
                self.reserved_bwd_size = self.reserved_bwd[ops[i]][cur_mbs][cur_seqlen][tp]

        if self.memory_predict_type == "MAX":
            return max(self.reserved_fwd_size + self.reserved_bwd_size, weight_size)
        elif self.memory_predict_type == "MIN":
            return max(self.reserved_fwd_size, self.reserved_bwd_size, weight_size)
        else:
            raise RuntimeError(
                f"unknown memory_predict_type {self.memory_predict_type}"
            )
    
    def get_machine_idx_from_rank(self, rank):
        assert rank < self.total_gpus, f'rank must less than sum of gpus: {rank} {self.total_gpus}'
        for machine_idx in range(self.machine_topo.num_machines):
            if self.machine_topo.machine_gpus[machine_idx] > rank:
                return machine_idx
            rank -= self.machine_topo.machine_gpus[machine_idx]
        return -1

    def predict_stage_time(
        self,
        rank,
        num_micro_batches,
        in_cross_node,
        out_cross_node,
        print_detail=False,
    ):        
        _, _, _, _, _, pp_rank, cur_mbs, cur_seqlen, cur_tp, _, cur_usp, cur_rsp, cur_dp = self.calculate_node_rank(rank)
        
        ops = self.ops_in_each_stage[pp_rank]

        ## fwd bwd time is in [us], comm time is in [ms].
        fwd_comp, bwd_comp, in_comm, out_comm, tp_comm, usp_comm, rsp_comm, dp_comm, op_comp_time = self.get_comp_comm_time(rank, self.machine_idx_rank_map[rank], ops, cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp, in_cross_node, out_cross_node)
        
        for op in op_comp_time:
            for fwd_bwd in op_comp_time[op]:
                op_comp_time[op][fwd_bwd] = op_comp_time[op][fwd_bwd] / 1000 * num_micro_batches

        if print_detail:
            print(
                f"Time(ms)=[fwd_compute = {fwd_comp * num_micro_batches / 1000 :.2f}, bwd_compute = {bwd_comp * num_micro_batches / 1000 :.2f}, in_comm_time = {in_comm * num_micro_batches :.2f}, out_comm_time = {out_comm * num_micro_batches :.2f}, tp_comm_time = {tp_comm * num_micro_batches :.2f}, usp_comm_time = {usp_comm * num_micro_batches :.2f}, rsp_comm_time = {rsp_comm * num_micro_batches :.2f}, dp_comm_time = {dp_comm * num_micro_batches :.2f}]"
            )

        ## return [ms]
        return (
            fwd_comp / 1000 * num_micro_batches,
            bwd_comp / 1000 * num_micro_batches,
            tp_comm * num_micro_batches,
            usp_comm * num_micro_batches,
            rsp_comm * num_micro_batches,
            dp_comm * num_micro_batches,
            op_comp_time,
        )

    def predict_stage_memory(
        self, rank, print_detail=False, breakdown=False, with_reference = False
    ):        
        _, _, _, _, _, pp_rank, cur_mbs, cur_seqlen, cur_tp, _, cur_usp, cur_rsp, cur_dp = self.calculate_node_rank(rank)
        
        ops = self.ops_in_each_stage[pp_rank]
        
        num_stages_behind = self.config.stages[pp_rank].num_stages_behind

        weight_size = self.get_weight_size(ops, cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp)
        weight_size_no_embedding = self.get_weight_size_no_embed(ops, cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp)
        main_param_size = weight_size * self.memory_ratio_main_param
        gradient_size = weight_size * self.memory_ratio_gradient
        optimizer_size = weight_size * self.memory_ratio_optimizer
        if self.dist_optimizer:
            optimizer_size /= cur_dp
        reserved_memory_size = self.get_reserved_size(
            ops, cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp, weight_size
        )
        activation_size = self.get_activation_size(
            ops, cur_mbs, cur_seqlen, cur_tp, cur_usp, cur_rsp, cur_dp, num_stages_behind
        )

        memory_sum = (
            weight_size
            + main_param_size
            + gradient_size
            + optimizer_size
            + reserved_memory_size
            + activation_size
        )
        
        if with_reference:
            self.args.data_parallel_size = cur_dp
            self.args.micro_batch_size = cur_mbs
            weight_and_optimizer_memory, activation_memory, total_memory = report_theoretical_memory(self.args, cur_mbs)
        
        if print_detail:
            print(
                f"MEMORY=[{memory_sum:.0f}]. self.weights = {weight_size:.0f}, main_params = {main_param_size:.0f}, gradients = {gradient_size:.0f}, optimizer = {optimizer_size:.0f}, activation = {activation_size:.0f}, reserved = {reserved_memory_size:.0f}"
            )

        ret = [memory_sum]
        if breakdown:
            ret.extend([weight_size, weight_size_no_embedding])
        if with_reference:
            ret.extend([total_memory])
            if breakdown:
                ret.extend([weight_and_optimizer_memory, activation_memory])

        return ret

    def predict_single_performance(self, rank, in_cross_node=True, out_cross_node=True, print_detail=False, print_log=False):
        micro_batch_size = self.config.micro_bs
        num_micro_batches = self.config.global_bs // micro_batch_size

        fwd_time, bwd_time, tp_comm_time, usp_comm_time, rsp_comm_time, dp_comm_time, op_comp_time = self.predict_stage_time(
            rank,
            num_micro_batches,
            in_cross_node,
            out_cross_node,
            print_detail,
        )
        total_time = fwd_time + bwd_time + usp_comm_time + rsp_comm_time + dp_comm_time
        memory_ret = (
            self.predict_stage_memory(
                rank, print_detail=print_detail, breakdown=True, with_reference=True
            )
        )
        memory_sum, weight_size, weight_size_no_embedding, total_memory, weight_and_optimizer_memory, activation_memory = memory_ret[0], memory_ret[1], memory_ret[2], memory_ret[3], memory_ret[4], memory_ret[5]
        
        self.config.time_list.append(total_time)
        self.config.fwd_time_list.append(fwd_time)
        self.config.bwd_time_list.append(bwd_time)
        self.config.tp_comm_time_list.append(tp_comm_time)
        self.config.usp_comm_time_list.append(usp_comm_time)
        self.config.rsp_comm_time_list.append(rsp_comm_time)
        self.config.dp_comm_time_list.append(dp_comm_time)
        
        self.config.embed_fwd_time_list.append(op_comp_time["dec-embedding"]["fwd"])
        self.config.embed_bwd_time_list.append(op_comp_time["dec-embedding"]["bwd"])
        self.config.att_fwd_time_list.append(op_comp_time["dec-self-attention"]["fwd"])
        self.config.att_bwd_time_list.append(op_comp_time["dec-self-attention"]["bwd"])
        self.config.mlp_fwd_time_list.append(op_comp_time["dec-mlp"]["fwd"])
        self.config.mlp_bwd_time_list.append(op_comp_time["dec-mlp"]["bwd"])
        self.config.post_fwd_time_list.append(op_comp_time["dec-post-process"]["fwd"])
        self.config.post_bwd_time_list.append(op_comp_time["dec-post-process"]["bwd"])
        
        self.config.memory_list.append(memory_sum)
        self.config.weight_size_list.append(weight_size)
        self.config.weight_size_no_embed_list.append(weight_size_no_embedding)
        
        self.config.ref_memory_list.append(total_memory)
        self.config.ref_weight_and_optimizer_memory.append(weight_and_optimizer_memory)
        self.config.ref_activation_memory.append(activation_memory)
        
        if print_log:
            print(f'rank: {rank}\n \
                    total_time: {total_time}\n \
                    fwd_time: {fwd_time}\n \
                    bwd_time: {bwd_time}\n \
                    tp_comm_time: {tp_comm_time}\n \
                    usp_comm_time: {usp_comm_time}\n \
                    rsp_comm_time: {rsp_comm_time}\n \
                    dp_comm_time: {dp_comm_time}\n \
                    embed_comp_fwd_time: {op_comp_time["dec-embedding"]["fwd"]}\n \
                    embed_comp_bwd_time: {op_comp_time["dec-embedding"]["bwd"]}\n \
                    att_comp_fwd_time: {op_comp_time["dec-self-attention"]["fwd"]}\n \
                    att_comp_bwd_time: {op_comp_time["dec-self-attention"]["bwd"]}\n \
                    mlp_comp_fwd_time: {op_comp_time["dec-mlp"]["fwd"]}\n \
                    mlp_comp_bwd_time: {op_comp_time["dec-mlp"]["bwd"]}\n \
                    post_comp_fwd_time: {op_comp_time["dec-post-process"]["fwd"]}\n \
                    post_comp_bwd_time: {op_comp_time["dec-post-process"]["bwd"]}\n \
                    memory_sum: {memory_sum}\n \
                    memory_weight: {weight_size}\n \
                    memory_weight_no_embed: {weight_size_no_embedding}\n \
                    ref_total_memory: {total_memory}\n \
                    ref_weight_and_optimizer_memory: {weight_and_optimizer_memory}\n \
                    ref_activation_memory: {activation_memory}')
    
    def predict_all_machine_performance(self, print_detail=False, print_log=False):
        cur_rank = 0
        cur_machine = 0
        cur_machine_gpu_idx = 0
        for stage in self.config.num_stages:
            for gpus in self.config.num_gpus[stage]:
                for rank in range(gpus):
                    in_cross_node = (cur_machine_gpu_idx == 0)
                    out_cross_node = (cur_machine_gpu_idx == self.machine_topo.machine_gpus[cur_machine] - 1)
                    self.predict_single_performance(cur_rank, in_cross_node, out_cross_node, print_detail, print_log)
                    cur_machine_gpu_idx = cur_machine_gpu_idx + 1
                    if cur_machine_gpu_idx == self.machine_topo.machine_gpus[cur_machine]:
                        cur_machine_gpu_idx = 0
                        cur_machine += 1
                    cur_rank += 1
        if print_log:
            print(f'self.config.time_list = {self.config.time_list}\n \
                    self.config.fwd_time_list = {self.config.fwd_time_list}\n \
                    self.config.bwd_time_list = {self.config.bwd_time_list}\n \
                    self.config.tp_comm_time_list = {self.config.tp_comm_time_list}\n \
                    self.config.usp_comm_time_list = {self.config.usp_comm_time_list}\n \
                    self.config.rsp_comm_time_list = {self.config.rsp_comm_time_list}\n \
                    self.config.dp_comm_time_list = {self.config.dp_comm_time_list}\n \
                    self.config.memory_list = {self.config.memory_list}\n \
                    self.config.weight_size_list = {self.config.weight_size_list}\n \
                    self.config.weight_size_no_embed_list = {self.config.weight_size_no_embed_list}\n \
                    self.config.ref_memory_list = {self.config.ref_memory_list}\n \
                    self.config.ref_weight_and_optimizer_memory = {self.config.ref_weight_and_optimizer_memory}\n \
                    self.config.ref_activation_memory = {self.config.ref_activation_memory}')

def main():
    args = parse_args()
    args, machine_topo = read_topo(args)
    config, config_dict = read_config_from_json(args, return_config_dict=True)
    args = config_to_args(config, config_dict, args)
    performance_model = HetacesoPerformanceModel(args, machine_topo, config, config_dict)
    performance_model.read_profiled()
    performance_model.predict_single_performance(0, False, False, True, True)

main()