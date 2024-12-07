# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import csv
import datetime
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import List
from model_ops_info import get_full_op_list

# model_size: (num_layers, total_seqlen, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": (24, 2048, 1024, 1024 * 4, 16, 1024 // 16, 51200, "fp16"),
    "1_3B": (24, 2048, 2048, 2048 * 4, 32, 2048 // 32, 51200, "fp16"),
    "2_6B": (32, 2048, 2560, 2560 * 4, 32, 2560 // 32, 51200, "fp16"),
    "6_7B": (32, 2048, 4096, 4096 * 4, 32, 4096 // 32, 51200, "fp16"),
    "13B": (40, 2048, 5120, 5120 * 4, 40, 5120 // 40, 51200, "fp16"),
    # "scale-layer": (1, 1024, 512, 512 * 4, 8, 512 // 8, 51200, "fp16"),
}

## NOTE: For GPT and T5 models, we use fp16, which will introduce a "main_param" in Megatron
memory_ratio = {
    "resnet": {"main_params": 0, "optimizer": 2},
    "gpt": {"main_params": 2, "optimizer": 4},
    "t5": {"main_params": 2, "optimizer": 4},
}

MAX_VALUE = 2**30
MIN_VALUE = -(2**30)
GLOBAL_TIMER = None


@dataclass
class AcesoStageInfo:
    index: int
    num_stages_behind: int
    num_gpus: int
    ops: List[str]
    tp_size: int
    cp_size: int
    usp_size: int
    rsp_size: int
    dp_size: int
    rsp_split: List[int]
    dp_split: List[int]


@dataclass
class AcesoConfig:
    global_bs: int
    micro_bs: int
    total_seqlen: int
    stages: List[AcesoStageInfo]
    num_stages: int
    history: str = ""

    time_list: List[float] = field(default_factory=list)
    memory_list: List[float] = field(default_factory=list)
    compute_time_list: List[float] = field(default_factory=list)
    total_gpu_time: float = 0

    breakdown_ideal_time_per_gpu: List[float] = field(default_factory=list)
    breakdown_eff_loss_time_per_gpu: List[float] = field(default_factory=list)

    ## for choosing partner stages according to efficient time
    efficient_time_list: List[float] = field(default_factory=list)
    ## used for adaptive model
    adaptive_times: int = 0


def debug_info(info, print_debug_info):
    if print_debug_info:
        print(info)


def get_config(
    num_layers,
    total_seqlen,
    global_batch_size,
    aggregate_mbs,
    
    num_stages,
    num_gpu_list,
    num_ops_list,
    
    tp_size_list,
    cp_size_list,
    usp_size_list,
    rsp_size_list,
    dp_size_list,
    rsp_split_list,
    dp_split_list,
    full_op_list,
):
    op_start_index = 0
    stages_info_list = []
    assert num_layers * 2 + 2 == sum(num_ops_list), f"num_layers: {num_layers} not match num_ops_list: {num_ops_list}"
    assert num_stages == len(num_gpu_list) == len(num_ops_list) == len(tp_size_list) == len(cp_size_list) == len(usp_size_list) == len(rsp_size_list) == len(dp_size_list) == len(rsp_split_list), f"num_stages: {num_stages} not match num_gpu_list: {num_gpu_list}, num_ops_list: {num_ops_list}, tp_size_list: {tp_size_list}, cp_size_list: {cp_size_list}, usp_size_list: {usp_size_list}, rsp_size_list: {rsp_size_list}, dp_size_list: {dp_size_list}, rsp_split_list: {rsp_split_list}"
    for i in range(num_stages):
        assert num_gpu_list[i] == tp_size_list[i] * cp_size_list[i] * dp_size_list[i], f'3d parallelism mul is not equal to num gpus: {num_gpu_list[i]} != {tp_size_list[i]} * {cp_size_list[i]} * {dp_size_list[i]}'
        assert cp_size_list[i] == usp_size_list[i] * rsp_size_list[i], f'context parallelism mul is not equal to num gpus: {cp_size_list[i]} != {usp_size_list[i]} * {rsp_size_list[i]}'
        assert len(rsp_split_list[i]) == rsp_size_list[i], f'rsp split list format error, {len(rsp_split_list)}, {rsp_size_list[i]}'
        assert total_seqlen == sum(rsp_split_list[i]), f'sum of rsp split is not equal to total seqlen'
        assert aggregate_mbs == sum(dp_split_list[i]), f'sum of dp split is not equal to total mbs'
        stage_info = AcesoStageInfo(
            index=i,
            num_stages_behind=(num_stages - 1 - i),
            num_gpus=num_gpu_list[i],
            ops=list(full_op_list[op_start_index : op_start_index + num_ops_list[i]]),
            tp_size=tp_size_list[i],
            cp_size=cp_size_list[i],
            usp_size=usp_size_list[i],
            rsp_size=rsp_size_list[i],
            dp_size=dp_size_list[i],
            rsp_split=rsp_split_list[i],
            dp_split=dp_split_list[i]
        )
        stages_info_list.append(stage_info)
        op_start_index += num_ops_list[i]

    current_config = AcesoConfig(
        global_bs=global_batch_size,
        micro_bs=aggregate_mbs,
        total_seqlen=total_seqlen,
        stages=stages_info_list,
        num_stages=num_stages,
    )
    return current_config


def config_details(config, get_string=False):
    if config is None:
        return ""
    num_ops_stage = []
    num_gpu_list = []
    tp_size_list = []
    cp_size_list = []
    usp_size_list = []
    rsp_size_list = []
    dp_size_list = []
    rsp_split_list = []
    dp_split_list = []
    total_mbs = config.micro_bs
    for i in range(config.num_stages):
        num_ops_stage.append(len(config.stages[i].ops))
        num_gpu_list.append(config.stages[i].num_gpus)
        tp_size_list.append(config.stages[i].tp_size)
        cp_size_list.append(config.stages[i].cp_size)
        usp_size_list.append(config.stages[i].usp_size)
        rsp_size_list.append(config.stages[i].rsp_size)
        dp_size_list.append(config.stages[i].dp_size)
        rsp_split_list.append(config.stages[i].rsp_split)
        dp_split_list.append(config.stages[i].dp_split)
    if get_string:
        return f"{num_ops_stage}, {num_gpu_list}, {total_mbs}, {tp_size_list}, {cp_size_list}, {usp_size_list}, {rsp_size_list}, {dp_size_list}, {rsp_split_list}, {dp_split_list}"
    else:
        return num_ops_stage, num_gpu_list, total_mbs, tp_size_list, cp_size_list, usp_size_list, rsp_size_list, dp_size_list, rsp_split_list, dp_split_list


def dump_config_to_json(config, file_name, args):
    model_name = args.model_name
    model_size = args.model_size
    assert model_name in ["gpt"]
    num_layers = args.num_layers
    config_dict = {}
    config_dict["model_name"] = model_name
    config_dict["model_size"] = model_size
    if model_name == "gpt":
        (
            _,
            seq_len,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            kv_channels,
            vocab_size,
            _,
        ) = gpt_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["total_seqlen"] = seq_len
        config_dict["max_position_embeddings"] = seq_len
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["hidden_size"] = hidden_size
    else:
        raise RuntimeError(f"{model_name} not supportted.")

    config_dict["global_batch_size"] = config.global_bs
    config_dict["micro_batch_size"] = config.micro_bs
    config_dict["num_stages"] = config.num_stages
    config_dict["num_gpus"] = []
    
    num_ops_in_each_stage = []
    
    tp_size_of_each_stage = []
    cp_size_of_each_stage = []
    usp_size_of_each_stage = []
    rsp_size_of_each_stage = []
    dp_size_of_each_stage = []
    rsp_split_list_of_each_stage = []
    dp_split_list_of_each_stage = []
    
    for i in range(config.num_stages):
        tp_size_of_each_stage.append(config.stages[i].tp_size)
        cp_size_of_each_stage.append(config.stages[i].cp_size)
        usp_size_of_each_stage.append(config.stages[i].usp_size)
        rsp_size_of_each_stage.append(config.stages[i].rsp_size)
        dp_size_of_each_stage.append(config.stages[i].dp_size)
        rsp_split_list_of_each_stage.append(config.stages[i].rsp_split)
        dp_split_list_of_each_stage.append(config.stages[i].dp_split)
        num_ops_in_each_stage.append(len(config.stages[i].ops))

        config_dict["num_gpus"].append(config.stages[i].num_gpus)

    config_dict["num_ops_in_each_stage"] = num_ops_in_each_stage
    config_dict["tensor_parallel_size_of_each_stage"] = tp_size_of_each_stage
    config_dict["context_parallel_size_of_each_stage"] = cp_size_of_each_stage["ulysses_context_parallel_size_of_each_stage"] = usp_size_of_each_stage
    config_dict["ring_context_parallel_size_of_each_stage"] = rsp_size_of_each_stage
    config_dict
    config_dict["data_parallel_size_of_each_stage"] = dp_size_of_each_stage
    config_dict["ring_context_parallel_split_of_each_stage"] = rsp_split_list_of_each_stage
    config_dict["data_parallel_split_of_each_stage"] = dp_split_list_of_each_stage

    json.dump(config_dict, open(file_name, "w"), indent=4)
    print(f"config has been saved to {file_name}")


def read_config_from_json(args, return_config_dict=False):
    config_file_name = args.initial_point
    with open(config_file_name, "r") as f:
        config_dict = json.load(f)

    model_name = config_dict["model_name"]
    model_size = config_dict["model_size"]
    
    num_layers = config_dict["num_layers"]
    total_seqlen = config_dict["total_seqlen"]
    aggregate_mbs = config_dict["micro_batch_size"]
    global_batch_size = config_dict["global_batch_size"]
    
    num_stages = config_dict["num_stages"]
    num_gpus = config_dict["num_gpus"]
    num_ops_list = config_dict["num_ops_in_each_stage"]
    
    tp_size_list = config_dict["tensor_parallel_size_of_each_stage"]
    cp_size_list = config_dict["context_parallel_size_of_each_stage"]
    usp_size_list = config_dict["ulysses_context_parallel_size_of_each_stage"]
    rsp_size_list = config_dict["ring_context_parallel_size_of_each_stage"]
    rsp_split_list = config_dict["ring_context_parallel_split_of_each_stage"]
    dp_size_list = config_dict["data_parallel_size_of_each_stage"]
    dp_split_list = config_dict["data_parallel_split_of_each_stage"]
    
    full_op_list = get_full_op_list(args)

    if return_config_dict:
        return (
            get_config(
                num_layers,
                total_seqlen,
                global_batch_size,
                aggregate_mbs,
                num_stages,
                num_gpus,
                num_ops_list,
                tp_size_list,
                cp_size_list,
                usp_size_list,
                rsp_size_list,
                dp_size_list,
                rsp_split_list,
                dp_split_list,
                full_op_list,
            ),
            config_dict,
        )
    else:
        return get_config(
            num_layers,
            total_seqlen,
            global_batch_size,
            aggregate_mbs,
            num_stages,
            num_gpus,
            num_ops_list,
            tp_size_list,
            usp_size_list,
            cp_size_list,
            rsp_size_list,
            dp_size_list,
            rsp_split_list,
            dp_split_list,
            full_op_list,
        )


def save_config_info_to_csv(config, reserved_mem_list, dir_name):
    info_to_csv = [
        ["stage-index", "time", "memory(total)", "memory(normal)", "memory(reserved)"]
    ]
    for i in range(config.num_stages):
        info_to_csv.append(
            [
                f"stage-{i}",
                f"{config.time_list[i]:.2f}",
                f"{config.memory_list[i]:.2f}",
                f"{(config.memory_list[i]-reserved_mem_list[i]):.2f}",
                f"{reserved_mem_list[i]:.2f}",
            ]
        )

    with open(dir_name + 'res.csv', mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)


def save_distribution_to_csv(num_targets_list, num_hops_list, file_name):
    info_to_csv = [["num_targets"] + num_targets_list, ["num_hops"] + num_hops_list]

    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)


class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.count = 0
        self.start_time = time.time()
        self.elapsed_history = 0.0
        self.elapsed_list = []

    def start(self):
        """Start the timer."""
        assert not self.started_, f"{self.name_} timer has already been started"
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, "timer is not started"
        elapsed_time = time.time() - self.start_time
        self.elapsed_ += elapsed_time
        self.elapsed_list.append(elapsed_time)
        self.started_ = False
        self.count += 1

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self.count = 0
        self.elapsed_history = time.time() - self.start_time

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
            self.count = 0
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_

    def elapsed_since_first_invoke(self, reset=True):
        """Calculate the elapsed time."""
        return time.time() - self.start_time


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + "-time", value, iteration)

    def log(self, reset=True):
        print(f"===== timers ====")
        for name in sorted(self.timers):
            count = self.timers[name].count
            if count == 0:
                elapsed_time = self.timers[name].elapsed_history
            else:
                elapsed_time = self.timers[name].elapsed(reset=reset)
            print("{}: {:.2f} s [count = {}]".format(name, elapsed_time, count))


def print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


def add_model_args(parser):
    group = parser.add_argument_group(title="model information")
    group.add_argument("--model-name", type=str, default="gpt", help="")
    group.add_argument("--model-size", type=str, default=None, help="")
    group.add_argument("--num-layers", type=int, default=None, help="")
    group.add_argument("--global-batch-size", type=int, default=None, help="")
    group.add_argument("--micro-batch-size", nargs="+", type=int, default=None, help="")
    group.add_argument("--seq-len", type=int, default=2048, help="")
    group.add_argument("--decoder-seq-len", type=int, default=512, help="")
    group.add_argument("--max-tp", type=int, default=None, help="")
    group.add_argument('--dist-optimizer', action='store_true', help='')

    return parser


def add_hardware_args(parser):
    group = parser.add_argument_group(title="hardware information")
    group.add_argument("--num-nodes", type=int, default=None, help="")
    group.add_argument("--node-rank", type=int, default=None, help="rank of this node to estimate")
    group.add_argument("--num-gpus-per-node", type=int, default=None, help="")
    group.add_argument("--memory-limit", type=int, default=28000, help="")

    return parser


def add_path_args(parser):
    group = parser.add_argument_group(title="path information")
    group.add_argument("--log-path", type=str, default=None, help="")
    group.add_argument("--profiled-gpt-path", type=str, default=None, help="")
    group.add_argument("--profiled-dist-p2p-path", type=str, default=None, help="")
    group.add_argument("--profiled-local-p2p-path", type=str, default=None, help="")
    group.add_argument("--profiled-local-comm-path", type=str, default=None, help="")
    group.add_argument("--config-save-path", type=str, default=None, help="")
    group.add_argument("--config-suffix", type=str, default=None, help="")

    return parser


def add_budget_args(parser):
    group = parser.add_argument_group(title="budgets for the search algorithm")
    group.add_argument("--max-num-hops", type=int, default=None, help="")
    group.add_argument("--max-num-trials", type=int, default=100, help="")
    group.add_argument("--time-budget-per-trial", type=int, default=None, help="")
    group.add_argument("--time-budget-total", type=int, default=10, help="")
    group.add_argument("--start-num-stages", type=int, default=None, help="")
    group.add_argument("--end-num-stages", type=int, default=None, help="")

    return parser


def add_heuristic_args(parser):
    group = parser.add_argument_group(title="heuristics in the search algorithm")
    group.add_argument("--op-group-size", type=int, default=1, help="")
    group.add_argument("--max-op-move-steps", type=int, default=5, help="")
    group.add_argument("--memory-pred-type", type=str, default="MAX", help="")
    group.add_argument("--initial-point", type=str, default="balance", help="")
    group.add_argument("--high-memory-rate", type=float, default=0.9, help="")

    return parser


def add_debug_args(parser):
    group = parser.add_argument_group(title="debug arguments")
    group.add_argument("--print-debug-info", action="store_true", help="")
    group.add_argument("--print-move-op-details", action="store_true", help="")

    return parser


def add_test_args(parser):
    group = parser.add_argument_group(title="arguments under development")
    group.add_argument(
        "--do-not-consider-shared-tensor-space",
        action="store_false",
        help="",
        dest="consider_shared_space",
    )
    group.add_argument(
        "--do-not-consider-reserved-space",
        action="store_false",
        help="",
        dest="consider_reserved_space",
    )
    group.add_argument("--predict-delta-time", action="store_true", help="")
    group.add_argument("--add-action-finetune-dim", action="store_true", help="")
    group.add_argument("--add-action-tp-dp-exchange", action="store_true", help="")
    group.add_argument("--peak-mem-in-backward", type=int, default=0, help="")
    group.add_argument("--add-action-tune-tp-dp", action="store_true", help="")
    group.add_argument("--finetune-after-trial", type=int, default=0, help="")
    group.add_argument(
        "--no-multi-process", action="store_false", help="", dest="multi_process"
    )
    group.add_argument("--random-order-actions", action="store_true", help="")
    group.add_argument("--support-comm-predict", action="store_true", help="")
    group.add_argument("--forbid-turn-back", action="store_true", help="")
    group.add_argument("--sort-metric", type=str, default="max_stage_time", help="")
    group.add_argument("--print-gpu-mig-details", action="store_true", help="")
    group.add_argument("--finetune-tp-dp-after-trial", action="store_true", help="")
    group.add_argument("--init-dim", type=str, default="tp", help="")
    group.add_argument("--num-partners-in-op-mig", type=int, default=1, help="")
    group.add_argument(
        "--do-not-continue-when-fail",
        action="store_false",
        help="",
        dest="continue_when_fail",
    )
    group.add_argument("--adaptive-hyper-parameters", type=int, default=5, help="")
    group.add_argument("--num-of-saved-configs", type=int, default=1, help="")
    group.add_argument("--simple-prim-mbs", action="store_true", help="")
    group.add_argument("--simple-prim-mig", action="store_true", help="")
    group.add_argument("--only-top-1-target", action="store_true", help="")
    group.add_argument("--consider-collective-memory", action="store_true", help="")
    group.add_argument("--save-to-csv", type=str, default=None, help="")

    return parser


global_args = None


def parse_args():
    global global_args
    if global_args is not None:
        return global_args

    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_hardware_args(parser)
    parser = add_path_args(parser)
    parser = add_budget_args(parser)
    parser = add_heuristic_args(parser)
    parser = add_debug_args(parser)
    parser = add_test_args(parser)

    args = parser.parse_args()

    args.num_gpus = args.num_gpus_per_node * args.num_nodes

    if os.path.exists(args.initial_point):
        with open(args.initial_point, "r") as f:
            config_dict = json.load(f)
            args.model_name = config_dict["model_name"]
            args.model_size = config_dict["model_size"]

    if args.model_name not in ["resnet", "gpt", "t5", "scale-layer"]:
        raise RuntimeError(f"model {args.model_name} is not supported yet.")

    if args.num_layers is None:
        if args.model_name == "gpt":
            args.num_layers = gpt_configs[args.model_size][0]
        elif args.model_name == "scale-layer":
            raise RuntimeError(f"should provide --num-layers for scale-layer exp")

    if args.micro_batch_size is None:
        if args.model_name in ["resnet"]:
            args.micro_batch_size = [16, 32, 48, 64]
        else:
            args.micro_batch_size = [1, 2, 4, 8]

    if args.start_num_stages is None or args.end_num_stages is None:
        args.start_num_stages = 1
        args.end_num_stages = min(args.num_gpus, 16)
    if args.max_tp is None:
        args.max_tp = args.num_gpus_per_node

    if args.time_budget_per_trial is None:
        assert (
            args.time_budget_total is not None
        ), "a time budget should be given, with --time-budget-total"
        args.time_budget_per_trial = args.time_budget_total

    args.min_mbs = min(args.micro_batch_size)

    if args.model_name == "scale-layer":
        args.memory_main_params = memory_ratio["gpt"]["main_params"]
        args.memory_optimizer = memory_ratio["gpt"]["optimizer"]
    else:
        args.memory_main_params = memory_ratio[args.model_name]["main_params"]
        args.memory_optimizer = memory_ratio[args.model_name]["optimizer"]

    if args.model_name not in ["t5"]:
        args.resharding = True
    else:
        args.resharding = False

    cur_time = datetime.datetime.now()
    args.config_suffix = f"{cur_time.year}-{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}-{cur_time.second}"
    global_args = args
    return args


def update_args(new_args):
    global global_args
    global_args = new_args


def sort_configs(config_list, sort_metric):
    new_list = []

    if sort_metric == "max_stage_time":
        for i in range(len(config_list)):
            max_stage_time = max(config_list[i].time_list)
            if len(new_list) == 0:
                new_list.append(config_list[i])
            else:
                j = 0
                inserted = False
                for j in range(len(new_list)):
                    if max_stage_time < max(new_list[j].time_list):
                        new_list.insert(j, config_list[i])
                        inserted = True
                        break
                if not inserted:
                    new_list.append(config_list[i])
    elif sort_metric == "total_gpu_time":
        for i in range(len(config_list)):
            gpu_time = config_list[i].total_gpu_time
            if len(new_list) == 0:
                new_list.append(config_list[i])
            else:
                j = 0
                inserted = False
                for j in range(len(new_list)):
                    if gpu_time < new_list[j].total_gpu_time:
                        new_list.insert(j, config_list[i])
                        inserted = True
                        break
                if not inserted:
                    new_list.append(config_list[i])
    else:
        raise RuntimeError(f"sort_metric {sort_metric} not supported.")
    return new_list


def check_legality(config, args):
    num_gpus_from_start = []
    num_gpus = 0
    for i in range(config.num_stages):
        if config.stages[i].num_gpus not in [1, 2, 4, 8]:
            return False
        num_gpus += config.stages[i].num_gpus
        num_gpus_from_start.append(num_gpus)
    if num_gpus != args.num_gpus:
        return False
    num_gpus_at_boundary_list = [
        args.num_gpus_per_node * i for i in range(1, args.num_nodes + 1)
    ]
    for num_gpus_at_boundary in num_gpus_at_boundary_list:
        if num_gpus_at_boundary not in num_gpus_from_start:
            return False
    return True


def format_size_list(size_list):
    output_string = "["
    for i in range(len(size_list)):
        max_val = max(size_list[i])
        min_val = min(size_list[i])
        if min_val == max_val:
            output_string += f"{max_val}, "
        else:
            output_string += f"{min_val}~{max_val}, "
    output_string += "]"
    return output_string


def print_simple_config_info(
    config,
    info="",
    add_history=False,
    print_debug_info=False,
):
    if config is None:
        return

    (
        num_ops_stage,
        _,
        tp_per_stage,
        cp_per_stage,
        usp_per_stage,
        rsp_per_stage,
        dp_per_stage,
        rsp_split_list,
        dp_split_list,
        base_batch_size,
    ) = config_details(config)
    gpu_list = [config.stages[i].num_gpus for i in range(config.num_stages)]
    detailed_tp_size = format_size_list(tp_per_stage)
    detailed_cp_size = format_size_list(cp_per_stage)
    detailed_usp_size = format_size_list(usp_per_stage)
    detailed_rsp_size = format_size_list(rsp_per_stage)
    detailed_dp_size = format_size_list(dp_per_stage)

    history = "{}|{:.2f}|{:.2f}| op = {}| tp = {} | cp = {} | usp = {} | rsp = {} | dp = {} | rsp_split = {} | dp_split = {} | gpus = {} | micro_bs = {} | time = {} | memory = {}".format(
        info,
        max(config.time_list),
        max(config.memory_list),
        num_ops_stage,
        detailed_tp_size,
        detailed_cp_size,
        detailed_usp_size,
        detailed_rsp_size,
        detailed_dp_size,
        rsp_split_list,
        dp_split_list,
        gpu_list,
        base_batch_size,
        list(map(int, config.time_list)),
        list(map(int, config.memory_list)),
    )
    debug_info(history, print_debug_info)
    if add_history:
        config.history += history + "\n"

    return


def is_visited(visited_set, hash_str, target=""):
    if hash_str not in visited_set:
        return False
    else:
        if target in visited_set[hash_str]:
            return True
        else:
            return False


def mark_visited(visited_set, hash_str, target=""):
    if hash_str not in visited_set:
        visited_set[hash_str] = [target]
    else:
        visited_set[hash_str].append(target)


def num_visited(visited_set, hash_str):
    if hash_str not in visited_set:
        return 0
    else:
        return len(visited_set[hash_str])


def save_search_trend_in_csv(search_time_list, exec_time_list, file_name):
    result_list = [["search_time (s)", "config time (ms)"]]
    assert len(search_time_list) == len(exec_time_list)
    for i in range(len(search_time_list)):
        new_line = [search_time_list[i], exec_time_list[i]]
        result_list.append(new_line)

    with open(file_name, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(result_list)


def save_and_print_top_configs(result_dict, args):

    sorted_time_list = [MAX_VALUE]
    sorted_config_stage_list = [MAX_VALUE]
    for i in range(args.start_num_stages, args.end_num_stages + 1):
        if i in result_dict and result_dict[i] is not None:
            config_time, config_mem, explored_cases, search_time, case_distribution = (
                result_dict[i]
            )
            if config_time > 0:
                for j in range(len(sorted_time_list)):
                    current_config_time = sorted_time_list[j]
                    if config_time < current_config_time:
                        sorted_time_list.insert(j, config_time)
                        sorted_config_stage_list.insert(j, i)
                        break
    sorted_config_stage_list.pop()
    sorted_time_list.pop()

    ##### save configs:
    data = []
    for i in range(
        min(
            (args.end_num_stages - args.start_num_stages + 1),
            args.num_of_saved_configs,
            len(sorted_time_list),
        )
    ):
        stage_num = sorted_config_stage_list[i]
        src_file = f"{args.config_save_path}{args.model_name}_{args.model_size}_{stage_num}stages_{args.config_suffix}.json"
        dst_file = f"{args.config_save_path}top_configs/{args.model_name}_{args.model_size}_{stage_num}stages_{args.config_suffix}.json"
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)

        config_time, config_mem, explored_cases, search_time, case_distribution = (
            result_dict[stage_num]
        )
        config_thpt = args.global_batch_size / (config_time / 1000)
        data.append(
            [
                stage_num,
                f"{config_time/1000:.2f}",
                f"{config_thpt:.2f}",
                f"{config_mem:.0f}",
                f"{explored_cases}",
            ]
        )

    header = [
        "# of stages",
        "est_iteration_time(s)",
        "est_thpt(samples/s)",
        "est_mem(MB)",
        "# of explored cases",
    ]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print("\t".join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))

    for row in data:
        print("\t".join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))


def print_search_details(
    config,
    args,
    num_stages,
    num_targets_list,
    num_hops_list,
    search_time_list,
    config_time_list,
    reserved_mem_list,
    num_explored_cases,
):
    print(f"\n========== Best Result (num_stages = {num_stages}) ==========")
    if config is not None:
        print_simple_config_info(
            config, print_debug_info=True
        )
        print(config.history)
        print(f"num_targets: {num_targets_list}")
        print(f"num_hops: {num_hops_list}")
        save_config_info_to_csv(
            config,
            reserved_mem_list,
            f"{args.config_save_path}csv/info_{args.model_name}_{args.model_size}_{config.num_stages}stages_{args.config_suffix}.csv",
        )
        save_distribution_to_csv(
            num_targets_list,
            num_hops_list,
            f"{args.log_path}trends/distribution_{args.model_name}_{args.model_size}_{config.num_stages}stages_{args.config_suffix}.csv",
        )
    else:
        print("No feasible solution.")

    sum_time = 0
    accum_list = []
    for i in range(len(search_time_list)):
        if i == 0:
            accum_list.append(0)
        else:
            time_one_trial = search_time_list[i]
            sum_time += int(time_one_trial + 1)
            accum_list.append(sum_time)

    print(
        f"search time = {sum(search_time_list):.2f} s.  {accum_list} \nnum_explored_cases = {num_explored_cases}"
    )
    print(f"time trend: {list(map(int, config_time_list))}\n")
    save_search_trend_in_csv(
        accum_list,
        config_time_list,
        f"{args.log_path}trends/{args.model_name}_{args.model_size}_{num_stages}stages_init_{args.initial_point}_{args.max_num_hops}hops_{args.config_suffix}.csv",
    )


## some globally used values
timers = Timers()
