import json
import jsbeautifier
from aceso_utils import gpt_configs
from dataclasses import dataclass, field
from typing import List
import copy
import os

LOG_LEVEL = int(os.environ.get("LOG_LEVEL", '0'))
num_ops_per_layer = 13
num_gpus_per_physical_node = 4

@dataclass 
class AcesoStageInfo:
    index: int
    num_stages_behind: int
    num_gpus: int
    ops: List[str]
    tp_size: List[int]
    dp_size: List[int]

@dataclass
class AcesoConfig:
    global_bs: int
    micro_bs: int
    num_micro_batches: int
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

    ## for elaceso

    num_layers_in_each_stage: List[int] = field(default_factory=list)
    weight_size_list: List[float] = field(default_factory=list)
    weight_size_no_embed_list: List[float] = field(default_factory=list)

    fwd_time_list: List[float] = field(default_factory=list)

def add_base_arguments(parser):
    parser.add_argument('--model-name', type=str, default=None, help='')
    parser.add_argument('--model-size', type=str, default=None, help='')
    parser.add_argument('--profiled-time-path', type=str, default=None, help='')
    parser.add_argument('--config-save-path', type=str, default=None, help='')
    parser.add_argument('--micro-batch-size', nargs='+', type=int, default=None, help='')

    parser.add_argument('--num-gpus-per-node', type=int, default=None, help='')
    parser.add_argument('--memory-limit', type=int, default=28000, help='')
    parser.add_argument('--recompute-mode', type=str, default='selective', help='')

    parser.add_argument('--inter-node-band', type=str, default=None, help='')
    parser.add_argument('--dist-optimizer', action='store_true', help='')

    # parser.add_argument('--no-embedding-replication', action='store_true', help='')

    return parser

def print_args(args):
    """Print arguments."""
    print('------------------------ arguments ------------------------',
            flush=True)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print('-------------------- end of arguments ---------------------',
            flush=True)

def print_simple_config_info(config, info=""):
    if config is None:
        return ""
    
    num_ops_stage = []
    num_layers_stage = []
    tp_size_list = []
    dp_size_list = []
    gpu_list = []

    for i in range(config.num_stages):
        num_ops_stage.append(len(config.stages[i].ops))
        num_layers_stage.append(len(config.stages[i].ops)//num_ops_per_layer)
        tp_size_list.append(config.stages[i].tp_size[0])
        dp_size_list.append(config.stages[i].dp_size[0])
        gpu_list.append(config.stages[i].num_gpus)

    print("{}|{:.2f}|{:.2f}| layer# = {} | op# = {} | tp = {} | dp = {} | gpus = {} | micro_bs = {} | time = {} | memory = {}| weight = {} | weight (no-embed) = {}\n".format(
        info, max(config.time_list), max(config.memory_list), num_layers_stage, num_ops_stage, tp_size_list, dp_size_list, gpu_list, config.micro_bs, list(map(int, config.time_list)), list(map(int, config.memory_list)), list(map(int, config.weight_size_list)), list(map(int, config.weight_size_no_embed_list))))
    
    return 

def get_op_list():
    return ["dec-embedding", "dec-self-attention", "dec-mlp", "dec-post-process"]

def get_full_op_list(num_layers):
    op_list = get_op_list()
    head_ops = [op_list[0]]
    decoder_layer = op_list[1:3]
    tail_ops = op_list[3:]
    full_op_list = head_ops + decoder_layer * num_layers + tail_ops
    return full_op_list
    

def get_config(num_ops_list, tp_size_list, dp_size_list, aggregate_mbs, global_batch_size, full_op_list, num_layers_in_each_stage):
    op_start_index = 0
    num_stages = len(num_ops_list)
    stages_info_list = []
    for i in range(num_stages):
        stage_info = AcesoStageInfo(
            index = i, 
            num_stages_behind = (num_stages - 1 - i),
            num_gpus = tp_size_list[op_start_index] * dp_size_list[op_start_index],
            ops = list(full_op_list[op_start_index: op_start_index + num_ops_list[i]]),
            tp_size = list(tp_size_list[op_start_index: op_start_index + num_ops_list[i]]),
            dp_size = list(dp_size_list[op_start_index: op_start_index + num_ops_list[i]]),
        )
        stages_info_list.append(stage_info)
        op_start_index += num_ops_list[i]

    current_config = AcesoConfig(global_bs=global_batch_size, micro_bs=aggregate_mbs, num_micro_batches=global_batch_size//aggregate_mbs, stages=stages_info_list, num_stages=num_stages, num_layers_in_each_stage=num_layers_in_each_stage)

    return current_config

def read_config_from_json(config_file_name, return_config_dict=False):

    with open(config_file_name, "r") as f:
        config_dict = json.load(f)

    num_layers = config_dict["num_layers"]
    aggregate_mbs = config_dict["micro_batch_size"]
    global_batch_size = config_dict["global_batch_size"]
    num_ops_list = config_dict["num_ops_in_each_stage"]
    num_layers_in_each_stage = config_dict["num_layers_in_each_stage"]

    tp_size_list = []
    for _tp_size_list in config_dict["model_parallel_size_of_each_op"]:
        tp_size_list += _tp_size_list

    dp_size_list = []
    for _dp_size_list in config_dict["data_parallel_size_of_each_op"]:
        dp_size_list += _dp_size_list

    full_op_list = get_full_op_list(num_layers)
    config = get_config(num_ops_list, tp_size_list, dp_size_list, aggregate_mbs, global_batch_size, full_op_list, num_layers_in_each_stage)

    if return_config_dict:
        return config, config_dict
    else:
        return config

def dump_config_to_json(config, file_name, model_name, model_size, num_layers):

    config_dict = {}
    config_dict["config_version"] = "v3" 
    config_dict["model_name"] = model_name
    config_dict["model_size"] = model_size

    if model_name == "gpt":
        _, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, _ = gpt_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["seq_length"] = seq_len
        config_dict["max_position_embeddings"] = seq_len
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["hidden_size"] = hidden_size        
    else:
        raise RuntimeError(f"{model_name} not supportted.")

    config_dict["global_batch_size"] = config.global_bs
    config_dict["micro_batch_size"] = config.micro_bs
    config_dict["num_stages"] = config.num_stages
    config_dict["device_mapping"] = config.device_mapping

    tp_size_of_each_op = []
    dp_size_of_each_op = []
    num_ops_in_each_stage = []
    num_layers_in_each_stage = []
    config_dict["num_gpus"] = []
    config_dict["resharding_stages"] = []
    for i in range(config.num_stages):
        tp_size_of_each_op.append(config.stages[i].tp_size)
        dp_size_of_each_op.append(config.stages[i].dp_size)
        num_ops_in_each_stage.append(len(config.stages[i].ops))
        num_layers_in_each_stage.append(len(config.stages[i].ops) // num_ops_per_layer)

        config_dict["num_gpus"].append(config.stages[i].num_gpus)
        if max(config.stages[i].tp_size) != min(config.stages[i].tp_size) \
            or max(config.stages[i].dp_size) != min(config.stages[i].dp_size):
            config_dict["resharding_stages"].append(True)
        else:
            config_dict["resharding_stages"].append(False)

    config_dict["num_ops_in_each_stage"] = num_ops_in_each_stage
    config_dict["num_layers_in_each_stage"] = num_layers_in_each_stage
    config_dict["model_parallel_size_of_each_op"] = tp_size_of_each_op
    config_dict["data_parallel_size_of_each_op"] = dp_size_of_each_op

    # print(f"[WARNING] should use jsbeautifier")
    # json.dump(config_dict, open(file_name, 'w'), separators=(',', ':'), indent=4)

    config_dict["memory_list"] = [f"{t:.0f}" for t in config.memory_list] 
    config_dict["time_list"] = [f"{t:.0f}" for t in config.time_list]

    config_dict["fwd_time_list"] = [f"{t:.0f}" for t in config.fwd_time_list]

    options = jsbeautifier.default_options()
    options.indent_size = 2
    beautiful_json = jsbeautifier.beautify(json.dumps(config_dict), options)

    with open(file_name, 'w') as f:
        f.write(beautiful_json)

    print(f"config has been saved to {file_name}\n")


def get_num_gpus(config):
    num_gpus = 0
    for stage_info in config.stages:
        num_gpus += stage_info.num_gpus
    return num_gpus