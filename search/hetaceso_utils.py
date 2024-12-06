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
    cp_size_list = []
    usp_size_list = []
    rsp_size_list = []
    dp_size_list = []
    rsp_split_list = []
    dp_split_list = []
    gpu_list = []

    for i in range(config.num_stages):
        num_ops_stage.append(len(config.stages[i].ops))
        num_layers_stage.append(len(config.stages[i].ops)//num_ops_per_layer)
        tp_size_list.append(config.stages[i].tp_size)
        cp_size_list.append(config.stages[i].cp_size)
        usp_size_list.append(config.stages[i].usp_size)
        rsp_size_list.append(config.stages[i].rsp_size)
        dp_size_list.append(config.stages[i].dp_size)
        rsp_split_list.append(config.stages[i].rsp_split)
        dp_split_list.append(config.stages[i].dp_split)
        gpu_list.append(config.stages[i].num_gpus)

    print("{}|{:.2f}|{:.2f}| layer# = {} | op# = {} | tp = {} | cp = {} | usp = {} | rsp = {} | dp = {} | rsp_split = {} | dp_split = {} | gpus = {} | micro_bs = {} | time = {} | memory = {}| weight = {} | weight (no-embed) = {}\n".format(
        info, max(config.time_list), max(config.memory_list), num_layers_stage, num_ops_stage, tp_size_list, cp_size_list, usp_size_list, rsp_size_list, dp_size_list, rsp_split_list, dp_split_list, gpu_list, config.micro_bs, list(map(int, config.time_list)), list(map(int, config.memory_list)), list(map(int, config.weight_size_list)), list(map(int, config.weight_size_no_embed_list))))
    
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
        stages=stages_info_list,
        num_stages=num_stages,
    )
    return current_config

def read_config_from_json(config_file_name, return_config_dict=False):

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

    full_op_list = get_full_op_list(num_layers)
    config = get_config(
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
    )

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