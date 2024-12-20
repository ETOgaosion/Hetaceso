# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import time
import csv
import gc
import pickle
import argparse
from model_configs import model_prof_configs

configs = {
    "tp": [1, 2, 1, 1, 2, 1, 1, 1],
    "usp": [1, 1, 2, 1, 2, 2, 1, 4],
    "rsp": [1, 1, 1, 2, 1, 2, 4, 1],
    "dp": [4, 2, 2, 2, 1, 1, 1, 1],
    "mbs": {
        "350M": [8, 8, 8, 8, 8, 8, 8, 8],
        "1_3B": [8, 8, 8, 8, 8, 8, 8, 8],
        "2_6B": [8, 8, 8, 8, 8, 8, 8, 8],
        "6_7B": [8, 8, 8, 8, 4, 4, 4, 4],
        "13B": [8, 4, 4, 4, 2, 2, 2, 2],
    }
}

# model_size: (num_layers, total_seqlen, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": (24, 2048, 1024, 1024 * 4, 16, 1024 // 16, 51200, "fp16"),
    "1_3B": (24, 2048, 2048, 2048 * 4, 32, 2048 // 32, 51200, "fp16"),
    "2_6B": (32, 2048, 2560, 2560 * 4, 32, 2560 // 32, 51200, "fp16"),
    "6_7B": (32, 2048, 4096, 4096 * 4, 32, 4096 // 32, 51200, "fp16"),
    "13B": (40, 2048, 5120, 5120 * 4, 40, 5120 // 40, 51200, "fp16"),
    # "scale-layer": (1, 1024, 512, 512 * 4, 8, 512 // 8, 51200, "fp16"),
}

def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(
        torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(
        torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(
        torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(
        torch.cuda.max_memory_reserved() / mega_bytes)
    print("[Rank {}] {}".format(torch.distributed.get_rank(), string),
              flush=True)
    return string

def parse_args():
    parser = argparse.ArgumentParser(
        description="communication-profiler arguments", allow_abbrev=False
    )

    parser.add_argument(
        "--prof-tp-size", type=int, default=None, help="Profiler tp size."
    )
    parser.add_argument(
        "--prof-cp-size", type=int, default=None, help="Profiler cp size."
    )
    parser.add_argument(
        "--prof-dp-size", type=int, default=None, help="Profiler dp size."
    )
    parser.add_argument("--prof-path", type=str, default=None, help="")
    parser.add_argument("--prof-cache-file", type=str, default=None, help="")
    parser.add_argument("--prof-model-name", type=str, default="all", help="")
    parser.add_argument("--prof-model-size", type=str, default="all", help="")
    parser.add_argument("--prof-warmup-times", type=int, default=0, help="")
    parser.add_argument("--prof-repeat-times", type=int, default=1, help="")
    parser.add_argument("--prof-op-time-path", type=str, default=None, help="")
    parser.add_argument("--max-num-gpus", type=int, default=4, help="")
    parser.add_argument("--max-data-size", type=int, default=4096, help="")
    parser.add_argument("--prof-mbs-list", nargs="+", type=int, default=None, help="")
    parser.add_argument("--prof-seqlen-list", nargs="+", type=int, default=None, help="")

    args = parser.parse_args()
    return args


def print_rank0(str):
    if torch.distributed.get_rank() == 0:
        print(str)


def print_cached_dicts(cached_dict):
    for item in cached_dict:
        print(f"{item}: {cached_dict[item]}")

def get_torch_data_type(data_type):
    if data_type == "fp16":
        torch_data_type = torch.half
    elif data_type == "fp32":
        torch_data_type = torch.float
    else:
        raise RuntimeError(f"data type {data_type} not support.")
    return torch_data_type

def get_num_item_per_mb(torch_data_type):
    if torch_data_type == torch.half:
        num_item_per_mb = 1024 * 1024 / 2
    elif torch_data_type == torch.float:
        num_item_per_mb = 1024 * 1024 / 4
    else:
        raise RuntimeError(f"data type {torch_data_type} not support.")
    return num_item_per_mb

def load_data_size_list(args, torch_data_type, tp, cp, dp, model_size, cfg_i):
    data_size_list = []
    seq_len = gpt_configs[model_size][1] // cp
    mbs = configs["mbs"][model_size][cfg_i] // dp
    file_name = (
        args.prof_op_time_path
        + f"{model}_{model_size}_mbs{mbs}_seqlen{seq_len}_tp{tp}.csv"
    )
    print(file_name)
    num_item_per_mb = get_num_item_per_mb(torch_data_type)
    if os.path.exists(file_name):
        f_op_time = open(file_name, "r")
        f_csv = csv.reader(f_op_time)
        headers = next(f_csv)
        for row in f_csv:
            for index in [-3, -5]:
                data_size = int(float(row[index]) * num_item_per_mb)
                if data_size not in data_size_list and data_size > 0:
                    data_size_list.append(data_size)
    else:
        print(f"file {file_name} not exist.")
    return data_size_list

def all_to_all_single(args, data_size, world_size, torch_data_type, cp_group):
    if data_size % world_size == 0:
        send_tensors = [torch.ones(
            data_size, dtype=torch_data_type
        ).cuda()] * 3
    else:
        _data_size = (data_size // world_size) * world_size
        send_tensors = [torch.ones(
            _data_size, dtype=torch_data_type
        ).cuda()] * 3
    stream = torch.cuda.Stream()
    for _ in range(args.prof_warmup_times):
        a2a_reqs = [None] * 3
        for i in range(4):
            if 0 <= i < 3:
                send_tensor = send_tensors[i]
                output_tensor = torch.empty_like(send_tensor)
                a2a_reqs[i] = dist.all_to_all_single(output_tensor, send_tensor, group=cp_group, async_op=True)
            if i > 0:
                with torch.cuda.stream(stream):
                    if a2a_reqs[i - 1]:
                        a2a_reqs[i - 1].wait()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times):
        a2a_reqs = [None] * 3
        for i in range(4):
            if 0 <= i < 3:
                send_tensor = send_tensors[i]
                output_tensor = torch.empty_like(send_tensor)
                a2a_reqs[i] = dist.all_to_all_single(output_tensor, send_tensor, group=cp_group, async_op=True)
            if i > 0:
                with torch.cuda.stream(stream):
                    a2a_reqs[i - 1].wait()
    torch.cuda.current_stream().wait_stream(stream)
    end.record()
    torch.cuda.synchronize()
    for tensor in send_tensors:
        tensor.cpu()
    output_tensor.cpu()
    send_tensor.cpu()
    del send_tensors, output_tensor, send_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / args.prof_repeat_times

def all_gather_single(args, data_size, world_size, torch_data_type, dp_group):
    send_tensor = torch.ones(
        data_size, dtype=torch_data_type
    ).cuda()
    tensor_list = [
        torch.zeros(data_size, dtype=torch_data_type).cuda()
        for _ in range(world_size)
    ]
    for _ in range(args.prof_warmup_times):
        dist.all_gather(tensor_list, send_tensor, group=dp_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times):
        dist.all_gather(tensor_list, send_tensor, group=dp_group)
    end.record()
    torch.cuda.synchronize()
    send_tensor.cpu()
    for tensor in tensor_list:
        tensor.cpu()
    del send_tensor, tensor_list
    gc.collect()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / args.prof_repeat_times

def all_reduce_single(args, data_size, torch_data_type, dp_group):
    send_tensor = torch.ones(
        data_size, dtype=torch_data_type
    ).cuda()
    for _ in range(args.prof_warmup_times):
        dist.all_reduce(send_tensor, group=dp_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times):
        dist.all_reduce(send_tensor, group=dp_group)
    end.record()
    torch.cuda.synchronize()
    send_tensor.cpu()
    del send_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / args.prof_repeat_times

def reduce_scatter_single(args, data_size, world_size, torch_data_type, dp_group, dp_size):
    if data_size % world_size == 0:
        send_tensor = torch.ones(
            data_size, dtype=torch_data_type
        ).cuda()
    else:
        _data_size = (data_size // world_size) * world_size
        send_tensor = torch.ones(
            _data_size, dtype=torch_data_type
        ).cuda()
    for _ in range(args.prof_warmup_times):
        input_list = list(send_tensor.chunk(dp_size, 0))
        for idx, tensor in enumerate(input_list):
            if not tensor.is_contiguous():
                input_list[idx] = tensor.contiguous()
        new_input_ = torch.empty_like(input_list[0])
        dist.reduce_scatter(new_input_, input_list, group=dp_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times):
        input_list = list(send_tensor.chunk(dp_size, 0))
        for idx, tensor in enumerate(input_list):
            if not tensor.is_contiguous():
                input_list[idx] = tensor.contiguous()
        new_input_ = torch.empty_like(input_list[0])
        dist.reduce_scatter(new_input_, input_list, group=dp_group)
    end.record()
    torch.cuda.synchronize()
    send_tensor.cpu()
    for tensor in input_list:
        tensor.cpu()
    new_input_.cpu()
    del send_tensor, input_list, new_input_
    gc.collect()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / args.prof_repeat_times


def profile(rank, world_size, tp_size, usp_size, rsp_size, dp_size, data_size_list, model, size, torch_data_type):
    args = parse_args()
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    dist.init_process_group(
        backend="nccl", world_size=world_size, rank=rank, init_method=init_method
    )
    print(f'rank {rank} initialized', flush=True)
    cp_size = usp_size * rsp_size
    if usp_size > 1:
        initialized = False
        for i in range(tp_size):
            usp_group_start = i
            usp_group_end = world_size
            if rank in range(usp_group_start, usp_group_end, tp_size):
                usp_group = dist.new_group(list(range(usp_group_start, usp_group_end, tp_size)), use_local_synchronization=True)
                print(f'rank {rank} cp_ranks: {dist.get_process_group_ranks(usp_group)}', flush=True)
                initialized = True
                break
        assert initialized, f'rank {rank} usp_group not initialized'
    if dp_size > 1:
        initialized = False
        for i in range(tp_size * cp_size):
            dp_group_start = i
            dp_group_end = world_size
            if rank in range(dp_group_start, dp_group_end, tp_size * cp_size):
                dp_group = dist.new_group(list(range(dp_group_start, dp_group_end, tp_size * cp_size)), use_local_synchronization=True)
                print(f'rank {rank} dp_ranks: {dist.get_process_group_ranks(dp_group)}', flush=True)
                initialized = True
                break
        assert initialized, f'rank {rank} dp_group not initialized'

    if os.path.exists(args.prof_cache_file):
        cached_results = pickle.load(open(args.prof_cache_file, "rb"))
        profiled_results = cached_results["profiled_results"]
    else:
        profiled_results = {}

    torch.cuda.set_device(rank)
    if torch_data_type == torch.float:
        mb_per_item = 4 / (1024 * 1024)
    elif torch_data_type == torch.half:
        mb_per_item = 2 / (1024 * 1024)
    else:
        raise RuntimeError(f"type {torch_data_type} not support.")

    if model == "gpt":
        collectives = []
        if usp_size > 1:
            collectives.append("all_to_all")
        if dp_size > 1:
            collectives.extend(["all_reduce"])
        # collectives = ["all_gather", "all_reduce", "reduce_scatter"]
    else:
        raise RuntimeError(f"Model {model} is not supported.")

    for collective_type in collectives:
        avg_time_list = {}
        print(
            f"{dist.get_rank()} Start profiling {collective_type}... len(data_size_list) = {len(data_size_list)}", flush=True
        )
        for idx, data_size in enumerate(data_size_list):
            data_size_in_mb = int(data_size * mb_per_item)
            if collective_type in ["all_gather", "all_to_all"]:
                full_data_size_in_mb = data_size_in_mb * world_size
            elif collective_type in ["all_reduce", "reduce_scatter"]:
                full_data_size_in_mb = data_size_in_mb

            if data_size_in_mb not in avg_time_list:
                print(
                    f"[rank {rank}] {model}_{size} profiling {collective_type} tp{tp_size} usp{usp_size} rsp{rsp_size} dp{dp_size} ({world_size}GPUs) ({data_size} = {data_size_in_mb} MB) Progress: {idx / len(data_size_list)}\n", flush=True
                )
                # report_memory(f"{collective_type} {data_size_in_mb}")
                hash_name = f"{collective_type}_tp{tp_size}_up{usp_size}_rsp{rsp_size}_dp{dp_size}_{data_size_in_mb}_{torch_data_type}"
                if hash_name in profiled_results:
                    print_rank0(f"hit in cache!")
                    avg_time_list[data_size_in_mb] = profiled_results[hash_name]
                elif full_data_size_in_mb > args.max_data_size:
                    avg_time_list[data_size_in_mb] = 1000000000
                else:
                    dist.barrier()

                    try:
                        if collective_type == "all_reduce":
                            avg_time_list[data_size_in_mb] = all_reduce_single(args, data_size, torch_data_type, dp_group)
                            gc.collect()
                            torch.cuda.empty_cache()
                        elif collective_type == "all_to_all":
                            avg_time_list[data_size_in_mb] = all_to_all_single(args, data_size, world_size, torch_data_type, usp_group)
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            raise RuntimeError(f"collective {collective_type} not support.")
                    except RuntimeError as e:
                        print(e)

                    assert data_size_in_mb in avg_time_list and avg_time_list[data_size_in_mb] is not None, f"rank {rank} {collective_type} {data_size_in_mb} profiling failed."
                    profiled_results[hash_name] = avg_time_list[data_size_in_mb]
        if rank == 0:
            for data_size_in_mb in avg_time_list:
                print(
                    f"[{collective_type}] {data_size_in_mb} MB: {avg_time_list[data_size_in_mb]:.2f} ms", flush=True
                )
            result_title = ["data_size(MB)", "time(ms)"]
            save_file_name = (
                f"prim_{model}_{size}_tp{tp_size}_usp{usp_size}_rsp{rsp_size}_dp{dp_size}_{collective_type}.csv"
            )
            f_result = open(args.prof_path + save_file_name, "w")
            f_csv = csv.writer(f_result)
            f_csv.writerow(result_title)
            for data_size_in_mb in avg_time_list:
                tmp_row = [0, 0]
                tmp_row[0] = "{:.0f}".format(data_size_in_mb)
                tmp_row[1] = "{:.3f}".format(float(avg_time_list[data_size_in_mb]))
                f_csv.writerow(tmp_row)

    if rank == 0:
        save_dict = {}
        save_dict["profiled_results"] = profiled_results
        pickle.dump(save_dict, open(args.prof_cache_file, "wb"))


def run_profile(task):
    model = task["model"]
    size = task["size"]
    world_size = args.max_num_gpus
    if args.prof_mbs_list is None:
        if isinstance(model_prof_configs[model]["mbs"], dict):
            mbs_list = model_prof_configs[model]["mbs"][size]
        else:
            mbs_list = model_prof_configs[model]["mbs"]
    else:
        mbs_list = args.prof_mbs_list
    if model_prof_configs[model].get("seqlen") is not None:
        if isinstance(model_prof_configs[model]["seqlen"], dict):
            seqlen_list = model_prof_configs[model]["seqlen"][size]
        else:
            seqlen_list = model_prof_configs[model]["seqlen"]

    data_type = model_prof_configs[model]["dtype"]
    tp_size_list = []
    for tp in configs["tp"]:
        if tp not in tp_size_list:
            tp_size_list.append(tp)
    
    print(f'mbs_list: {mbs_list}, seqlen_list: {seqlen_list}, tp_size_list: {tp_size_list}')

    torch_data_type = get_torch_data_type(data_type)

    for i in range(len(configs["tp"])):
        tp_size = configs["tp"][i]
        usp_size = configs["usp"][i]
        rsp_size = configs["rsp"][i]
        dp_size = configs["dp"][i]
        data_size_list = load_data_size_list(args, torch_data_type, tp_size, usp_size * rsp_size, dp_size, size, i)
        print(f"tp_size: {tp_size}, usp_size: {usp_size}, dp_size: {dp_size}")
        if usp_size > 1 or dp_size > 1:
            torch.multiprocessing.spawn(
                profile,
                args=(world_size, tp_size, usp_size, rsp_size, dp_size, data_size_list, model, size, torch_data_type),
                nprocs=world_size,
                join=True,
            )


if __name__ == "__main__":

    start_profiling_time = time.time()
    args = parse_args()

    ## get profiling tasks
    ## "task"s are defined by unique {model, size} pairs
    all_prof_tasks = []
    model_names = (
        ["resnet", "gpt"] if args.prof_model_name == "all" else [args.prof_model_name]
    )
    for model in model_names:
        model_sizes = (
            model_prof_configs[model]["model_size"]
            if args.prof_model_size == "all"
            else [args.prof_model_size]
        )
        for size in model_sizes:
            all_prof_tasks.append({"model": model, "size": size})

    ## TODO: distribute profiling tasks if using multiple nodes
    print(all_prof_tasks)

    ## run profiling tasks
    for prof_task in all_prof_tasks:
        run_profile(prof_task)

    end_profiling_time = time.time()
    print(f"[TOTAL PROFILING TIME] {end_profiling_time - start_profiling_time:2f} s")
