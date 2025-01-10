# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import dataclasses
import time
import csv
import gc
import pickle
import argparse
import matplotlib.pyplot as plt
from numpy import polyfit, polyval
import numpy as np
from model_configs import model_prof_configs

@dataclasses.dataclass
class GPTConfig:
    num_layers: int
    max_seqlen: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    kv_channels: int
    vocab_size: int
    params_dtype: torch.dtype

# model_size: (num_layers, total max_seqlen, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": GPTConfig(24, 65546, 1024, 4096, 16, 64, 51200, torch.float16),
    "1_3B": GPTConfig(24, 65546, 2048, 8192, 32, 64, 51200, torch.float16),
    "2_6B": GPTConfig(32, 32768, 2560, 10240, 32, 80, 51200, torch.float16),
    "6_7B": GPTConfig(32, 16384, 4096, 16384, 32, 128, 51200, torch.float16),
    "13B": GPTConfig(40, 16384, 5120, 20480, 40, 128, 51200, torch.float16),
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
        "--prof-tp-size", type=int, default=1, help="Profiler tp size."
    )
    parser.add_argument(
        "--prof-usp-size", type=int, default=1, help="Profiler usp size."
    )
    parser.add_argument(
        "--prof-rsp-size", type=int, default=1, help="Profiler rsp size."
    )
    parser.add_argument(
        "--prof-dp-size", type=int, default=1, help="Profiler dp size."
    )
    parser.add_argument("--prof-path", type=str, default=None, help="")
    parser.add_argument("--prof-fig-path", type=str, default=None, help="")
    parser.add_argument("--prof-cache-file", type=str, default=None, help="")
    parser.add_argument("--prof-model-name", type=str, default="all", help="")
    parser.add_argument("--prof-model-size", type=str, default="all", help="")
    parser.add_argument("--prof-mbs-list", type=int, nargs='+', default=[2], help="")
    parser.add_argument("--prof-basic-seqlen", type=int, default=1024, help="")
    parser.add_argument("--use-square-scope", action='store_true')
    parser.add_argument("--prof-warmup-times", type=int, default=0, help="")
    parser.add_argument("--prof-repeat-times", type=int, default=1, help="")
    parser.add_argument("--max-num-gpus", type=int, default=4, help="")

    args = parser.parse_args()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    return args


def print_rank0(str):
    if torch.distributed.get_rank() == 0:
        print(str)


def print_cached_dicts(cached_dict):
    for item in cached_dict:
        print(f"{item}: {cached_dict[item]}")

def get_num_item_per_mb(torch_data_type):
    if torch_data_type == torch.half:
        num_item_per_mb = 1024 * 1024 / 2
    elif torch_data_type == torch.float:
        num_item_per_mb = 1024 * 1024 / 4
    else:
        raise RuntimeError(f"data type {torch_data_type} not support.")
    return num_item_per_mb

def calculate_data_size(data_shape, torch_data_type):
    num_item_per_mb = get_num_item_per_mb(torch_data_type)
    data_size = 1
    for item in data_shape:
        data_size *= item
    return data_size * num_item_per_mb

def load_data_size_list(args, tp, usp, rsp, dp, model_size):
    data_size_list = []
    for batch_size in args.prof_mbs_list:
        if args.use_square_scope:
            seqlen = args.prof_basic_seqlen
            while seqlen < gpt_configs[model_size].max_seqlen:
                data_size_list.append(batch_size * (seqlen // (usp * rsp)) * gpt_configs[model_size].hidden_size // tp)
                seqlen *= 2
        else:
            for seqlen in range(args.prof_basic_seqlen, gpt_configs[model_size].max_seqlen + 1, args.prof_basic_seqlen):
                data_size_list.append(batch_size * (seqlen // (usp * rsp)) * gpt_configs[model_size].hidden_size // tp)
    return data_size_list

def all_to_all_single(args, data_size, torch_data_type, parallel_group):
    send_tensors = [torch.ones(
        data_size, dtype=torch_data_type
    ).cuda()] * 3
    stream = torch.cuda.Stream()
    for _ in range(args.prof_warmup_times):
        a2a_reqs = [None] * 3
        for i in range(4):
            if 0 <= i < 3:
                send_tensor = send_tensors[i]
                output_tensor = torch.empty_like(send_tensor)
                a2a_reqs[i] = dist.all_to_all_single(output_tensor, send_tensor, group=parallel_group, async_op=True)
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
                a2a_reqs[i] = dist.all_to_all_single(output_tensor, send_tensor, group=parallel_group, async_op=True)
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

def all_gather_single(args, data_size, world_size, torch_data_type, parallel_group):
    send_tensor = torch.ones(
        data_size, dtype=torch_data_type
    ).cuda()
    tensor_list = [
        torch.zeros(data_size, dtype=torch_data_type).cuda()
        for _ in range(world_size)
    ]
    for _ in range(args.prof_warmup_times):
        dist.all_gather(tensor_list, send_tensor, group=parallel_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times):
        dist.all_gather(tensor_list, send_tensor, group=parallel_group)
    end.record()
    torch.cuda.synchronize()
    send_tensor.cpu()
    for tensor in tensor_list:
        tensor.cpu()
    del send_tensor, tensor_list
    gc.collect()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / args.prof_repeat_times

def all_reduce_single(args, data_size, torch_data_type, parallel_group):
    send_tensor = torch.ones(
        data_size, dtype=torch_data_type
    ).cuda()
    for _ in range(args.prof_warmup_times):
        dist.all_reduce(send_tensor, group=parallel_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times):
        dist.all_reduce(send_tensor, group=parallel_group)
    end.record()
    torch.cuda.synchronize()
    send_tensor.cpu()
    del send_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / args.prof_repeat_times

def reduce_scatter_single(args, data_size, world_size, torch_data_type, parallel_group, parallel_size):
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
        input_list = list(send_tensor.chunk(parallel_size, 0))
        for idx, tensor in enumerate(input_list):
            if not tensor.is_contiguous():
                input_list[idx] = tensor.contiguous()
        new_input_ = torch.empty_like(input_list[0])
        dist.reduce_scatter(new_input_, input_list, group=parallel_group)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times):
        input_list = list(send_tensor.chunk(parallel_size, 0))
        for idx, tensor in enumerate(input_list):
            if not tensor.is_contiguous():
                input_list[idx] = tensor.contiguous()
        new_input_ = torch.empty_like(input_list[0])
        dist.reduce_scatter(new_input_, input_list, group=parallel_group)
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

def plot_profile_results(args, model, size, tp_size, usp_size, rsp_size, dp_size, avg_time_list, collective_type, parallel_type):
    x = list(avg_time_list.keys())
    y = list(avg_time_list.values())
    plt.plot(x, y, label=f"{model}_{size}_tp{tp_size}_usp{usp_size}_rsp{rsp_size}_dp{dp_size}_{collective_type}_{parallel_type}")
    coeff = polyfit(x, y, 1)
    y_fit = polyval(coeff, x)
    plt.plot(x, y_fit, 'g', label='Fit Curve')
    plt.xlabel("data size (MB)")
    plt.ylabel("time (ms)")
    plt.title(f"{model}_{size}_{collective_type}_{parallel_type}")
    plt.legend()
    plt.savefig(os.path.join(args.prof_fig_path, f"{model}_{size}_tp{tp_size}_usp{usp_size}_rsp{rsp_size}_dp{dp_size}_{collective_type}_{parallel_type}.png"))
    plt.close()
    # record coeff and mse
    mse = np.mean((y - y_fit) ** 2)
    print(coeff, mse)
    with open(os.path.join(args.prof_path, f'fit_curve_{model}_{size}_tp{tp_size}_usp{usp_size}_rsp{rsp_size}_dp{dp_size}_{collective_type}_{parallel_type}.txt'), 'w') as f:
        f.write('coeff: ' + str(coeff))
        f.write('mse: ' + str(mse))
    

def profile(rank, world_size, parallel_type, tp_size, usp_size, rsp_size, dp_size, data_size_list, model, size, torch_data_type):
    args = parse_args()
    parallel_size = 1
    parallel_group = None
    # We only use all first group of actual parallel groups to profile
    if tp_size > 1 and parallel_type == "tp":
        if rank in range(tp_size):
            tp_group = dist.new_group(list(range(tp_size)), use_local_synchronization=True)
            initialized = True
            parallel_group = tp_group
            parallel_size = tp_size
            print(f'rank {rank} tp_ranks: {dist.get_process_group_ranks(tp_group)}', flush=True)
    if usp_size > 1 and parallel_type == "usp":
        for i in range(tp_size):
            usp_group_start = i
            usp_group_end = usp_group_start + tp_size * usp_size
            if rank in range(usp_group_start, usp_group_end, tp_size):
                usp_group = dist.new_group(list(range(usp_group_start, usp_group_end, tp_size)), use_local_synchronization=True)
                parallel_group = usp_group
                parallel_size = usp_size
                print(f'rank {rank} usp_ranks: {dist.get_process_group_ranks(usp_group)}', flush=True)
                break
    cp_size = usp_size * rsp_size
    if dp_size > 1 and parallel_type == "dp":
        initialized = False
        for i in range(tp_size * cp_size):
            dp_group_start = i
            dp_group_end = world_size
            if rank in range(dp_group_start, dp_group_end, tp_size * cp_size):
                dp_group = dist.new_group(list(range(dp_group_start, dp_group_end, tp_size * cp_size)), use_local_synchronization=True)
                parallel_group = dp_group
                parallel_size = dp_size
                initialized = True
                print(f'rank {rank} dp_ranks: {dist.get_process_group_ranks(dp_group)}', flush=True)
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
    elif torch_data_type == torch.half or torch_data_type == torch.bfloat16:
        mb_per_item = 2 / (1024 * 1024)
    else:
        raise RuntimeError(f"type {torch_data_type} not support.")

    collectives = []
    if tp_size > 1 and parallel_type == "tp":
        collectives.extend(["all_gather", "reduce_scatter", "all_reduce"])
    if usp_size > 1 and parallel_type == "usp":
        collectives.extend(["all_to_all"])
    # if rsp_size > 1:
    if dp_size > 1 and parallel_type == "dp":
        collectives.extend(["all_reduce", "reduce_scatter"])
    
    for collective_type in collectives:
        avg_time_list = {}
        print(
            f"{dist.get_rank()} Start profiling {parallel_type} {collective_type}... len(data_size_list) = {len(data_size_list)}", flush=True
        )
        for idx, data_size in enumerate(data_size_list):
            data_size_in_mb = int(data_size * mb_per_item)

            if data_size_in_mb not in avg_time_list:
                print(
                    f"[rank {rank}] {model}_{size} profiling {parallel_type} {collective_type} tp{tp_size} usp{usp_size} rsp{rsp_size} dp{dp_size} ({world_size}GPUs) ({data_size} = {data_size_in_mb} MB) Progress: {idx / len(data_size_list)}\n", flush=True
                )
                # report_memory(f"{collective_type} {data_size_in_mb}")
                hash_name = f"{parallel_type}_{collective_type}_tp{tp_size}_up{usp_size}_rsp{rsp_size}_dp{dp_size}_{data_size_in_mb}_{torch_data_type}"
                if hash_name in profiled_results:
                    print_rank0(f"hit in cache!")
                    avg_time_list[data_size_in_mb] = profiled_results[hash_name]
                else:
                    dist.barrier()

                    try:
                        if collective_type == "all_reduce":
                            avg_time_list[data_size_in_mb] = all_reduce_single(args, data_size, torch_data_type, parallel_group)
                        elif collective_type == "reduce_scatter":
                            avg_time_list[data_size_in_mb] = reduce_scatter_single(args, data_size, world_size, torch_data_type, parallel_group, parallel_size)
                        elif collective_type == "all_gather":
                            avg_time_list[data_size_in_mb] = all_gather_single(args, data_size, world_size, torch_data_type, parallel_group)
                        elif collective_type == "all_to_all":
                            avg_time_list[data_size_in_mb] = all_to_all_single(args, data_size, torch_data_type, parallel_group)
                        else:
                            raise RuntimeError(f"collective {collective_type} not support.")
                        gc.collect()
                        torch.cuda.empty_cache()
                    except RuntimeError as e:
                        print(e)

                    assert data_size_in_mb in avg_time_list and avg_time_list[data_size_in_mb] is not None, f"rank {rank} {collective_type} {data_size_in_mb} profiling failed."
                    profiled_results[hash_name] = avg_time_list[data_size_in_mb]
        avg_time_list = dict(sorted(avg_time_list.items()))
        if rank == 0:
            for data_size_in_mb in avg_time_list:
                print(
                    f"[{collective_type}] {data_size_in_mb} MB: {avg_time_list[data_size_in_mb]:.2f} ms", flush=True
                )
            result_title = ["data_size(MB)", "time(ms)"]
            save_file_name = (
                f"prim_{model}_{size}_tp{tp_size}_usp{usp_size}_rsp{rsp_size}_dp{dp_size}_{collective_type}_{parallel_type}.csv"
            )
            f_result = open(args.prof_path + save_file_name, "w")
            f_csv = csv.writer(f_result)
            f_csv.writerow(result_title)
            for data_size_in_mb in avg_time_list:
                tmp_row = [0, 0]
                tmp_row[0] = "{:.0f}".format(data_size_in_mb)
                tmp_row[1] = "{:.3f}".format(float(avg_time_list[data_size_in_mb]))
                f_csv.writerow(tmp_row)
            f_result.close()
            plot_profile_results(args, model, size, tp_size, usp_size, rsp_size, dp_size, avg_time_list, collective_type, parallel_type)

    if rank == 0:
        save_dict = {}
        save_dict["profiled_results"] = profiled_results
        pickle.dump(save_dict, open(args.prof_cache_file, "wb"))


def run_profile(args, task):
    model = task["model"]
    size = task["size"]
    world_size = args.max_num_gpus

    data_type = model_prof_configs[model]["dtype"]
    tp_size = args.prof_tp_size
    usp_size = args.prof_usp_size
    rsp_size = args.prof_rsp_size
    dp_size = args.prof_dp_size
    data_size_list = load_data_size_list(args, tp_size, usp_size, rsp_size, dp_size, size)
    print(f"tp_size: {tp_size}, usp_size: {usp_size}, dp_size: {dp_size}")
    # Here we profile dp only, which will cross-nodes
    
    device = args.rank % torch.cuda.device_count()
    visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")
    print(f'device{device}, visible_devices: {visible_devices}')
    torch.cuda.set_device(int(visible_devices[device]))
    torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=args.rank)
    
    if dp_size > 1:
        profile(args.rank, world_size, "dp", tp_size, usp_size, rsp_size, dp_size, data_size_list, model, size, data_type)


if __name__ == "__main__":

    start_profiling_time = time.time()
    args = parse_args()

    ## get profiling tasks
    ## "task"s are defined by unique {model, size} pairs
    all_prof_tasks = []
    model_names = (["gpt"])
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
        run_profile(args, prof_task)

    end_profiling_time = time.time()
    print(f"[TOTAL PROFILING TIME] {end_profiling_time - start_profiling_time:2f} s")
