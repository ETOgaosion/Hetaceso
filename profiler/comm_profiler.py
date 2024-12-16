# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as multiproc
import time
import csv
import pickle
import argparse
from model_configs import model_prof_configs


def parse_args():
    parser = argparse.ArgumentParser(
        description="communication-profiler arguments", allow_abbrev=False
    )

    parser.add_argument(
        "--prof-tp-size", type=int, default=None, help="Profiler tp size."
    )
    parser.add_argument("--prof-path", type=str, default=None, help="")
    parser.add_argument("--prof-cache-file", type=str, default=None, help="")
    parser.add_argument("--prof-model-name", type=str, default="all", help="")
    parser.add_argument("--prof-model-size", type=str, default="all", help="")
    parser.add_argument("--prof-warmup-times", type=int, default=0, help="")
    parser.add_argument("--prof-repeat-times", type=int, default=1, help="")
    parser.add_argument("--prof-skip-running", action="store_true", help="")
    parser.add_argument("--prof-op-time-path", type=str, default=None, help="")
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


def run(rank, world_size, data_size_list, model, size, torch_data_type):
    args = parse_args()
    init_method = "tcp://"
    master_ip = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "6000")
    init_method += master_ip + ":" + master_port
    dist.init_process_group(
        backend="nccl", world_size=world_size, rank=rank, init_method=init_method
    )

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
        collectives = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]
    else:
        raise RuntimeError(f"Model {model} is not supported.")

    for collective_type in collectives:
        avg_time_list = {}
        print_rank0(
            f"Start profiling {collective_type}... len(data_size_list) = {len(data_size_list)}"
        )
        for data_size in data_size_list:
            data_size_in_mb = int(data_size * mb_per_item)
            if collective_type in ["all_gather", "all_to_all"]:
                full_data_size_in_mb = data_size_in_mb * world_size
            elif collective_type in ["all_reduce", "reduce_scatter"]:
                full_data_size_in_mb = data_size_in_mb

            if data_size_in_mb not in avg_time_list:
                print(
                    f"[rank {rank}] {model}_{size} profiling {collective_type} ({world_size}GPUs) ({data_size} = {data_size_in_mb} MB)...\n"
                )
                hash_name = f"{collective_type}_{world_size}gpus_{data_size_in_mb}_{torch_data_type}"
                if hash_name in profiled_results:
                    print_rank0(f"hit in cache!")
                    avg_time_list[data_size_in_mb] = profiled_results[hash_name]
                elif full_data_size_in_mb > args.max_data_size:
                    avg_time_list[data_size_in_mb] = 1000000000
                else:
                    time_list = []

                    try:
                        if not args.prof_skip_running:
                            if collective_type == "all_gather":
                                send_tensor = torch.ones(
                                    data_size, dtype=torch_data_type
                                ).cuda()
                                tensor_list = [
                                    torch.zeros(data_size, dtype=torch_data_type).cuda()
                                    for _ in range(world_size)
                                ]
                                for i in range(args.prof_warmup_times):
                                    dist.all_gather(tensor_list, send_tensor)
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                for i in range(args.prof_repeat_times):
                                    dist.all_gather(tensor_list, send_tensor)
                                end.record()
                                torch.cuda.synchronize()
                                time_list.append(start.elapsed_time(end) / args.prof_repeat_times)
                            elif collective_type == "all_reduce":
                                send_tensor = torch.ones(
                                    data_size, dtype=torch_data_type
                                ).cuda()
                                for i in range(args.prof_warmup_times):
                                    dist.all_reduce(send_tensor)
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                for i in range(args.prof_repeat_times):
                                    dist.all_reduce(send_tensor)
                                end.record()
                                torch.cuda.synchronize()
                                time_list.append(start.elapsed_time(end) / args.prof_repeat_times)
                            elif collective_type == "reduce_scatter":
                                if data_size % world_size == 0:
                                    send_tensor = torch.ones(
                                        data_size, dtype=torch_data_type
                                    ).cuda()
                                else:
                                    _data_size = (data_size // world_size) * world_size
                                    send_tensor = torch.ones(
                                        _data_size, dtype=torch_data_type
                                    ).cuda()
                                for i in range(args.prof_warmup_times):
                                    input_list = list(send_tensor.chunk(world_size, 0))
                                    for idx, tensor in enumerate(input_list):
                                        if not tensor.is_contiguous():
                                            input_list[idx] = tensor.contiguous()
                                    new_input_ = torch.empty_like(input_list[0])
                                    dist.reduce_scatter(new_input_, input_list)
                                start = torch.cuda.Event(enable_timing=True)
                                end = torch.cuda.Event(enable_timing=True)
                                start.record()
                                for i in range(args.prof_repeat_times):
                                    input_list = list(send_tensor.chunk(world_size, 0))
                                    for idx, tensor in enumerate(input_list):
                                        if not tensor.is_contiguous():
                                            input_list[idx] = tensor.contiguous()
                                    new_input_ = torch.empty_like(input_list[0])
                                    dist.reduce_scatter(new_input_, input_list)
                                end.record()
                                torch.cuda.synchronize()
                                time_list.append(start.elapsed_time(end) / args.prof_repeat_times)
                            elif collective_type == "all_to_all":
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
                                            a2a_reqs[i] = dist.all_to_all_single(output_tensor, send_tensor, group=dist.group.WORLD, async_op=True)
                                        if i > 0:
                                            with torch.cuda.stream(stream):
                                                a2a_reqs[i - 1].wait()
                                torch.cuda.current_stream().wait_stream(stream)
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
                                            a2a_reqs[i] = dist.all_to_all_single(output_tensor, send_tensor, group=dist.group.WORLD, async_op=True)
                                        if i > 0:
                                            with torch.cuda.stream(stream):
                                                a2a_reqs[i - 1].wait()
                                torch.cuda.current_stream().wait_stream(stream)
                                end.record()
                                torch.cuda.synchronize()
                                time_list.append(start.elapsed_time(end) / args.prof_repeat_times)
                    except RuntimeError as e:
                        print(e)
                        time_list = [1000000 for _ in range(args.prof_repeat_times)]

                    avg_time_list[data_size_in_mb] = (
                        sum(time_list) / args.prof_repeat_times
                    )
                    profiled_results[hash_name] = (
                        sum(time_list) / args.prof_repeat_times
                    )
        if rank == 0:
            for data_size_in_mb in avg_time_list:
                print(
                    f"[{collective_type}] {data_size_in_mb} MB: {avg_time_list[data_size_in_mb]:.2f} ms"
                )
            result_title = ["data_size(MB)", "time(ms)"]
            save_file_name = (
                f"prim_{model}_{size}_{collective_type}_{world_size}gpus.csv"
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
    tp_size_list = [1]

    if data_type == "fp16":
        torch_data_type = torch.half
        num_item_per_mb = 1024 * 1024 / 2
    elif data_type == "fp32":
        torch_data_type = torch.float
        num_item_per_mb = 1024 * 1024 / 4
    else:
        raise RuntimeError(f"data type {data_type} not support.")
    
    print(f'mbs_list: {mbs_list}, seqlen_list: {seqlen_list}, tp_size_list: {tp_size_list}')

    data_size_list = []
    for mbs in mbs_list:
        for seq_len in seqlen_list:
            for tp in tp_size_list:
                file_name = (
                    args.prof_op_time_path
                    + f"{model}_{size}_mbs{mbs}_seqlen{seq_len}_tp{tp}.csv"
                )
                print(file_name)
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

    torch.multiprocessing.spawn(
        run,
        args=(args.prof_tp_size, data_size_list, model, size, torch_data_type),
        nprocs=args.prof_tp_size,
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
