import os
import re
import csv
import pandas

output_path = "../results/parser"
if not os.path.exists(output_path):
    os.makedirs(output_path)

profile_gpt_path="../results/profiled-gpt-hetaceso"
profile_comm_path="../results/profiled-local-comm-hetaceso"
profile_p2p_path="../results/profiled-local-p2p-hetaceso"
estimate_result_path = "../results/estimate"
real_time_result_path = "../runtime/tests/unit_tests/flexpipe/profile"

excel_output_path = os.path.join(output_path, 'res.xlsx')

model_sizes = ['350M', '1_3B', '2_6B', '6_7B', '13B']
tests_num = [8, 8, 8, 8, 8]

config_parser = re.compile(r'gpu_configs/(?P<model_size>[\d\_\w]+)/gpt_mbs(?P<mbs>\d+)_tp(?P<tp>\d+)_usp(?P<usp>\d+)_rsp(?P<rsp>\d+)_dp(?P<dp>\d+).json')

estimate_result_total_time_parser = re.compile(r'total_time: (?P<total_time>\d.*\d)')
estimate_result_fwd_time_parser = re.compile(r'fwd_time: (?P<fwd_time>\d.*\d)')
estimate_result_bwd_time_parser = re.compile(r'bwd_time: (?P<bwd_time>\d.*\d)')
estimate_result_memory_sum_parser = re.compile(r'memory_sum: (?P<memory_sum>\d.*\d)')
estimate_result_ref_total_memory_parser = re.compile(r'ref_total_memory: (?P<ref_total_memory>\d.*\d)')

real_time_time_parser = re.compile(r'rank  0: (?P<time>\d.*\d)')
real_time_memory_parser = re.compile(r'memory (MB) \| allocated: (?P<allocated>\d.*\d) \| max allocated: (?P<max_allocated>\d.*\d) \| reserved: (?P<reserved>\d.*\d) \| max reserved: (?P<max_reserved>\d.*\d)')

def estimate_result_parser(file):
    res = {
        "config": {
            "model_size": "",
            "mbs": 0,
            "tp": 0,
            "usp": 0,
            "rsp": 0,
            "dp": 0,
        },
        "data": {
            "total_time": 0.0,
            "fwd_time": 0.0,
            "bwd_time": 0.0,
            "memory_sum": 0.0,
            "ref_total_memory": 0.0,
        }
    }
    with open(file, 'r') as fp:
        for line in fp.readlines():
            config_re = config_parser.search(line)
            total_time_re = estimate_result_total_time_parser.search(line)
            fwd_time_re = estimate_result_fwd_time_parser.search(line)
            bwd_time_re = estimate_result_bwd_time_parser.search(line)
            memory_sum_re = estimate_result_memory_sum_parser.search(line)
            ref_total_memory_re = estimate_result_ref_total_memory_parser.search(line)
            if config_re:
                res["config"] = {
                    "model_size": config_parser.group('model_size'),
                    "mbs": int(config_parser.group('mbs')),
                    "tp": int(config_parser.group('tp')),
                    "usp": int(config_parser.group('usp')),
                    "rsp": int(config_parser.group('rsp')),
                    "dp": int(config_parser.group('dp')),
                }
                
            if total_time_re:
                res["data"]["total_time"] = float(total_time_re.group('total_time'))
            if fwd_time_re:
                res["data"]["fwd_time"] = float(fwd_time_re.group('fwd_time'))
            if bwd_time_re:
                res["data"]["bwd_time"] = float(bwd_time_re.group('bwd_time'))
            if memory_sum_re:
                res["data"]["memory_sum"] = float(memory_sum_re.group('memory_sum'))
            if ref_total_memory_re:
                res["data"]["ref_total_memory"] = float(ref_total_memory_re.group('ref_total_memory'))
    return res

def estimate_results_parse_all(model_size_max):
    assert model_size_max < len(model_sizes), f'model size max {model_size_max} larger than model sizes len {len(model_sizes)}'
    results = {}
    for i in range(model_size_max + 1):
        results[model_sizes[i]] = []
        for test_i in range(tests_num[i]):
            results[model_sizes[i]].append(estimate_result_parser(os.path.join(estimate_result_path, model_sizes[i], f'test_{test_i}.txt')))
    return results

def realtime_result_times_parser(file):
    res = {
        "data": {
            "forward-backward": 0.0,
            "forward-compute": 0.0,
            "backward-compute": 0.0,
        }
    }
    with open(file, 'r') as fp:
        lines = fp.readlines()
        fwd_bwd_time_re = real_time_time_parser.search(lines[2])
        res["data"]["forward-backward"] = float(fwd_bwd_time_re.group('time'))
        fwd_time_re = real_time_time_parser.search(lines[7])
        res["data"]["forward-compute"] = float(fwd_time_re.group('time'))
        bwd_time_re = real_time_time_parser.search(lines[17])
        res["data"]["backward-compute"] = float(bwd_time_re.group('time'))
    return res

def realtime_result_memory_parser(file):
    res = {
        "data": {
            "memory-max-allocated": 0.0,
            "memory-max-reserved": 0.0
        }
    }
    with open(file, 'r') as fp:
        lines = fp.readlines()
        memory_re = real_time_memory_parser.search(lines[0])
        res["data"] = {
            "memory-max-allocated": float(memory_re.group('max_allocated')),
            "memory-max-reserved": float(memory_re.group('max_reserved')),
        }
    return res

def realtime_results_parse_all(model_size_max):
    assert model_size_max < len(model_sizes), f'model size max {model_size_max} larger than model sizes len {len(model_sizes)}'
    results = {
        "time": {},
        "mem": {}
    }
    for i in range(model_size_max + 1):
        results[model_sizes[i]] = []
        for test_i in range(tests_num[i]):
            time_res = realtime_result_times_parser(os.path.join(real_time_result_path, model_sizes[i], f'logs_{test_i}', 'times_iter5.log'))
            mem_res = realtime_result_memory_parser(os.path.join(real_time_result_path, model_sizes[i], f'logs_{test_i}', 'times_iter5.log'))
            results["time"][model_sizes[i]].append(time_res)
            results["mem"][model_sizes[i]].append(mem_res)
    return results


    