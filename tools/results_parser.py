import os
import re
import csv
import pandas

parser_config_path = "results_parser.cfg"

parser_config_max_model_size_parser = re.compile(r'model_size_max = (?P<max_model_size>[\d\_\w]+)')

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

config_parser = re.compile(r'gpu_configs/(?P<model_size>[\d\_\w]+)/gpt_mbs(?P<mbs>\d+)_tp(?P<tp>\d+)_usp(?P<usp>\d+)_rsp(?P<rsp>\d+)_dp(?P<dp>\d+)')

estimate_result_total_time_parser = re.compile(r'total_time: (?P<total_time>\d.*\d)')
estimate_result_fwd_time_parser = re.compile(r' fwd_time: (?P<fwd_time>\d.*\d)')
estimate_result_bwd_time_parser = re.compile(r' bwd_time: (?P<bwd_time>\d.*\d)')
estimate_result_embed_comp_fwd_time_parser = re.compile(r'embed_comp_fwd_time: (?P<embed_comp_fwd_time>\d.*\d)')
estimate_result_embed_comp_bwd_time_parser = re.compile(r'embed_comp_bwd_time: (?P<embed_comp_bwd_time>\d.*\d)')
estimate_result_attn_comp_fwd_time_parser = re.compile(r'attn_comp_fwd_time: (?P<attn_comp_fwd_time>\d.*\d)')
estimate_result_attn_comp_bwd_time_parser = re.compile(r'attn_comp_bwd_time: (?P<attn_comp_bwd_time>\d.*\d)')
estimate_result_mlp_comp_fwd_time_parser = re.compile(r'mlp_comp_fwd_time: (?P<mlp_comp_fwd_time>\d.*\d)')
estimate_result_mlp_comp_bwd_time_parser = re.compile(r'mlp_comp_bwd_time: (?P<mlp_comp_bwd_time>\d.*\d)')
estimate_result_post_comp_fwd_time_parser = re.compile(r'post_comp_fwd_time: (?P<post_comp_fwd_time>\d.*\d)')
estimate_result_post_comp_bwd_time_parser = re.compile(r'post_comp_bwd_time: (?P<post_comp_bwd_time>\d.*\d)')
estimate_result_memory_sum_parser = re.compile(r'memory_sum: (?P<memory_sum>\d.*\d)')
estimate_result_ref_total_memory_parser = re.compile(r'ref_total_memory: (?P<ref_total_memory>\d.*\d)')

real_time_forward_backward_time_parser = re.compile(r'forward-backward:')
real_time_forward_compute_time_parser = re.compile(r'forward-compute:')
real_time_backward_compute_time_parser = re.compile(r'backward-compute:')
real_time_dec_embedding_forward_time_parser = re.compile(r'dec-embedding-forward:')
real_time_dec_self_attention_forward_time_parser = re.compile(r'dec-self-attention-forward:')
real_time_dec_mlp_forward_time_parser = re.compile(r'dec-mlp-forward:')
real_time_dec_post_process_forward_time_parser = re.compile(r'dec-post-process-forward:')
real_time_rank0_time_parser = re.compile(r'rank  0: (?P<time>\d.*\d)')
real_time_memory_parser = re.compile(r'memory \(MB\) \| allocated: (?P<allocated>\d.*\d) \| max allocated: (?P<max_allocated>\d.*\d) \| reserved: (?P<reserved>\d.*\d) \| max reserved: (?P<max_reserved>\d.*\d)')

def estimate_result_parser(file):
    print(f'estimate_result_parser handling {file}')
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
            "embed_comp_fwd_time": 0.0,
            "embed_comp_bwd_time": 0.0,
            "attn_comp_fwd_time": 0.0,
            "attn_comp_bwd_time": 0.0,
            "mlp_comp_fwd_time": 0.0,
            "mlp_comp_bwd_time": 0.0,
            "post_comp_fwd_time": 0.0,
            "post_comp_bwd_time": 0.0,
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
            embed_fwd_comp_time_re = estimate_result_embed_comp_fwd_time_parser.search(line)
            embed_bwd_comp_time_re = estimate_result_embed_comp_bwd_time_parser.search(line)
            attn_fwd_comp_time_re = estimate_result_attn_comp_fwd_time_parser.search(line)
            attn_bwd_comp_time_re = estimate_result_attn_comp_bwd_time_parser.search(line)
            mlp_fwd_comp_time_re = estimate_result_mlp_comp_fwd_time_parser.search(line)
            mlp_bwd_comp_time_re = estimate_result_mlp_comp_bwd_time_parser.search(line)
            post_fwd_comp_time_re = estimate_result_post_comp_fwd_time_parser.search(line)
            post_bwd_comp_time_re = estimate_result_post_comp_bwd_time_parser.search(line)
            memory_sum_re = estimate_result_memory_sum_parser.search(line)
            ref_total_memory_re = estimate_result_ref_total_memory_parser.search(line)
            if config_re:
                res["config"] = {
                    "model_size": config_re.group('model_size'),
                    "mbs": int(config_re.group('mbs')),
                    "tp": int(config_re.group('tp')),
                    "usp": int(config_re.group('usp')),
                    "rsp": int(config_re.group('rsp')),
                    "dp": int(config_re.group('dp')),
                }
                
            if total_time_re:
                res["data"]["total_time"] = float(total_time_re.group('total_time'))
            if fwd_time_re:
                res["data"]["fwd_time"] = float(fwd_time_re.group('fwd_time'))
            if bwd_time_re:
                res["data"]["bwd_time"] = float(bwd_time_re.group('bwd_time'))
            if embed_fwd_comp_time_re:
                res["data"]["embed_comp_fwd_time"] = float(embed_fwd_comp_time_re.group('embed_comp_fwd_time'))
            if embed_bwd_comp_time_re:
                res["data"]["embed_comp_bwd_time"] = float(embed_bwd_comp_time_re.group('embed_comp_bwd_time'))
            if attn_fwd_comp_time_re:
                res["data"]["attn_comp_fwd_time"] = float(attn_fwd_comp_time_re.group('attn_comp_fwd_time'))
            if attn_bwd_comp_time_re:
                res["data"]["attn_comp_bwd_time"] = float(attn_bwd_comp_time_re.group('attn_comp_bwd_time'))
            if mlp_fwd_comp_time_re:
                res["data"]["mlp_comp_fwd_time"] = float(mlp_fwd_comp_time_re.group('mlp_comp_fwd_time'))
            if mlp_bwd_comp_time_re:
                res["data"]["mlp_comp_bwd_time"] = float(mlp_bwd_comp_time_re.group('mlp_comp_bwd_time'))
            if post_fwd_comp_time_re:
                res["data"]["post_comp_fwd_time"] = float(post_fwd_comp_time_re.group('post_comp_fwd_time'))
            if post_bwd_comp_time_re:
                res["data"]["post_comp_bwd_time"] = float(post_bwd_comp_time_re.group('post_comp_bwd_time'))
            if memory_sum_re:
                res["data"]["memory_sum"] = float(memory_sum_re.group('memory_sum'))
            if ref_total_memory_re:
                res["data"]["ref_total_memory"] = float(ref_total_memory_re.group('ref_total_memory'))
    return res

def estimate_results_parse_all(model_size_max):
    assert model_size_max in model_sizes, f'model size max {model_size_max} not in model sizes {model_sizes}'
    results = {}
    for i, model_size in enumerate(model_sizes):
        results[model_size] = []
        for test_i in range(tests_num[i]):
            results[model_size].append(estimate_result_parser(os.path.join(estimate_result_path, model_size, f'test_{test_i}.txt')))
        if model_size == model_size_max:
            break
    return results

def realtime_result_times_parser(file):
    print(f'realtime_result_times_parser handling {file}')
    res = {
        "data": {
            "forward-backward": 0.0,
            "forward-compute": 0.0,
            "backward-compute": 0.0,
            "dec-embedding-forward": 0.0,
            "dec-self-attention-forward": 0.0,
            "dec-mlp-forward": 0.0,
            "dec-post-process-forward": 0.0,
        }
    }
    with open(file, 'r') as fp:
        lines = fp.readlines()
        for idx, line in enumerate(lines):
            fwd_bwd_time_re = real_time_forward_backward_time_parser.search(line)
            if fwd_bwd_time_re:
                rank0_time_re = real_time_rank0_time_parser.search(lines[idx + 1])
                res["data"]["forward-backward"] = float(rank0_time_re.group('time'))
            fwd_time_re = real_time_forward_compute_time_parser.search(line)
            if fwd_time_re:
                rank0_time_re = real_time_rank0_time_parser.search(lines[idx + 1])
                res["data"]["forward-compute"] = float(rank0_time_re.group('time'))
            bwd_time_re = real_time_backward_compute_time_parser.search(line)
            if bwd_time_re:
                rank0_time_re = real_time_rank0_time_parser.search(lines[idx + 1])
                res["data"]["backward-compute"] = float(rank0_time_re.group('time'))
            dec_embed_fwd_time_re = real_time_dec_embedding_forward_time_parser.search(line)
            if dec_embed_fwd_time_re:
                rank0_time_re = real_time_rank0_time_parser.search(lines[idx + 1])
                res["data"]["dec-embedding-forward"] = float(rank0_time_re.group('time'))
            dec_self_attn_fwd_time_re = real_time_dec_self_attention_forward_time_parser.search(line)
            if dec_self_attn_fwd_time_re:
                rank0_time_re = real_time_rank0_time_parser.search(lines[idx + 1])
                res["data"]["dec-self-attention-forward"] = float(rank0_time_re.group('time'))
            dec_mlp_fwd_time_re = real_time_dec_mlp_forward_time_parser.search(line)
            if dec_mlp_fwd_time_re:
                rank0_time_re = real_time_rank0_time_parser.search(lines[idx + 1])
                res["data"]["dec-mlp-forward"] = float(rank0_time_re.group('time'))
            dec_post_fwd_time_re = real_time_dec_post_process_forward_time_parser.search(line)
            if dec_post_fwd_time_re:
                rank0_time_re = real_time_rank0_time_parser.search(lines[idx + 1])
                res["data"]["dec-post-process-forward"] = float(rank0_time_re.group('time'))
        assert res["data"]["forward-backward"] != 0.0, f'forward-backward is 0.0'
        assert res["data"]["forward-compute"] != 0.0, f'forward-compute is 0.0'
        assert res["data"]["backward-compute"] != 0.0, f'backward-compute is 0.0'
        assert res["data"]["dec-embedding-forward"] != 0.0, f'dec-embedding-forward is 0.0'
        assert res["data"]["dec-self-attention-forward"] != 0.0, f'dec-self-attention-forward is 0.0'
        assert res["data"]["dec-mlp-forward"] != 0.0, f'dec-mlp-forward is 0.0'
        assert res["data"]["dec-post-process-forward"] != 0.0, f'dec-post-process-forward is 0.0'
    return res

def realtime_result_memory_parser(file):
    print(f'realtime_result_memory_parser handling {file}')
    res = {
        "data": {
            "memory-max-allocated": 0.0,
            "memory-max-reserved": 0.0
        }
    }
    with open(file, 'r') as fp:
        lines = fp.readlines()
        memory_re = real_time_memory_parser.search(lines[0])
        if memory_re:
            res["data"] = {
                "memory-max-allocated": float(memory_re.group('max_allocated')),
                "memory-max-reserved": float(memory_re.group('max_reserved')),
            }
    assert res["data"]["memory-max-allocated"] != 0.0, f'memory-max-allocated is 0.0'
    assert res["data"]["memory-max-reserved"] != 0.0, f'memory-max-reserved is 0.0'
    return res

def realtime_results_parse_all(model_size_max):
    assert model_size_max in model_sizes, f'model size max {model_size_max} not in model sizes {model_sizes}'
    results = {
        "time": {},
        "mem": {}
    }
    for i, model_size in enumerate(model_sizes):
        results["time"][model_size] = []
        results["mem"][model_size] = []
        for test_i in range(tests_num[i]):
            time_res = realtime_result_times_parser(os.path.join(real_time_result_path, model_size, f'logs_{test_i}', 'times_iter3.log'))
            mem_res = realtime_result_memory_parser(os.path.join(real_time_result_path, model_size, f'logs_{test_i}', 'memory_iter3_rank0.log'))
            results["time"][model_size].append(time_res)
            results["mem"][model_size].append(mem_res)
        if model_size == model_size_max:
            break
    return results

def write_to_csv(data, file):
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

first_write = True

def write_to_excel_sheet(data, sheet_name, excel_file):
    global first_write
    df = pandas.DataFrame(data)
    if first_write:
        writer = pandas.ExcelWriter(excel_file, engine="openpyxl")
        first_write = False
    else:
        writer = pandas.ExcelWriter(excel_file, engine="openpyxl", mode='a')
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.close()

def res_to_datas(res_estimate, res_realtime, max_model_size):
    datas = {}
    for model_size in model_sizes:
        datas[model_size] = [["config", "type", "total_time", "fwd_time", "bwd_time", "embed_fwd_time", "attn_fwd_time", "mlp_fwd_time", "post_fwd_time", "memory_sum", "ref_total_memory"]]
        for i in range(tests_num[model_sizes.index(model_size)]):
            datas[model_size].append([
                f'mbs{res_estimate[model_size][i]["config"]["mbs"]}_tp{res_estimate[model_size][i]["config"]["tp"]}_usp{res_estimate[model_size][i]["config"]["usp"]}_rsp{res_estimate[model_size][i]["config"]["rsp"]}_dp{res_estimate[model_size][i]["config"]["dp"]}',
                'modeling',
                res_estimate[model_size][i]["data"]["total_time"],
                res_estimate[model_size][i]["data"]["fwd_time"],
                res_estimate[model_size][i]["data"]["bwd_time"],
                res_estimate[model_size][i]["data"]["embed_comp_fwd_time"],
                res_estimate[model_size][i]["data"]["attn_comp_fwd_time"],
                res_estimate[model_size][i]["data"]["mlp_comp_fwd_time"],
                res_estimate[model_size][i]["data"]["post_comp_fwd_time"],
                res_estimate[model_size][i]["data"]["memory_sum"],
                res_estimate[model_size][i]["data"]["ref_total_memory"],
            ])
            datas[model_size].append([
                f'mbs{res_estimate[model_size][i]["config"]["mbs"]}_tp{res_estimate[model_size][i]["config"]["tp"]}_usp{res_estimate[model_size][i]["config"]["usp"]}_rsp{res_estimate[model_size][i]["config"]["rsp"]}_dp{res_estimate[model_size][i]["config"]["dp"]}',
                'realtime',
                res_realtime["time"][model_size][i]["data"]["forward-backward"],
                res_realtime["time"][model_size][i]["data"]["forward-compute"],
                res_realtime["time"][model_size][i]["data"]["backward-compute"],
                res_realtime["time"][model_size][i]["data"]["dec-embedding-forward"],
                res_realtime["time"][model_size][i]["data"]["dec-self-attention-forward"],
                res_realtime["time"][model_size][i]["data"]["dec-mlp-forward"],
                res_realtime["time"][model_size][i]["data"]["dec-post-process-forward"],
                res_realtime["mem"][model_size][i]["data"]["memory-max-reserved"],
                res_realtime["mem"][model_size][i]["data"]["memory-max-reserved"],
            ])
        if model_size == max_model_size:
            break
    return datas

def main():
    with open(parser_config_path, 'r') as fp:
        for line in fp.readlines():
            max_model_size_re = parser_config_max_model_size_parser.search(line)
            if max_model_size_re:
                max_model_size = max_model_size_re.group('max_model_size')
    estimate_res = estimate_results_parse_all(max_model_size)
    realtime_res = realtime_results_parse_all(max_model_size)
    datas = res_to_datas(estimate_res, realtime_res, max_model_size)
    for model_size in model_sizes:
        write_to_csv(datas[model_size], os.path.join(output_path, f'{model_size}.csv'))
        write_to_excel_sheet(datas[model_size], model_size, excel_output_path)
        if model_size == max_model_size:
            break

main()