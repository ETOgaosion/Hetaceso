import torch
import os
import matplotlib.pyplot as plt
import argparse
import gc
import dataclasses
import pickle
import csv

from numpy import polyfit, polyval
import numpy as np

from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward as flash_attn_varlen_fwd,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward as flash_attn_varlen_bwd,
)
from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd

from transformer_engine.pytorch import get_cu_seqlens_and_indices
from transformer_engine.pytorch import get_swa_mask, get_alibi, get_cu_seqlens, get_cu_seqlens_and_indices, get_indices, get_qkv_layout

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

# model_size: (num_layers, max_seqlen, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype)
gpt_configs = {
    "350M": GPTConfig(24, 32768, 1024, 4096, 16, 64, 51200, torch.float16),
    "1_3B": GPTConfig(24, 32768, 2048, 8192, 32, 64, 51200, torch.float16),
    "2_6B": GPTConfig(32, 16384, 2560, 10240, 32, 80, 51200, torch.float16),
    "6_7B": GPTConfig(32, 8192, 4096, 16384, 32, 128, 51200, torch.float16),
    "13B": GPTConfig(40, 8192, 5120, 20480, 40, 128, 51200, torch.float16),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Flash Attention Profiler")
    parser.add_argument("--model-size", type=str, default="350M")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--basic-seqlen", type=int, default=1024)
    parser.add_argument("--max-seqlen", type=int, default=32768)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--warm-up-times", type=int, default=10)
    parser.add_argument("--profile-times", type=int, default=100)
    parser.add_argument("--use-square_scope", action="store_true")
    parser.add_argument("--use-causal", action="store_true")
    parser.add_argument('--output-dir', type=str, default='../results/profiled-flash-attn-hetaceso')
    parser.add_argument('--output-fig-dir', type=str, default='../results/profiled-flash-attn-hetaceso/figs')
    parser.add_argument('--cache-file', type=str, default='../results/profiled-flash-attn-hetaceso/cache.pkl')
    return parser.parse_args()

args = parse_args()
args.max_seqlen = gpt_configs[args.model_size].max_seqlen
args.num_heads = gpt_configs[args.model_size].num_attention_heads
args.head_size = gpt_configs[args.model_size].hidden_size // gpt_configs[args.model_size].num_attention_heads

def generate_inputs(input_shape):
    b, s, h, d = input_shape
    q, k, v = torch.randn([b, s, h, d], dtype=torch.bfloat16, device='cuda:0'), torch.randn([b, s, h, d], dtype=torch.bfloat16, device='cuda:0'), torch.randn([b, s, h, d], dtype=torch.bfloat16, device='cuda:0')
    return q, k, v

def profile_flash_attn_single(inputs, causal=False):
    q, k, v = inputs
    softmax_scale = q.shape[-1] ** (-0.5)
    window_size  = (-1, 0) if causal else (-1, -1)
    for _ in range(args.warm_up_times):
        flash_attn_func(q, k, v, dropout_p=0.1, softmax_scale=softmax_scale, causal=causal, window_size=window_size)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.profile_times):
        flash_attn_func(q, k, v, dropout_p=0.1, softmax_scale=softmax_scale, causal=causal, window_size=window_size)
    end.record()
    torch.cuda.synchronize()
    # return in ms
    q, k, v = q.cpu(), k.cpu(), v.cpu()
    del q, k, v
    gc.collect()
    torch.cuda.empty_cache()
    return start.elapsed_time(end) / args.profile_times

def plot_profiled_flash_attn(results, suffix):
    fig, ax = plt.subplots()
    seqlens = [line[1] for line in results[1:]]
    times = [line[4] for line in results[1:]]
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Time (ms)')
    ax.plot(seqlens, times, '+', label='Flash Attention')
    coeff = polyfit(seqlens, times, 2)
    y_fit = polyval(coeff, seqlens)
    ax.plot(seqlens, y_fit, 'g', label=f'Fit Curve (y = {coeff[0]:.2f}x^2 + {coeff[1]:.2f}x + {coeff[2]:.2f})')
    ax.set_title(f'Flash Attention Profiling {args.model_size}')
    ax.legend()
    plt.savefig(os.path.join(args.output_fig_dir, f'flash_attn_profiled{suffix}.png'), dpi=1000)
    # plt.show()
    mse = np.mean((times - y_fit) ** 2)
    print(coeff, mse)
    with open(os.path.join(args.output_fig_dir, f'fit_curve{suffix}.txt'), 'w') as f:
        f.write(f'coeff: ' + str(coeff) + '\nmse: ' + str(mse))

def profile_flash_attn_all():
    profiled_results = {}
    if os.path.exists(args.cache_file):
        cached_results = pickle.load(open(args.cache_file, "rb"))
        profiled_results = cached_results["profiled_results"]
    result = [['batch_size', 'seq_len', 'num_heads', 'head_size', 'time']]
    if args.use_square_scope:
        seqlen = args.basic_seqlen
        while seqlen <= args.max_seqlen:
            if (args.batch_size, seqlen, args.num_heads, args.head_size) in profiled_results:
                time = profiled_results[(args.batch_size, seqlen, args.num_heads, args.head_size)]
                result.append([args.batch_size, seqlen, args.num_heads, args.head_size, time])
                continue
            inputs = generate_inputs([args.batch_size, seqlen, args.num_heads, args.head_size])
            time = profile_flash_attn_single(inputs, args.use_causal)
            result.append([args.batch_size, seqlen, args.num_heads, args.head_size, time])
            print(f'finish profile [{args.batch_size}, {seqlen}, {args.num_heads}, {args.head_size}], Proccess: {seqlen}/{args.max_seqlen}')
            profiled_results[(args.batch_size, seqlen, args.num_heads, args.head_size)] = time
            seqlen *= 2
    else:
        for seqlen in range(args.basic_seqlen, args.max_seqlen + 1, args.basic_seqlen):
            if (args.batch_size, seqlen, args.num_heads, args.head_size) in profiled_results:
                time = profiled_results[(args.batch_size, seqlen, args.num_heads, args.head_size)]
                result.append([args.batch_size, seqlen, args.num_heads, args.head_size, time])
                continue
            inputs = generate_inputs([args.batch_size, seqlen, args.num_heads, args.head_size])
            time = profile_flash_attn_single(inputs, args.use_causal)
            result.append([args.batch_size, seqlen, args.num_heads, args.head_size, time])
            print(f'finish profile [{args.batch_size}, {seqlen}, {args.num_heads}, {args.head_size}], Proccess: {seqlen}/{args.max_seqlen}')
            profiled_results[(args.batch_size, seqlen, args.num_heads, args.head_size)] = time
    suffix = f'_{args.model_size}_batch_size{args.batch_size}_seqlen{args.basic_seqlen}'
    if args.use_causal:
        suffix += '_causal'
    with open(os.path.join(args.output_dir, f'flash_attn_profiled{suffix}.csv'), 'w') as f:
        csv_writer = csv.writer(f)
        for line in result:
            csv_writer.writerow(line)
    with open(args.cache_file, "wb") as f:
        pickle.dump({"profiled_results": profiled_results}, f)
    plot_profiled_flash_attn(result, suffix)
    
profile_flash_attn_all()