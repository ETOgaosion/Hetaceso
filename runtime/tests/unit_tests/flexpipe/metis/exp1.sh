#!/bin/bash 
# "Usage: $0 -n <node_rank> -m <model_name> -s <sequence_length> -u <micro_batch_size> -g <global_batch_size> -f <flex_config>"
./run_rank.sh -n 0 -m GPT_1-3B -s 4096  -u 2 -g 1024  -f ./config/exp1/1-3B/GPT_1-3B_seq-4096_a10g_dps-1_l40s_dps-3.json