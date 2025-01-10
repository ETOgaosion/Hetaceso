#!/bin/bash
MODEL_SIZE=("350M" "1_3B" "2_6B" "6_7B")
# MODEL_SIZE=("350M" "1_3B" "2_6B" "6_7B" "13B")

for model_size in "${MODEL_SIZE[@]}"
do
    echo "Profiling Flash Attention on GPT model size: $model_size"
    python flash_attn_profiler.py
    --model-size $model_size
done