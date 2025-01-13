#!/bin/bash
MODEL_SIZE=("350M" "1_3B" "2_6B" "6_7B")
# MODEL_SIZE=("350M" "1_3B" "2_6B" "6_7B" "13B")

REPROFILE=${1:-0}

RUNTIME_PATH=$(pwd)/../results/
mkdir -p $RUNTIME_PATH
PROFILING_PATH=${RUNTIME_PATH}profiled-flash-attn-hetaceso/
FIG_PATH=$PROFILING_PATH/fig

if [ $REPROFILE -eq 1 ]; then
    rm -rf ${PROFILING_PATH}
fi
mkdir -p ${PROFILING_PATH}
mkdir -p ${FIG_PATH}

for model_size in "${MODEL_SIZE[@]}"
do
    echo "Profiling Flash Attention on GPT model size: $model_size"
    python flash_attn_profiler.py \
    --model-size $model_size \
    --output-dir $PROFILING_PATH \
    --output-fig-dir $FIG_PATH \
    --cache-file ${PROFILING_PATH}/cache.pkl
done