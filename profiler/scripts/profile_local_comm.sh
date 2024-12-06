#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=2
NODE_RANK=$1

MACHINE=${1:-0}
REPROFILE=${2:-0}

RUNTIME_PATH=$(pwd)/../results/
PROFILING_PATH=${RUNTIME_PATH}profiled-local-comm-hetaceso/
PROFILING_OP_TIME_PATH=${RUNTIME_PATH}profiled-gpt-hetaceso/

mkdir -p ${PROFILING_PATH}
MAX_NUM_GPUS=4
MODEL_NAME=gpt
MODEL_SIZE=350M

if [[ $MACHINE -eq "0" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=eno2
    export CUDA_VISIBLE_DEVICES=3,4,5,7
else
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=eno1
fi

for ((num_gpus=2; num_gpus<=$MAX_NUM_GPUS; num_gpus=num_gpus*2))
do
    echo [TIME] before profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S')
    echo [TIME] before profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    python3 comm_profiler.py \
        --prof-path $PROFILING_PATH \
        --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_comm_profile.pkl \
        --prof-op-time-path $PROFILING_OP_TIME_PATH \
        --prof-tp-size $num_gpus \
        --prof-model-name $MODEL_NAME \
        --prof-model-size $MODEL_SIZE \
        --prof-warmup-times 5 \
        --prof-repeat-times 20 \
        --max-data-size 4096 \
        2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_comm${num_gpus}gpus.log

    echo [TIME] after profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
    echo [TIME] after profiling communication ${num_gpus}-gpus : $(date '+%Y-%m-%d-%H-%M-%S')

done