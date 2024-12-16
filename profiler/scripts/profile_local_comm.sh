#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=2
NODE_RANK=0

MACHINE=${1:-0}
REPROFILE=${2:-0}

RUNTIME_PATH=$(pwd)/../results/
mkdir -p $RUNTIME_PATH
PROFILING_PATH=${RUNTIME_PATH}profiled-local-comm-hetaceso/rank$NODE_RANK/
PROFILING_OP_TIME_PATH=${RUNTIME_PATH}profiled-gpt-hetaceso/rank$NODE_RANK/

if [ $REPROFILE -eq 1 ]; then
    rm -rf ${PROFILING_PATH}
fi
mkdir -p ${PROFILING_PATH}
MAX_NUM_GPUS=4
MODEL_NAME=gpt
MODEL_SIZE=all

if [[ $MACHINE -eq "0" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=eno2
    export CUDA_VISIBLE_DEVICES=3,4,5,7
    MAX_DATA_SIZE=4096
elif [[ $MACHINE -eq "1" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=eno1
    export CUDA_VISIBLE_DEVICES=3,4,5,6
    MAX_DATA_SIZE=4096
elif [[ $MACHINE -eq "2" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=ens1f0
    MAX_DATA_SIZE=16384
fi

echo [TIME] before profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S')
echo [TIME] before profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

python3 comm_profiler.py \
    --prof-path $PROFILING_PATH \
    --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_comm_profile.pkl \
    --prof-op-time-path $PROFILING_OP_TIME_PATH \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-warmup-times 20 \
    --prof-repeat-times 100 \
    --max-num-gpus $MAX_NUM_GPUS \
    --max-data-size $MAX_DATA_SIZE \
    2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_comm${MAX_NUM_GPUS}gpus.log

echo [TIME] after profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
echo [TIME] after profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S')
