#! /bin/bash
MASTER_ADDR=10.156.154.20
MASTER_PORT=7000
NNODES=2
NODE_RANK=$1
GPUS_PER_NODE=4

MACHINE=${2:-0}
RSP_SIZE=${3:-2}
DP_SIZE=${4:-2}
REPROFILE=${5:-0}

RUNTIME_PATH=$(pwd)/../results/
mkdir -p $RUNTIME_PATH
PROFILING_PATH=${RUNTIME_PATH}profiled-dist-comm-hetaceso/rank$NODE_RANK/

if [ $REPROFILE -eq 1 ]; then
    rm -rf ${PROFILING_PATH}
fi
mkdir -p ${PROFILING_PATH}
MAX_NUM_GPUS=4
MODEL_NAME=gpt
MODEL_SIZE=all

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

if [[ $MACHINE -eq "0" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=eno2
    export CUDA_VISIBLE_DEVICES=3,4,5,7
elif [[ $MACHINE -eq "1" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=eno1
    export CUDA_VISIBLE_DEVICES=3,4,5,6
elif [[ $MACHINE -eq "2" ]]; then
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_SOCKET_IFNAME=ens1f0
fi

echo [TIME] before profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S')
echo [TIME] before profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

torchrun $DISTRIBUTED_ARGS \
    comm_profiler_dist.py \
    --prof-path $PROFILING_PATH \
    --prof-fig-path $PROFILING_PATH/fig \
    --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_comm_profile.pkl \
    --prof-model-name $MODEL_NAME \
    --prof-model-size $MODEL_SIZE \
    --prof-rsp-size ${RSP_SIZE} \
    --prof-dp-size ${DP_SIZE} \
    --prof-warmup-times 20 \
    --prof-repeat-times 100 \
    --max-num-gpus $MAX_NUM_GPUS \
    2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_comm${MAX_NUM_GPUS}gpus.log

echo [TIME] after profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
echo [TIME] after profiling communication ${MAX_NUM_GPUS}-gpus : $(date '+%Y-%m-%d-%H-%M-%S')
