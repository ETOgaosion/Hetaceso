#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=10.156.154.242
MASTER_PORT=6000
NNODES=3
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

if [ "$NODE_RANK" -ne 1 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
else
    export CUDA_VISIBLE_DEVICES=4,5,6,7
fi

if [ "$NODE_RANK" -eq 2 ]; then
    export NCCL_SOCKET_IFNAME=eno1
else
    export NCCL_SOCKET_IFNAME=eno2
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

mkdir -p logs
mkdir -p logs/csv
mkdir -p nccl

torchrun $DISTRIBUTED_ARGS \
    test_process_group.py \