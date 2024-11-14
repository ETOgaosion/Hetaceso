#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_MPU=1
export NCCL_SOCKET_IFNAME=eno2
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_DEBUG=TRACE
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=3
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


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