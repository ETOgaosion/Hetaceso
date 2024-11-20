#! /bin/bash
RUNTIME_PATH=$(pwd)/../results/
PROFILING_PATH=${RUNTIME_PATH}profiled-p2p-hetaceso/
mkdir ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}p2p_inter_node.csv

MASTER_ADDR=localhost
NODE_RANK=$1

if [[ $NODE_RANK -eq 0 || $NODE_RANK -eq 1 ]]; then
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=7000 \
    NNODES=2 \
    GPUS_PER_NODE=1 \
    NODE_RANK=$NODE_RANK \
    FILE_NAME=$FILE_NAME \
    python3 p2p_band_profiler.py
else
    echo "Node rank $NODE_RANK is not in the list of nodes to profile"
fi