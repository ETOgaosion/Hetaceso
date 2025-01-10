#! /bin/bash
MASTER_ADDR=10.156.154.20
NODE_RANK=$1
OTHER_NODE_RANK=$2

MACHINE=${3:-0}
REPROFILE=${4:-0}

RUNTIME_PATH=$(pwd)/../results/
PROFILING_PATH=${RUNTIME_PATH}profiled-dist-p2p-hetaceso/${NODE_RANK}-${OTHER_NODE_RANK}
if [ $REPROFILE -eq 1 ]; then
    rm -rf ${PROFILING_PATH}
fi
mkdir -p ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}p2p_inter_node.csv
FIG_PATH=${PROFILING_PATH}/fig

if [ "$MACHINE" -eq 0 ]; then
    export NCCL_SOCKET_IFNAME=eno2
elif [ "$MACHINE" -eq 1 ]; then
    export NCCL_SOCKET_IFNAME=eno1
elif [ "$MACHINE" -eq 2 ]; then
    export NCCL_SOCKET_IFNAME=ens1f0
fi

if [[ $NODE_RANK -eq 0 || $NODE_RANK -eq 1 ]]; then
    MASTER_ADDR=$MASTER_ADDR \
    MASTER_PORT=7000 \
    NNODES=2 \
    GPUS_PER_NODE=1 \
    NODE_RANK=$NODE_RANK \
    FILE_NAME=$FILE_NAME \
    FIG_PATH=$FIG_PATH \
    python3 p2p_band_profiler.py
else
    echo "Node rank $NODE_RANK is not in the list of nodes to profile"
fi