#! /bin/bash
NODE_RANK=0

REPROFILE=${1:-0}

RUNTIME_PATH=$(pwd)/../results/
mkdir -p $RUNTIME_PATH
PROFILING_PATH=${RUNTIME_PATH}profiled-local-p2p-hetaceso/rank$NODE_RANK
if [ $REPROFILE -eq 1 ]; then
    rm -rf ${PROFILING_PATH}
fi
mkdir -p ${PROFILING_PATH}
FILE_NAME=${PROFILING_PATH}/p2p_intra_node.csv
FIG_PATH=${PROFILING_PATH}/fig
mkdir -p $FIG_PATH

MASTER_ADDR=localhost \
MASTER_PORT=7000 \
NNODES=1 \
GPUS_PER_NODE=2 \
NODE_RANK=$NODE_RANK \
FILE_NAME=$FILE_NAME \
FIG_PATH=$FIG_PATH \
USE_LATER_HALF=1 \
python3 p2p_band_profiler.py