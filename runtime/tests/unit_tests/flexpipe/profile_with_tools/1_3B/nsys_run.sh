#!/bin/bash

export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_FILE=./nccl.log
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_IB_DISABLE=1
# export NCCL_SET_THREAD_NAME=1
# export NCCL_TOPO_FILE=nccl/rank_topo.xml
# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_P2P_DISABLE=1

if [ -e export.sh ]; then
    source export.sh
fi

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# fixed Model related configuration here, pls not overlap with json config
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=32
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=8
GLOBAL_BATCH_SIZE=32

TEST_NUM=${1:-0}
MACHINE=${2:-0}
TRAIN_ITERS=${3:-3}
RETRAIN=${4:-1}

if [[ $MACHINE -eq "0" ]]; then
    export NCCL_SOCKET_IFNAME=eno2
    export CUDA_VISIBLE_DEVICES=3,4,5,7
elif [[ $MACHINE -eq "1" ]]; then
    export NCCL_SOCKET_IFNAME=eno1
    export CUDA_VISIBLE_DEVICES=3,4,5,6
elif [[ $MACHINE -eq "2" ]]; then
    export NCCL_SOCKET_IFNAME=ens1f0
fi

VOCAB_FILE=../../../../../vocabs/gpt2-vocab.json
MERGE_FILE=../../../../../vocabs/gpt2-merges.txt

if [ $RETRAIN -eq 1 ]; then
    rm -rf logs_${TEST_NUM}
fi
mkdir -p logs_${TEST_NUM}
mkdir -p logs_${TEST_NUM}/profile_nsys

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
"

# Model related configuration here, pls not overlap with json config
GPT_ARGS="
    --no-async-tensor-model-parallel-allreduce \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.00015 \
    --train-iters $TRAIN_ITERS \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type GPT2BPETokenizer \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --no-scatter-gather-tensors-in-pipeline \
"

PROFILE_ARGS="
    --profile \
    --profile-method nsys \
    --profile-step-start 1 \
    --profile-step-end $TRAIN_ITERS \
    --profile-ranks 0 1 2 3 \
"

FLEX_ARGS="
    --flexpipe-config ./test_pretrain_${TEST_NUM}.json \
    --log-path ./logs_${TEST_NUM} \
    --nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --distributed-backend nccl \
"

NSIGHT_PROFILE_ARGS=(
    # output
    -w true
    -o logs_${TEST_NUM}/profile_nsys/res
    -f true
    -x true
    #  cuda                   os           python
    -t cuda,nvtx,cudnn,cublas,osrt,syscall,python-gil
    # GPU/CUDA
    --capture-range=cudaProfilerApi --capture-range-end=stop
    --cudabacktrace=all
    --cuda-memory-usage=true
    --python-backtrace=cuda
    --gpuctxsw=true
    --gpu-metrics-devices=all
    --enable nvml_metrics # NVML Power and temperature
    # --soc-metrics=true
    # CPU
    --cpuctxsw=process-tree
    # Network
    # NVSHMEM_NVTX=common
    # NIC/IB metrics
    --enable network_interface # Check Multiple --enable
    # Python backtrace
    --python-sampling=true
    --python-functions-trace=/opt/nvidia/nsight-systems-cli/2024.7.1/target-linux-x64/PythonFunctionsTrace/annotations.json
)

mkdir -p logs
mkdir -p logs/csv

# export USE_FUSED_ATTN=1 && \
export USE_FLASH_ATTN=1

nsys profile \
    ${NSIGHT_PROFILE_ARGS[@]} \
    torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PROFILE_ARGS \
    $FLEX_ARGS \
    $DATA_ARGS \