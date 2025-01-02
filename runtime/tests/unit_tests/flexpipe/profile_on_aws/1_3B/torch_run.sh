#!/bin/bash

export DEBUG_COMMUNICATE=1
export DEBUG_MPU=1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_JIT=0
export PYTORCH_NVFUSER_DISABLE=fallback
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

GPUS_PER_NODE=16
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
GLOBAL_BATCH_SIZE=1024

TEST_NUM=${1:-0}
TRAIN_ITERS=${2:-3}
RETRAIN=${3:-1}

VOCAB_FILE=../../../../../vocabs/gpt2-vocab.json
MERGE_FILE=../../../../../vocabs/gpt2-merges.txt

if [ $RETRAIN -eq 1 ]; then
    rm -rf logs_${TEST_NUM}
fi
mkdir -p logs_${TEST_NUM}
mkdir -p logs_${TEST_NUM}/profile_torch

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
    --disable-all-timers \
    --profile \
    --profile-method torch \
    --profile-ranks 0 1 2 3 \
    --profile-output-dir logs_${TEST_NUM}/profile_torch \
"

FLEX_ARGS="
    --flexpipe-config ./prof_pretrain_${TEST_NUM}.json \
    --log-path ./logs_${TEST_NUM} \
    --nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
"

mkdir -p logs
mkdir -p logs/csv

# export USE_FUSED_ATTN=1 && \
export TIMERS_LOG_LEVEL=0 && \
export USE_FLASH_ATTN=1 && \
export TIMERS_LOG_LEVEL=0 && \
torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $GPT_ARGS \
    $PROFILE_ARGS \
    $FLEX_ARGS \
    $DATA_ARGS \
    --distributed-backend nccl \