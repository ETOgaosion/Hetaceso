#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export DEBUG_COMMUNICATE=1
# export DEBUG_PARALLEL_STATES=1
# export NCCL_IB_DISABLE=1
# export NCCL_IBEXT_DISABLE=1
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_FILE=./nccl.log
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_SOCKET_IFNAME=eno1
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export TORCH_NCCL_BLOCKING_WAIT=1     # 强制同步等待并显示错误
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1  # 启用异步错误检测
GPUS_PER_NODE=8
NNODES=2
MASTER_ADDR=172.31.37.189
MASTER_PORT=6000
VOCAB_FILE=../../../../vocabs/gpt2-vocab.json
MERGE_FILE=../../../../vocabs/gpt2-merges.txt

# 初始化变量
NODE_RANK=""
MODEL_NAME=""
SEQ_LENGTH=""
MICRO_BATCH_SIZE=""
GLOBAL_BATCH_SIZE=""
FLEX_CONFIG=""
PRESET_RANKS=""

# 解析命令行选项
while getopts "n:m:s:u:g:f:p:" opt; do
    case $opt in
        n)
            NODE_RANK=$OPTARG
            ;;
        m)
            MODEL_NAME=$OPTARG
            ;;
        s)
            SEQ_LENGTH=$OPTARG
            ;;
        u)
            MICRO_BATCH_SIZE=$OPTARG
            ;;
        g)
            GLOBAL_BATCH_SIZE=$OPTARG
            ;;
        f)
            FLEX_CONFIG=$OPTARG
            ;;
        \?)
            echo "invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "option -$OPTARG need an arg." >&2
            exit 1
            ;;
    esac
done

if [ -z "$NODE_RANK" ] || [ -z "$MODEL_NAME" ] || [ -z "$SEQ_LENGTH" ] || [ -z "$MICRO_BATCH_SIZE" ] || [ -z "$GLOBAL_BATCH_SIZE" ] || [ -z "$FLEX_CONFIG" ]; then
    echo "Error: all args（node_rank, model_name, sequence_length, micro_batch_size, global_batch_size, flex_config, preset_ranks）are essential。"
    echo "Usage: $0 -n <node_rank> -m <model_name> -s <sequence_length> -u <micro_batch_size> -g <global_batch_size> -f <flex_config>"
    exit 1
fi

if [ "$MODEL_NAME" = "GPT_350M" ]; then
    HIDDEN_SIZE=1024
    NUM_ATTENTION_HEADS=16
elif [ "$MODEL_NAME" = "GPT_1-3B" ]; then
    HIDDEN_SIZE=2048
    NUM_ATTENTION_HEADS=32
elif [ "$MODEL_NAME" = "GPT_2-6B" ]; then
    HIDDEN_SIZE=2560
    NUM_ATTENTION_HEADS=32
elif [ "$MODEL_NAME" = "GPT_6-7B" ]; then
    HIDDEN_SIZE=4096
    NUM_ATTENTION_HEADS=32
else
    echo "error: not support model_name: $MODEL_NAME"
    exit 1
fi


WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH


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
    --train-iters 2 \
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
    --distributed-backend nccl \
"

filename=$(basename "$FLEX_CONFIG")
file_base=$(echo "$filename" | sed 's/\.[^.]*$//')

FLEX_ARGS="
    --flexpipe-config ${FLEX_CONFIG} \
    --log-path ./logs_${MODEL_NAME}_${SEQ_LENGTH} \
    --nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
"

mkdir -p ./logs
mkdir -p ./logs_${MODEL_NAME}_${SEQ_LENGTH}/csv



# export USE_FUSED_ATTN=1 && \
export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \
torchrun $DISTRIBUTED_ARGS \
    pretrain_gpt.py \
    $GPT_ARGS \
    $FLEX_ARGS \
    $DATA_ARGS \
    2>&1 | tee ./logs_${MODEL_NAME}_${SEQ_LENGTH}/node_${NODE_RANK}.log