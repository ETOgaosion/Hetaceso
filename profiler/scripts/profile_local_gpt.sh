#! /bin/bash
MASTER_ADDR=localhost
MASTER_PORT=7000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

MACHINE=${1:-0}
REPROFILE=${2:-0}

RUNTIME_PATH=$(pwd)/../results/
mkdir -p $RUNTIME_PATH
PROFILING_PATH=${RUNTIME_PATH}profiled-gpt-hetaceso/rank$NODE_RANK/

VOCAB_FILE=/workspace/Hetaceso/runtime/vocabs/gpt2-vocab.json
MERGE_FILE=/workspace/Hetaceso/runtime/vocabs/gpt2-merges.txt
#  num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype are fake.
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
"
GPT_ARGS="
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --no-async-tensor-model-parallel-allreduce \
    --lr 0.00015 \
    --train-iters 20 \
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
"

FLEX_ARGS="
    --log-path ./logs \
    --nproc-per-node $GPUS_PER_NODE \
    --nnodes $NNODES \
"

if [ $REPROFILE -eq 1 ]; then
    rm -rf ${PROFILING_PATH}
fi
mkdir -p ${PROFILING_PATH}
mkdir -p logs
mkdir -p logs/csv
MAX_NUM_GPUS=4
MAX_TP_SIZE=2
MODEL_NAME=gpt
MODEL_SIZE=all

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

for ((tp_size=1; tp_size<=$MAX_TP_SIZE; tp_size=tp_size*2))
do
    for ((usp_size=1; usp_size<=$MAX_NUM_GPUS/tp_size; usp_size=usp_size*2))
    do
        for ((rsp_size=1; rsp_size<=$MAX_NUM_GPUS/tp_size/usp_size; rsp_size=rsp_size*2))
        do
        GPUS_PER_NODE=$((tp_size*usp_size*rsp_size))
        DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

        FLEX_ARGS="
            --log-path ./logs \
            --nproc-per-node $GPUS_PER_NODE \
            --nnodes $NNODES \
        "

        echo [TIME] before profiling usp $usp_size rsp $rsp_size tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

        torchrun $DISTRIBUTED_ARGS \
            op_profiler.py \
            ${DATA_ARGS} \
            ${GPT_ARGS} \
            ${FLEX_ARGS} \
            --use-mcore-models \
            --prof-op \
            --prof-tp-size $tp_size \
            --prof-usp-size $usp_size \
            --prof-rsp-size $rsp_size \
            --prof-path $PROFILING_PATH \
            --prof-cache-file ${PROFILING_PATH}${MODEL_NAME}_op_profile.pkl \
            --prof-model-name $MODEL_NAME \
            --prof-model-size $MODEL_SIZE \
            --prof-warmup-times 20 \
            --prof-repeat-times 100 \
            2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_op_usp${usp_size}_rsp${rsp_size}_tp${tp_size}.log

        echo [TIME] after profiling usp $usp_size rsp $rsp_size tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
        done
    done
done