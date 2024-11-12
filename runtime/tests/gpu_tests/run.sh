#!/bin/bash

export GPUS_PER_NODE=8
export MASTER_ADDR=localhost
export MASTER_PORT=6000

python -m torch.distributed.run \
--nproc_per_node $GPUS_PER_NODE --nnodes 1 --node_rank 0 \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch_distributed_gpu_test.py