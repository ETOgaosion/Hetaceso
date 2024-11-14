import torch
import os

master_addr = os.getenv("MASTER_ADDR", "localhost")
master_port = os.getenv("MASTER_PORT", "6000")
world_size = int(os.getenv("WORLD_SIZE", 1))
node_rank = int(os.getenv("RANK", 0))

torch.distributed.init_process_group(
    backend="nccl",
    init_method=f"tcp://{master_addr}:{master_port}",
    world_size=world_size,
    rank=node_rank,
)

print(f"Hello from node {node_rank} of {world_size}")