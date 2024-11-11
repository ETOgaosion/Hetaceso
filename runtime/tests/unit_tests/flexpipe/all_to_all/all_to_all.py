import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import combinations

def init_process(rank, world_size, backend='nccl'):
    # Set up environment variables for master address and port
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['NCCL_SOCKET_IFNAME'] = 'eno2'
    
    # Initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def create_communication_groups(world_size):
    # Generate all unique GPU pairs and create communication groups for them
    # pairs = [
    #     # (0, 1), (2, 3), (4, 5), (6, 7), (1, 2), (3, 4), (5, 6), (0, 7),
    #     # (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 0), (7, 1),
    #     # (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 0), (6, 1), (7, 2),
    #     # (0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3),
    #     # (0, 5), (1, 6), (2, 7), (3, 0), (4, 1), (5, 2), (6, 3), (7, 4),
    #     # (0, 6), (1, 7), (2, 0), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5),
    #     # (0, 7), (1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)
    # ]
    pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 0),
        (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 0),
        (3, 1), (3, 2), (3, 4), (3, 5), (3, 6), (3, 7), (3, 0),
        (4, 1), (4, 2), (4, 3), (4, 5), (4, 6), (4, 7), (4, 0),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 6), (5, 7), (5, 0),
        (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 7), (6, 0),
        (7, 1), (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 0)
    ]
    # pairs = [
    # ]
    print(pairs)
    comm_groups = {}
    
    for i, (r1, r2) in enumerate(pairs):
        group = dist.new_group([r1, r2])
        comm_groups[(r1, r2)] = group
    
    return comm_groups

def all_to_all_test(rank, world_size, test_rank = None, test_rank_2 = None, neg = False):
    init_process(rank, world_size)
    
    # Create communication groups
    comm_groups = create_communication_groups(world_size)
    
    # Each process prepares a tensor to send
    shape = [2, 4, 256, 8, 64]
    send_tensor = torch.ones(shape, device=f'cuda:{rank}', dtype=torch.bfloat16) * rank
    recv_tensor = torch.zeros(shape, device=f'cuda:{rank}', dtype=torch.bfloat16)
    
    for (src, dst), group in comm_groups.items():
        if test_rank is not None:
            if test_rank_2 is not None:
                if not neg:
                    if (rank == src or rank == dst) and ((src in test_rank and dst in test_rank_2) or (src in test_rank_2 and dst in test_rank)):
                        # Determine the peer for each rank in the pair
                        peer = dst if rank == src else src
                        
                        # Perform all_to_all_single within the group
                        dist.all_to_all_single(recv_tensor, send_tensor, group=group)
                        
                        # Print results for verification
                        print(f'Rank {rank} (peer {peer}): send_tensor = {send_tensor.shape}, recv_tensor = {recv_tensor.shape}')
                else:
                    if (rank == src or rank == dst) and (src in test_rank or dst in test_rank) and not ((src in test_rank and dst in test_rank_2) or (src in test_rank_2 and dst in test_rank)):
                        # Determine the peer for each rank in the pair
                        peer = dst if rank == src else src
                        print(f'Rank {rank} (peer {peer}): start test')
                        
                        # Perform all_to_all_single within the group
                        dist.all_to_all_single(recv_tensor, send_tensor, group=group)
                        
                        # Print results for verification
                        print(f'Rank {rank} (peer {peer}): send_tensor = {send_tensor.shape}, recv_tensor = {recv_tensor.shape}')
            else:
                if (rank == src or rank == dst) and (src in test_rank or dst in test_rank):
                    # Determine the peer for each rank in the pair
                    peer = dst if rank == src else src
                    
                    # Perform all_to_all_single within the group
                    dist.all_to_all_single(recv_tensor, send_tensor, group=group)
                    
                    # Print results for verification
                    print(f'Rank {rank} (peer {peer}): send_tensor = {send_tensor.shape}, recv_tensor = {recv_tensor.shape}')
        else:
            if rank == src or rank == dst:
                # Determine the peer for each rank in the pair
                peer = dst if rank == src else src
                
                # Perform all_to_all_single within the group
                dist.all_to_all_single(recv_tensor, send_tensor, group=group)
                
                # Print results for verification
                print(f'Rank {rank} (peer {peer}): send_tensor = {send_tensor.shape}, recv_tensor = {recv_tensor.shape}')
    
    # Cleanup
    # dist.destroy_process_group()
    print(f'rank: {rank}, exit')

if __name__ == "__main__":
    world_size = 8
    mp.spawn(all_to_all_test, args=(world_size,), nprocs=world_size)