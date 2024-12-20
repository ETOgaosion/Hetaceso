import torch.profiler

def train_step():
    # 模拟训练逻辑
    x = torch.randn(2, 2, device="cuda")
    y = torch.randn(2, 2, device="cuda")
    return x + y

# 正确用法：分开独立使用 Profiler
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    record_shapes=True, profile_memory=True,
    with_stack=True, with_modules=True, with_flops=True,
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log1')
) as prof:
    for i in range(5):
        train_step()
        prof.step()

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log2')
) as prof:
    train_step()