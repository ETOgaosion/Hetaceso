
# Debug 9.12
#### tokenizer.py
modify:
[args.max_mp_size](runtime/megatron/training/tokenizer/tokenizer.py#L69)

#### parallel_state.py
modify:
[print_args](runtime/megatron/training/arguments.py#L528) #先注释掉打印参数

[入口1](runtime/megatron/core/parallel_state.py#L445)
 #全部改成和aceso相同
- [modify func](runtime/megatron/core/parallel_state.py#L498) 
- [modify func](runtime/megatron/core/parallel_state.py#L712)
[入口2](runtime/megatron/core/parallel_state.py#L448)
- [modify func](runtime/megatron/core/parallel_state.py#L887)
- [modify func](runtime/megatron/core/parallel_state.py#L529)



[注释](runtime/megatron/core/tensor_parallel/random.py#L190) 涉及到expert_parallel 先注释掉