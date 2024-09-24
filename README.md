# Aceso for Heterogeneous environment

## Environment Setup

execute:

```sh
chmod +x script/*.sh
./script/start_docker.sh
```

## TODO

- [ ] Share weights between embedding layer and post process output layer, use Megatron's `_allreduce_word_embedding_grads` mechanism and `_EMBEDDING_GROUP` in parallel_state.py. Need to add first and last pp rank into it. Currently use `untie_embeddings_and_output_weights` to disable weight sharing, remove it if finish implementation.
    - [ ] allreduce position embedding grads not supported either
- [ ] Other Configurations debug
    - [ ] flexible tp/dp
    - [ ] flexible recompute
- [ ] Megatron new features support
    - [ ] RoPE (with CP)
    - [ ] MoE
    - [ ] overlap
    - [ ] Zero-1 (Distributed Saved Activation)
- [ ] Connect with Transformer Engine/Fused Attention and Add support for Megatron-CP
- [ ] Add support for Double-CP (Ring-Ulysses)
- [ ] Support Profile and Search
- [ ] Optimize codes
    - [ ] Use dry run instead of hard-coded tensor shapes

## Develop Norm

1. Never push directly to dev branch, use pull request and discuss with other participants
2. Debug use dev-[username] branch, sync with dev branch
3. Function development use dev-[username]-[functionname] branch, can be independent
4. Use [Black formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
5. Try to include a function or whole debug process in one commit and PR, for others to check conviniently