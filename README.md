# Aceso for Heterogeneous environment

## Environment Setup

Clone the project:

```sh
git clone https://github.com/ETOgaosion/Hetaceso.git --recurse-submodules
```

If you forget to clone submodules, please do:

```sh
git submodules update --init --recursive
```

Then start docker, and enter the container::

```sh
chmod +x script/*.sh
./script/start_docker.sh
docker exec -it hetaceso-[USERNAME] bash
```

In the container, check the network:

```sh
root# curl google.com
<HTML><HEAD><meta http-equiv="content-type" content="text/html;charset=utf-8">
<TITLE>301 Moved</TITLE></HEAD><BODY>
<H1>301 Moved</H1>
The document has moved
<A HREF="http://www.google.com/">here</A>.
</BODY></HTML>
```

Install Transformer Engine in this project:

```sh
pip install -e external/TransformerEngine
```

It takes 10s minutes to finish setup.

## TODO

- [ ] Imbalanced assignment of dp/sp workloads
- [ ] Megatron new features support
    - [ ] RoPE (with CP)
    - [ ] MoE
    - [ ] overlap
    - [ ] Zero-1 (Distributed Saved Activation)
- [ ] Support for Double-CP (Ring-Ulysses)
- [ ] Support Profile and Search

## Develop Norm

1. Never push directly to dev branch, use pull request and discuss with other participants
2. Debug use dev-[username] branch, sync with dev branch
3. Function development use dev-[username]-[functionname] branch, can be independent
4. Use [Black formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
5. Try to include a function or whole debug process in one commit and PR, for others to check conviniently