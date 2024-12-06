num_gpus_per_node=4
num_nodes=1
MBS=${1:-4}
TP=${2:-1}
USP=${3:-1}
RSP=${4:-1}
DP=${5:-4}

CONFIG_TEST=mbs${MBS}_tp${TP}_usp${USP}_rsp${RSP}_dp${DP}

config=single_gpu_configs/gpt_350M_$CONFIG_TEST.json
PROFILED_GPT_PATH=../results/profiled-gpt-hetaceso/
PROFILED_DIST_P2P_PATH=../results/profiled-dist-p2p-hetaceso/
PROFILED_LOCAL_P2P_PATH=../results/profiled-local-p2p-hetaceso/
PROFILED_LOCAL_COMM_PATH=../results/profiled-local-comm-hetaceso/

SAVE_TO_CSV=../results/search_results/

mkdir -p $SAVE_TO_CSV

echo $config

python3 aceso_cost_model.py \
    --initial-point $config \
    --profiled-gpt-path $PROFILED_GPT_PATH \
    --profiled-dist-p2p-path $PROFILED_DIST_P2P_PATH \
    --profiled-local-p2p-path $PROFILED_LOCAL_P2P_PATH \
    --profiled-local-comm-path $PROFILED_LOCAL_COMM_PATH \
    --num-gpus-per-node $num_gpus_per_node \
    --num-nodes $num_nodes \
    --save-to-csv $SAVE_TO_CSV \
    --dist-optimizer \
    --support-comm-predict