RANK=${1:-0}
MBS=${2:-8}
TP=${3:-1}
USP=${4:-1}
RSP=${5:-1}
DP=${6:-4}

CONFIG_TEST=mbs${MBS}_tp${TP}_usp${USP}_rsp${RSP}_dp${DP}

TOPO_INDEX=${7:-0}
TOPO_DIR=machine_topos
TOPO_FILE=$TOPO_DIR/topo_$TOPO_INDEX.json

config=single_gpu_configs/gpt_350M_${CONFIG_TEST}.json
PROFILED_GPT_PATH=../results/profiled-gpt-hetaceso/
PROFILED_DIST_P2P_PATH=../results/profiled-dist-p2p-hetaceso/
PROFILED_LOCAL_P2P_PATH=../results/profiled-local-p2p-hetaceso/
PROFILED_LOCAL_COMM_PATH=../results/profiled-local-comm-hetaceso/

SAVE_TO_CSV=../results/search_results/

mkdir -p $SAVE_TO_CSV

echo $config

python3 hetaceso_cost_model.py \
    --initial-point $config \
    --profiled-gpt-path $PROFILED_GPT_PATH \
    --profiled-dist-p2p-path $PROFILED_DIST_P2P_PATH \
    --profiled-local-p2p-path $PROFILED_LOCAL_P2P_PATH \
    --profiled-local-comm-path $PROFILED_LOCAL_COMM_PATH \
    --topo-file $TOPO_FILE \
    --save-to-csv $SAVE_TO_CSV \
    --dist-optimizer \
    --rank ${RANK} \
    --node-rank ${RANK}