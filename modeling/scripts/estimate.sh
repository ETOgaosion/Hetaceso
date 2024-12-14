RANK=${1:-0}
MODEL_SIZE=${2:-350M}
MBS=${3:-8}
TP=${4:-1}
USP=${5:-1}
RSP=${6:-1}
DP=${7:-4}

CONFIG_TEST=mbs${MBS}_tp${TP}_usp${USP}_rsp${RSP}_dp${DP}
config=gpu_configs/${MODEL_SIZE}/gpt_${CONFIG_TEST}.json

if [ ! -f $config ]; then
    echo "Config file not found!"
    exit 1
fi

TOPO_INDEX=${7:-0}
TOPO_DIR=machine_topos
TOPO_FILE=$TOPO_DIR/topo_$TOPO_INDEX.json

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