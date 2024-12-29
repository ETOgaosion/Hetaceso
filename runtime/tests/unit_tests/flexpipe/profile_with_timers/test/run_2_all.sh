#!/bin/bash
MACHINE=${1:-0}
TRAIN_ITERS=${2:-3}
RETRAIN=${3:-1}

for i in {6..7}
do
    ./run_2.sh $i $MACHINE $TRAIN_ITERS $RETRAIN
done