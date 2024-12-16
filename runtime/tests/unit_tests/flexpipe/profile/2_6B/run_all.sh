#!/bin/bash
MACHINE=${1:-0}

for i in {0..7}
do
    ./run.sh $i $MACHINE
done