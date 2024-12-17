#!/bin/bash
MACHINE=${1:-0}
MAX_TEST=${2:-3}

echo "========== test 350M ==========="
cd 350M && ./run_all.sh $MACHINE
if [ $MAX_TEST -eq 1 ]; then
    exit
fi
echo "========== test 1_3B ==========="
cd ../1_3B && ./run_all.sh $MACHINE
if [ $MAX_TEST -eq 2 ]; then
    exit
fi
echo "========== test 2_6B ==========="
cd ../2_6B && ./run_all.sh $MACHINE
if [ $MAX_TEST -eq 3 ]; then
    exit
fi
echo "========== test 6_7B ==========="
cd ../6_7B && ./run_all.sh $MACHINE
if [ $MAX_TEST -eq 4 ]; then
    exit
fi
echo "========== test 13B ==========="
cd ../13B && ./run_all.sh $MACHINE
if [ $MAX_TEST -eq 5 ]; then
    exit
fi