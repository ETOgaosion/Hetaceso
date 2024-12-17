#!/bin/bash
MACHINE=${1:-0}

echo "========== test 350M ==========="
cd 350M && ./run_all.sh $MACHINE
echo "========== test 1_3B ==========="
cd ../1_3B && ./run_all.sh $MACHINE
echo "========== test 2_6B ==========="
cd ../2_6B && ./run_all.sh $MACHINE
# echo "========== test 6_7B ==========="
# cd ../6_7B && ./run_all.sh
# echo "========== test 13B ==========="
# cd ../13B && ./run_all.sh