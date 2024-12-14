#!/bin/bash

echo "========== test 350M ==========="
./estimate_350M.sh
echo "========== test 1_3B ==========="
./estimate_1_3B.sh
echo "========== test 2_6B ==========="
./estimate_2_6B.sh
echo "========== test 6_7B ==========="
./estimate_6_7B.sh
echo "========== test 13B ==========="
./estimate_13B.sh