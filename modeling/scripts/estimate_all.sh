#!/bin/bash

echo "========== test 350M ==========="
./scripts/estimate_350M.sh
echo "========== test 1_3B ==========="
./scripts/estimate_1_3B.sh
echo "========== test 2_6B ==========="
./scripts/estimate_2_6B.sh
echo "========== test 6_7B ==========="
./scripts/estimate_6_7B.sh
echo "========== test 13B ==========="
./scripts/estimate_13B.sh