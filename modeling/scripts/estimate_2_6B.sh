#!/bin/bash
# 2_6B
mkdir -p ../results/estimate/2_6B

./scripts/estimate.sh 0 2_6B 8 1 1 1 4 > ../results/estimate/2_6B/test_0.txt
./scripts/estimate.sh 0 2_6B 8 2 1 1 2 > ../results/estimate/2_6B/test_1.txt
./scripts/estimate.sh 0 2_6B 8 1 2 1 2 > ../results/estimate/2_6B/test_2.txt
./scripts/estimate.sh 0 2_6B 8 1 1 2 2 > ../results/estimate/2_6B/test_3.txt
./scripts/estimate.sh 0 2_6B 8 2 2 1 1 > ../results/estimate/2_6B/test_4.txt
./scripts/estimate.sh 0 2_6B 8 1 2 2 1 > ../results/estimate/2_6B/test_5.txt
./scripts/estimate.sh 0 2_6B 8 1 1 4 1 > ../results/estimate/2_6B/test_6.txt
./scripts/estimate.sh 0 2_6B 8 1 4 1 1 > ../results/estimate/2_6B/test_7.txt