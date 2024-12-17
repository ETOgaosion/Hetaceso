#!/bin/bash
# 6_7B
mkdir -p ../results/estimate/6_7B

./scripts/estimate.sh 0 6_7B 8 1 1 1 4 > ../results/estimate/6_7B/test_0.txt
./scripts/estimate.sh 0 6_7B 8 2 1 1 2 > ../results/estimate/6_7B/test_1.txt
./scripts/estimate.sh 0 6_7B 8 1 2 1 2 > ../results/estimate/6_7B/test_2.txt
./scripts/estimate.sh 0 6_7B 8 1 1 2 2 > ../results/estimate/6_7B/test_3.txt
./scripts/estimate.sh 0 6_7B 4 2 2 1 1 > ../results/estimate/6_7B/test_4.txt
./scripts/estimate.sh 0 6_7B 4 1 2 2 1 > ../results/estimate/6_7B/test_5.txt
./scripts/estimate.sh 0 6_7B 4 1 1 4 1 > ../results/estimate/6_7B/test_6.txt
./scripts/estimate.sh 0 6_7B 4 1 4 1 1 > ../results/estimate/6_7B/test_7.txt