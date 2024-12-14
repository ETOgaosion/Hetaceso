#!/bin/bash
# 13B
mkdir -p ../results/estimate/13B

./scripts/estimate.sh 0 13B 8 1 1 1 4 > ../results/estimate/13B/test_0.txt
./scripts/estimate.sh 0 13B 4 2 1 1 2 > ../results/estimate/13B/test_1.txt
./scripts/estimate.sh 0 13B 4 1 2 1 2 > ../results/estimate/13B/test_2.txt
./scripts/estimate.sh 0 13B 4 1 1 2 2 > ../results/estimate/13B/test_3.txt
./scripts/estimate.sh 0 13B 2 2 2 1 1 > ../results/estimate/13B/test_4.txt
./scripts/estimate.sh 0 13B 2 1 2 2 1 > ../results/estimate/13B/test_5.txt
./scripts/estimate.sh 0 13B 2 1 1 4 1 > ../results/estimate/13B/test_6.txt
./scripts/estimate.sh 0 13B 2 1 4 1 1 > ../results/estimate/13B/test_7.txt