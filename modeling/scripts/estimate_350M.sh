#!/bin/bash
# 350M
mkdir -p ../results/estimate/350M

./scripts/estimate.sh 0 350M 8 1 1 1 4 > ../results/estimate/350M/test_0.txt
./scripts/estimate.sh 0 350M 8 2 1 1 2 > ../results/estimate/350M/test_1.txt
./scripts/estimate.sh 0 350M 8 1 2 1 2 > ../results/estimate/350M/test_2.txt
./scripts/estimate.sh 0 350M 8 1 1 2 2 > ../results/estimate/350M/test_3.txt
./scripts/estimate.sh 0 350M 8 2 2 1 1 > ../results/estimate/350M/test_4.txt
./scripts/estimate.sh 0 350M 8 1 2 2 1 > ../results/estimate/350M/test_5.txt
./scripts/estimate.sh 0 350M 8 1 1 4 1 > ../results/estimate/350M/test_6.txt
./scripts/estimate.sh 0 350M 8 1 4 1 1 > ../results/estimate/350M/test_7.txt