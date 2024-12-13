#!/bin/bash
mkdir -p ../results/estimate

./scripts/estimate.sh 0 8 1 1 1 4 > ../results/estimate/test_0.txt
./scripts/estimate.sh 0 8 2 1 1 2 > ../results/estimate/test_1.txt
./scripts/estimate.sh 0 8 1 2 1 2 > ../results/estimate/test_2.txt
./scripts/estimate.sh 0 8 1 1 2 2 > ../results/estimate/test_3.txt
./scripts/estimate.sh 0 8 2 2 1 1 > ../results/estimate/test_4.txt
./scripts/estimate.sh 0 8 1 2 2 1 > ../results/estimate/test_5.txt
./scripts/estimate.sh 0 8 1 1 4 1 > ../results/estimate/test_6.txt
./scripts/estimate.sh 0 8 1 4 1 1 > ../results/estimate/test_7.txt