#!/bin/bash 
addrs=(172.31.37.189 172.31.31.19 172.31.33.155 172.31.20.160)
for addr in "${addrs[@]}"; do
    echo "in ${addr}"
    ssh -p 22 ${addr} -o "StrictHostKeyChecking no" "cd /home/ubuntu/code/python/Hetaceso/runtime/tests/unit_tests/flexpipe/metis && ./kill_local.sh"
done