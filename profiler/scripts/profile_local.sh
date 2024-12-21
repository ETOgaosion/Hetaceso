#!/bin/bash

MACHINE=${1:-0}

./scripts/profile_local_p2p.sh $MACHINE
./scripts/profile_local_gpt.sh $MACHINE
./scripts/profile_local_comm.sh $MACHINE