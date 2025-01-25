#!/usr/bin/env bash

# Default random seed
RANDOM_SEED=88888

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed|-r) RANDOM_SEED="$2"; shift ;; # Set the random seed
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "Using random seed: $RANDOM_SEED"

# This script runs metasurface_progress_seed.lua for N_quarter = 1..4,
# each with num_shapes=100000, random_seed=$RANDOM_SEED, and store_data = true (-s).
# All tasks are run in the background in parallel. Then we wait for all to complete.

../build/S4 -a "1 100000 $RANDOM_SEED -s" -t 32 metasurface_progress_seed.lua &
../build/S4 -a "2 100000 $RANDOM_SEED -s" -t 32 metasurface_progress_seed.lua &
../build/S4 -a "3 100000 $RANDOM_SEED -s" -t 32 metasurface_progress_seed.lua &
../build/S4 -a "4 100000 $RANDOM_SEED -s" -t 32 metasurface_progress_seed.lua &

# Wait for all background tasks to finish
wait

echo "All tasks completed."

