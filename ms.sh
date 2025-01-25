#!/usr/bin/env bash

# Default values
NUM_SHAPES=10000
RANDOM_SEED=88888

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -ns|--numshapes)
            NUM_SHAPES="$2"
            shift
            ;;
        -r|--seed)
            RANDOM_SEED="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [-ns <NUM_SHAPES>] [-r <SEED>]"
            exit 1
            ;;
    esac
    shift
done

echo "Using number of shapes: $NUM_SHAPES"
echo "Using random seed: $RANDOM_SEED"

# This script runs metasurface_seed.lua for N_quarter = 1..4,
# with the chosen num_shapes, random_seed, and store_data = true (-s).
# All tasks are run in the background in parallel; then we wait.

../build/S4 -a "1 $NUM_SHAPES $RANDOM_SEED -s" -t 32 metasurface_seed.lua &
../build/S4 -a "2 $NUM_SHAPES $RANDOM_SEED -s" -t 32 metasurface_seed.lua &
../build/S4 -a "3 $NUM_SHAPES $RANDOM_SEED -s" -t 32 metasurface_seed.lua &
../build/S4 -a "4 $NUM_SHAPES $RANDOM_SEED -s" -t 32 metasurface_seed.lua &

wait

echo "All tasks completed."

