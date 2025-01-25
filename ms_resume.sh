#!/usr/bin/env bash

# Default values
NUM_SHAPES=10000
RANDOM_SEED=88888
PREFIX=""
NUM_G=40

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
        -p|--prefix)
            PREFIX="$2"
            shift
            ;;
        -g|--numg)
            NUM_G="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [-ns <NUM_SHAPES>] [-r <SEED>] [-p <PREFIX>] [-g <NUM_G>]"
            exit 1
            ;;
    esac
    shift
done

echo "Using number of shapes: $NUM_SHAPES"
echo "Using random seed: $RANDOM_SEED"
if [ -z "$PREFIX" ]; then
    echo "No prefix set => new run with a datetime-based prefix (no resume)."
else
    echo "Using prefix: '$PREFIX'"
    echo "If an existing output file matches this prefix, we'll resume. Otherwise, a new run."
fi
echo "Using NumG: $NUM_G"

# We run metasurface_seed_resume.lua for N_quarter = 1..4 in parallel.
# Pass the arguments in the order:
#   1) N_quarter
#   2) NUM_SHAPES
#   3) RANDOM_SEED
#   4) PREFIX
#   5) NUM_G
# Then a final "-s" to store results. 
# (You can add "-v" as well, if desired, but this example always uses "-s".)

for NQ in 1 2 3 4
do
    ../build/S4 \
        -a "$NQ $NUM_SHAPES $RANDOM_SEED $PREFIX $NUM_G -s" \
        -t 32 \
        metasurface_seed_resume.lua &
done

wait

echo "All tasks completed."

