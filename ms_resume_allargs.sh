#!/usr/bin/env bash

# Default values
NUM_SHAPES=100000
RANDOM_SEED=88888
PREFIX=""
NUM_G=80
BASE_OUTER=0.25
RAND_OUTER=0.20

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
        -bo|--baseouter)
            BASE_OUTER="$2"
            shift
            ;;
        -ro|--randouter)
            RAND_OUTER="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [options]"
            echo "  -ns|--numshapes <INT>     (default 100000)"
            echo "  -r|--seed <INT>           (default 88888)"
            echo "  -p|--prefix <STRING>      (default empty => new run)"
            echo "  -g|--numg <INT>           (default 80)"
            echo "  -bo|--baseouter <FLOAT>   (default 0.25)"
            echo "  -ro|--randouter <FLOAT>   (default 0.20)"
            exit 1
            ;;
    esac
    shift
done

echo "Using number of shapes:   $NUM_SHAPES"
echo "Using random seed:        $RANDOM_SEED"
if [ -z "$PREFIX" ]; then
    echo "No prefix => new run with datetime."
else
    echo "Using prefix: '$PREFIX' => resume if file exists."
fi
echo "Using NumG:               $NUM_G"
echo "Using base_outer:         $BASE_OUTER"
echo "Using rand_outer:         $RAND_OUTER"

# This script runs "metasurface_allargs_resume.lua" for N_quarter in {1..4} in parallel.
# We pass the arguments in the order:
#   1) N_quarter
#   2) NUM_SHAPES
#   3) RANDOM_SEED
#   4) PREFIX
#   5) NUM_G
#   6) BASE_OUTER
#   7) RAND_OUTER
# Then we add "-s" to store data. (Add "-v" if you want verbose.)

for NQ in 1 2 3 4
do
    ../build/S4 \
      -a "$NQ $NUM_SHAPES $RANDOM_SEED $PREFIX $NUM_G $BASE_OUTER $RAND_OUTER -s" \
      -t 32 \
      metasurface_allargs_resume.lua &
done

wait

echo "All tasks completed."

