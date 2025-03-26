#!/usr/bin/env bash
# Defaults:
PREFIX=""
RANDOM_SEED=88888
NUM_G=80
BASE_OUTER=0.225  # Changed to 0.225 for NIR (range 0.05-0.4)
RAND_OUTER=0.175  # Changed to 0.175 for NIR (range 0.05-0.4)
NUM_SHAPES=100000
# Parse CLI arguments in a simple loop
# (Feel free to rename or reorder these options as you wish.)
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -p|--prefix)
            PREFIX="$2"
            shift
            ;;
        -r|--seed)
            RANDOM_SEED="$2"
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
        -ns|--numshapes)
            NUM_SHAPES="$2"
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            echo "Usage: $0 [options]"
            echo "  -p|--prefix <STRING>      (default empty => new run with datetime)"
            echo "  -r|--seed <INT>           (default 88888)"
            echo "  -g|--numg <INT>           (default 80)"
            echo "  -bo|--baseouter <FLOAT>   (default 0.225)"
            echo "  -ro|--randouter <FLOAT>   (default 0.175)"
            echo "  -ns|--numshapes <INT>     (default 100000)"
            exit 1
            ;;
    esac
    shift
done
# Print chosen values
echo "Using prefix:         '$PREFIX' (empty => new run with datetime)"
echo "Using random_seed:     $RANDOM_SEED"
echo "Using NumG:            $NUM_G"
echo "Using base_outer:      $BASE_OUTER"
echo "Using rand_outer:      $RAND_OUTER"
echo "Using num_shapes:      $NUM_SHAPES"
echo "We will loop N_quarter from 1..4 in parallel."
# For each N_quarter in 1..4, we pass the arguments in the order:
#   prefix random_seed num_g base_outer rand_outer N_quarter num_shapes
# Then add "-s" to enable storing data. (You can also add "-v" for verbose.)
for NQ in 1 2 3 4
do
    ../build/S4 \
      -a "$PREFIX $RANDOM_SEED $NUM_G $BASE_OUTER $RAND_OUTER $NQ $NUM_SHAPES -s" \
      -t 32 \
      metasurface_resume_random_state_nir.lua &
    # If you want verbose logs:
    # ../build/S4 \
    #   -a "$PREFIX $RANDOM_SEED $NUM_G $BASE_OUTER $RAND_OUTER $NQ $NUM_SHAPES -s -v" \
    #   -t 32 \
    #   metasurface_resume_random_state_nir.lua &
done
wait
echo "All tasks completed."
