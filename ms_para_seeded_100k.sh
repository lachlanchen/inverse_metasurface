#!/usr/bin/env bash

# This script runs metasurface_progress_seed.lua for N_quarter = 1..4,
# each with num_shapes=100000, random_seed=88888, and store_data = true (-s).
# All tasks are run in the background in parallel. Then we wait for all to complete.

../build/S4 -a "1 100000 88888 -s" -t 32 metasurface_progress_seed.lua &
../build/S4 -a "2 100000 88888 -s" -t 32 metasurface_progress_seed.lua &
../build/S4 -a "3 100000 88888 -s" -t 32 metasurface_progress_seed.lua &
../build/S4 -a "4 100000 88888 -s" -t 32 metasurface_progress_seed.lua &

# Wait for all background tasks to finish
wait

echo "All tasks completed."

