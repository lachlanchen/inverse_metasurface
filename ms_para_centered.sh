#!/usr/bin/env bash

# Run metasurface_progress.lua for (N_quarter = 1..4) each with num_shapes=1000,
# running all tasks in parallel and storing results (no verbose printing).

../build/S4 -a "1 10000 -s" -t 32 metasurface_progress_centered.lua &
../build/S4 -a "2 10000 -s" -t 32 metasurface_progress_centered.lua &
../build/S4 -a "3 10000 -s" -t 32 metasurface_progress_centered.lua &
../build/S4 -a "4 10000 -s" -t 32 metasurface_progress_centered.lua &

# Wait for all background tasks to finish
wait

echo "All tasks completed."

