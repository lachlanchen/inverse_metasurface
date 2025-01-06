#!/usr/bin/env bash

# Run metasurface_progress.lua for (N_quarter = 1..4) each with num_shapes=1000,
# storing results (no verbose printing).

../build/S4 -a "1 1000 -s" metasurface_progress.lua
../build/S4 -a "2 1000 -s" metasurface_progress.lua
../build/S4 -a "3 1000 -s" metasurface_progress.lua
../build/S4 -a "4 1000 -s" metasurface_progress.lua

