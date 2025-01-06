--------------------------------------------------------------------------------
-- metasurface_c4.lua
--
-- This script:
--   1) Generates a C4-symmetric hollow polygon (ring) by creating a polygon
--      in one quadrant and rotating it to other quadrants.
--   2) Saves the outer and inner polygons to the "shapes/" folder.
--   3) Loops over a range of frequencies, and for each frequency:
--        a) Randomly generates refractive index (n, k).
--        b) Sets up an S4 simulation with the fixed C4-symmetric shape.
--        c) Computes reflection (R) and transmission (T).
--        d) Prints the results.
--------------------------------------------------------------------------------

-- Optional: Set a random seed for reproducibility
-- math.randomseed(12345)

--------------------------------------------------------------------------------
-- Helper Function: Rotate a point (x, y) by angle_degrees around the origin
--------------------------------------------------------------------------------
function rotate_point(x, y, angle_degrees)
    local angle_rad = math.rad(angle_degrees)
    local cos_a = math.cos(angle_rad)
    local sin_a = math.sin(angle_rad)
    local x_new = x * cos_a - y * sin_a
    local y_new = x * sin_a + y * cos_a
    return x_new, y_new
end

--------------------------------------------------------------------------------
-- Helper Function: Generate a C4-symmetric polygon by creating points in one
--                   quadrant and rotating them to other quadrants.
--
-- Parameters:
--   N_per_quadrant - Number of points per quadrant
--   base_radius    - Average radius of the polygon
--   rand_amt       - Maximum random deviation from base_radius
--
-- Returns:
--   verts - Flat table {x1, y1, x2, y2, ..., xN, yN}
--------------------------------------------------------------------------------
function generate_c4_polygon(N_per_quadrant, base_radius, rand_amt)
    local verts = {}
    local two_pi = 2 * math.pi
    local quarter_pi = math.pi / 2

    for q = 0, 3 do  -- Four quadrants: 0, 90, 180, 270 degrees
        for i = 1, N_per_quadrant do
            local angle = (quarter_pi * q) + (two_pi * i) / (4 * N_per_quadrant)
            local r = base_radius + rand_amt * (2 * math.random() - 1)
            local x = r * math.cos(angle)
            local y = r * math.sin(angle)
            table.insert(verts, x)
            table.insert(verts, y)
        end
    end

    return verts
end

--------------------------------------------------------------------------------
-- Helper Function: Save a polygon to a text file
--
-- Parameters:
--   filename - Path to the output file
--   polygon  - Flat table {x1, y1, x2, y2, ..., xN, yN}
--------------------------------------------------------------------------------
function save_polygon_to_file(filename, polygon)
    local file = io.open(filename, "w")
    if not file then
        error("Could not open '" .. filename .. "' for writing. " ..
              "Ensure the 'shapes/' directory exists and is writable.")
    end
    for i = 1, #polygon, 2 do
        local x = polygon[i]
        local y = polygon[i+1]
        file:write(string.format("%.6f,%.6f\n", x, y))
    end
    file:close()
end

--------------------------------------------------------------------------------
-- Helper Function: Generate a random refractive index (n, k)
--
-- Returns:
--   n - Refractive index in [1.0, 3.0]
--   k - Extinction coefficient in [0.0, 0.2]
--------------------------------------------------------------------------------
function random_nk()
    local n = 1.0 + 2.0 * math.random()  -- [1.0, 3.0]
    local k = 0.2 * math.random()        -- [0.0, 0.2]
    return n, k
end

--------------------------------------------------------------------------------
-- MAIN SCRIPT
--------------------------------------------------------------------------------

-- 1) Define the C4-symmetric hollow polygon once

-- Parameters for the outer and inner polygons
local N_quadrant = 100        -- Number of points per quadrant for smoothness
local outer_base_radius = 0.30
local outer_rand_amt = 0.03
local inner_base_radius = 0.15
local inner_rand_amt = 0.02

-- Generate outer and inner polygons
local outer_poly = generate_c4_polygon(N_quadrant, outer_base_radius, outer_rand_amt)
local inner_poly = generate_c4_polygon(N_quadrant, inner_base_radius, inner_rand_amt)

-- 2) Save the polygons to the "shapes/" folder
--    Ensure the "shapes/" folder exists before running the script
local timestamp = os.date("%Y%m%d_%H%M%S")
local outer_filename = string.format("shapes/fixed_outer_%s.txt", timestamp)
local inner_filename = string.format("shapes/fixed_inner_%s.txt", timestamp)

save_polygon_to_file(outer_filename, outer_poly)
save_polygon_to_file(inner_filename, inner_poly)

-- 3) Loop over frequencies and perform simulations
--    Frequencies from 0.3 to 1.0 in steps of 0.1
for freq = 0.3, 1.0, 0.1 do
    -- a) Generate random (n, k)
    local n, k = random_nk()
    local eps_real = n * n - k * k
    local eps_imag = 2 * n * k

    -- b) Create a new S4 simulation
    local S = S4.NewSimulation()

    -- c) Define the lattice and Fourier orders
    S:SetLattice({1, 0}, {0, 1})      -- 2D square lattice with period = 1
    S:SetNumG(40)                     -- Number of Fourier orders (adjust for accuracy)

    -- d) Define materials
    S:AddMaterial("Vacuum", {1, 0})                  -- Vacuum material
    S:AddMaterial("RandomMaterial", {eps_real, eps_imag})  -- Randomly defined material

    -- e) Define the layer stack
    S:AddLayer("AirAbove", 0, "Vacuum")     -- Top semi-infinite layer
    S:AddLayer("MetaLayer", 0.5, "Vacuum")  -- Metasurface layer with background Vacuum
    S:AddLayerCopy("AirBelow", 0, "AirAbove") -- Bottom semi-infinite layer

    -- f) Paint the outer polygon with "RandomMaterial"
    S:SetLayerPatternPolygon(
        "MetaLayer",        -- Layer name
        "RandomMaterial",   -- Material to paint
        {0, 0},             -- Center position (x, y)
        0,                  -- Rotation angle in degrees
        outer_poly          -- Polygon vertices
    )

    -- g) Paint the inner polygon with "Vacuum" to create the hollow (carve out the center)
    S:SetLayerPatternPolygon(
        "MetaLayer",  -- Layer name
        "Vacuum",     -- Material to paint (carve out)
        {0, 0},       -- Center position (x, y)
        0,            -- Rotation angle in degrees
        inner_poly    -- Polygon vertices
    )

    -- h) Set up the plane-wave excitation (normal incidence, s-polarized)
    S:SetExcitationPlanewave(
        {0, 0},   -- Incidence angles (phi=0, theta=0 => normal incidence)
        {1, 0},   -- s-polarization: amplitude=1, phase=0 degrees
        {0, 0}    -- p-polarization: amplitude=0, phase=0 degrees
    )

    -- i) Set the frequency
    S:SetFrequency(freq)

    -- j) Compute reflection (R) and transmission (T)
    local fwdAbove, backAbove = S:GetPoyntingFlux("AirAbove", 0)
    local R = backAbove      -- Reflection: backward flux in "AirAbove"
    local fwdBelow, backBelow = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdBelow       -- Transmission: forward flux in "AirBelow"

    -- k) Print the results
    print(string.format("freq=%.2f  n=%.2f  k=%.2f  R=%.4f  T=%.4f  R+T=%.4f",
        freq, n, k, R, T, R + T))
end

