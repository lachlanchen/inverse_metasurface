--------------------------------------------------------------------------------
-- metasurface_hollow_polygon_save.lua
--
-- - For each frequency (0.3..1.0), we:
--   1) Generate random (n,k).
--   2) Create a NEW S4 simulation.
--   3) Make a ring by painting:
--        (a) OUTER polygon => "RandomMaterial"
--        (b) INNER polygon => "Vacuum" (carving out the center).
--   4) Save the polygon coordinates to "shapes/*.txt" for later plotting.
--   5) Compute Reflection/Transmission.
--   6) Print R, T to stdout.
--------------------------------------------------------------------------------

-- Optional: for reproducible random numbers
-- math.randomseed(12345)

-- Make sure you have a "shapes/" folder in the same directory, or adjust paths.

--------------------------------------------------------------------------------
--  Function: generate_random_polygon
--     Creates an N-point polygon by sampling angle [0..2π]
--     and adding random radial noise around base_radius ± rand_amt.
--
--  Returns: flat list {x1, y1, x2, y2, ..., xN, yN}.
--------------------------------------------------------------------------------
function generate_random_polygon(N, base_radius, rand_amt)
    local verts = {}
    local two_pi = 2*math.pi
    for i=1,N do
        local angle = two_pi * (i-1)/N
        local r = base_radius + rand_amt*(2*math.random() - 1)
        local x = r * math.cos(angle)
        local y = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end
    return verts
end

--------------------------------------------------------------------------------
--  Function: random_nk
--     Returns random n in [1..3] and k in [0..0.2] (example ranges).
--------------------------------------------------------------------------------
function random_nk()
    local n = 1.0 + 2.0 * math.random()
    local k = 0.2 * math.random()
    return n, k
end

--------------------------------------------------------------------------------
--  Function: save_polygon_to_file
--     Saves a polygon (flat list of x,y) to a text file (one vertex per line).
--------------------------------------------------------------------------------
function save_polygon_to_file(filename, polygon)
    local file = io.open(filename, "w")
    for i=1,#polygon,2 do
        local x = polygon[i]
        local y = polygon[i+1]
        file:write(string.format("%.6f,%.6f\n", x, y))
    end
    file:close()
end

--------------------------------------------------------------------------------
--  MAIN LOOP over frequencies
--------------------------------------------------------------------------------
for freq = 0.3, 1.0, 0.1 do
    -- 1) Generate random (n,k)
    local n, k = random_nk()
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k

    -- 2) Create a new S4 simulation
    S = S4.NewSimulation()
    S:SetLattice({1,0},{0,1})
    S:SetNumG(40)

    -- 3) Materials
    S:AddMaterial("Vacuum",         {1,0})
    S:AddMaterial("RandomMaterial", {eps_real, eps_imag})

    -- 4) Layers
    S:AddLayer("AirAbove",   0,             "Vacuum")
    S:AddLayer("MetaLayer",  0.5,           "Vacuum")  -- background is vacuum
    S:AddLayerCopy("AirBelow", 0,           "AirAbove")

    -- 5) Generate outer & inner polygons (ring).
    local outer_poly = generate_random_polygon(100, 0.30, 0.03)
    local inner_poly = generate_random_polygon(100, 0.15, 0.02)

    -- Paint the outer polygon with RandomMaterial
    S:SetLayerPatternPolygon(
        "MetaLayer",
        "RandomMaterial",
        {0,0},
        0,
        outer_poly
    )

    -- Paint the inner polygon with Vacuum to carve out the center
    S:SetLayerPatternPolygon(
        "MetaLayer",
        "Vacuum",
        {0,0},
        0,
        inner_poly
    )

    -- 6) Set up excitation
    S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
    S:SetFrequency(freq)

    -- 7) Reflection & Transmission
    local fwdA, backA = S:GetPoyntingFlux("AirAbove", 0)
    local R = backA
    local fwdB, backB = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdB

    -- 8) Print to stdout
    print(string.format("freq=%.2f  n=%.2f k=%.2f  R=%.4f T=%.4f  R+T=%.4f",
        freq, n, k, R, T, R+T))

    -- 9) Save the polygons to text files in shapes/ folder.
    --    We'll use freq and a timestamp in the filename.
    local stamp = os.date("%Y%m%d_%H%M%S")
    local outer_filename = string.format("shapes/outer_f%.2f_%s.txt", freq, stamp)
    local inner_filename = string.format("shapes/inner_f%.2f_%s.txt", freq, stamp)

    save_polygon_to_file(outer_filename, outer_poly)
    save_polygon_to_file(inner_filename, inner_poly)
end

