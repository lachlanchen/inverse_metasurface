--------------------------------------------------------------------------------
-- metasurface_freeform.lua
--
-- In each frequency loop:
--   1) Generate random (n,k) for the "RandomMaterial".
--   2) Generate a random ~100-point polygon by varying the radius slightly.
--   3) Create a new S4 simulation.
--   4) Paint a "free-form" shape in the metasurface layer (background is 'RandomMaterial',
--      or we invert it so that the shape is vacuum and background is 'RandomMaterial').
--   5) Compute reflection and transmission, then print them.
--------------------------------------------------------------------------------

-- Optional: for reproducible random numbers
math.randomseed(12345)

-- Function to generate a smooth random polygon with ~N points
--   base_radius: average radius
--   rand_amt: max random deviation from base_radius
-- Return value: a flat table {x1, y1, x2, y2, ..., xN, yN}.
function generate_random_polygon(N, base_radius, rand_amt)
    local verts = {}
    local two_pi = 2*math.pi
    for i=1,N do
        -- angle from 0..2pi
        local angle = two_pi * (i-1)/N
        -- random radial offset around base_radius
        local r = base_radius + rand_amt*(2*math.random() - 1)
        -- convert to (x,y)
        local x = r * math.cos(angle)
        local y = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end
    return verts
end

-- A helper for random n,k
function random_nk()
    -- Example: n in [1,3], k in [0,0.2]
    local n = 1.0 + 2.0*math.random()
    local k = 0.2 * math.random()
    return n, k
end

-- Frequency sweep
for freq = 0.3, 1.0, 0.1 do

    -- 1) Generate random (n, k)
    local n, k = random_nk()
    -- 2) Convert (n + i k) -> (eps_real, eps_imag)
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k

    -- 3) Generate the random polygon with ~100 points
    --    We'll choose base_radius=0.3, random deviation=0.05 (tweak as desired)
    local polygon_points = generate_random_polygon(100, 0.3, 0.05)

    -- 4) Create a new S4 simulation
    S = S4.NewSimulation()
    S:SetLattice({1,0}, {0,1})
    S:SetNumG(40)

    -- 5) Define materials
    S:AddMaterial("Vacuum",         {1,0})
    S:AddMaterial("RandomMaterial", {eps_real, eps_imag})

    -- 6) Define layer stack
    --    We treat 'RandomMaterial' as the background of the meta layer
    S:AddLayer("AirAbove",   0,           "Vacuum")
    S:AddLayer("MetaLayer",  0.5,         "RandomMaterial")
    S:AddLayerCopy("AirBelow", 0,         "AirAbove")

    -- 7) "Carve out" a free-form vacuum shape in the RandomMaterial background
    S:SetLayerPatternPolygon(
        "MetaLayer",          -- which layer
        "Vacuum",            -- material in the polygon
        {0,0},               -- center of polygon
        0,                   -- tilt angle in degrees
        polygon_points       -- table of (x,y) pairs
    )

    -- 8) Incidence: normal incidence, s-pol
    S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})

    -- 9) Set frequency
    S:SetFrequency(freq)

    -- 10) Compute R, T
    local fwdA, backA = S:GetPoyntingFlux("AirAbove", 0)
    local R = backA
    local fwdB, backB = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdB

    -- Print results
    print(string.format(
      "freq=%.2f | n=%.2f, k=%.2f | R=%.4f, T=%.4f, R+T=%.4f",
       freq, n, k, R, T, R+T
    ))
end

