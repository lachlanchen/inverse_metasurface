--------------------------------------------------------------------------------
-- metasurface_hollow_polygon.lua
--
-- Goal:
--  - For each frequency from 0.3 to 1.0 in steps of 0.1:
--    1) Generate random (n,k) for the "RandomMaterial".
--    2) Create a NEW S4 simulation from scratch.
--    3) Paint a hollow polygon ring:
--       - Outer polygon: "RandomMaterial"
--       - Inner polygon: "Vacuum" (carving out the inside)
--    4) Compute Reflection (R) & Transmission (T), then print them.
--
-- This demonstrates how to create a "hollow" shape in S4.
--------------------------------------------------------------------------------

-- Optional: for reproducible random numbers
-- math.randomseed(12345)

-------------------------------------------------------------------------------
-- Function: generate_random_polygon
--    Creates a polygon with N vertices by sampling angle from 0..2π and 
--    giving each point a random radius near base_radius ± rand_amt.
--
-- Returns: flat table {x1, y1, x2, y2, ..., xN, yN}.
-------------------------------------------------------------------------------
function generate_random_polygon(N, base_radius, rand_amt)
    local verts = {}
    local two_pi = 2 * math.pi
    for i = 1, N do
        local angle = two_pi * (i - 1) / N
        -- random radial offset around base_radius
        local r = base_radius + rand_amt * (2*math.random() - 1)
        local x = r * math.cos(angle)
        local y = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end
    return verts
end

-------------------------------------------------------------------------------
-- Function: random_nk
--    Returns random n in [1, 3] and k in [0, 0.2].
-------------------------------------------------------------------------------
function random_nk()
    local n = 1.0 + 2.0 * math.random()  -- random in [1..3]
    local k = 0.2 * math.random()        -- random in [0..0.2]
    return n, k
end

-------------------------------------------------------------------------------
-- Frequency loop
-------------------------------------------------------------------------------
for freq = 0.3, 1.0, 0.1 do
    -- 1) Generate random (n,k)
    local n, k = random_nk()

    -- 2) Convert to complex permittivity
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k

    -- 3) Create a new S4 simulation
    S = S4.NewSimulation()

    -- 4) Lattice and Fourier orders
    S:SetLattice({1,0}, {0,1})
    S:SetNumG(40)

    -- 5) Define materials
    S:AddMaterial("Vacuum",        {1,0})
    S:AddMaterial("RandomMaterial",{eps_real, eps_imag})

    -- 6) Define layers
    S:AddLayer("AirAbove",   0,             "Vacuum")         -- semi-infinite
    S:AddLayer("MetaLayer",  0.5,           "Vacuum")         -- background=Vacuum
    S:AddLayerCopy("AirBelow", 0,           "AirAbove")       -- semi-infinite

    -- 7) Generate outer and inner polygons for the ring.
    --    For example:
    --       outer polygon ~ radius=0.3 ± 0.03
    --       inner polygon ~ radius=0.15 ± 0.02
    local outerN = 100  -- number of vertices in outer polygon
    local innerN = 100  -- number of vertices in inner polygon

    local outer_poly = generate_random_polygon(outerN, 0.30, 0.03)
    local inner_poly = generate_random_polygon(innerN, 0.15, 0.02)

    -- Paint the OUTER polygon with "RandomMaterial"
    S:SetLayerPatternPolygon(
        "MetaLayer",        -- layer
        "RandomMaterial",   -- material
        {0,0},              -- center
        0,                  -- tilt angle
        outer_poly          -- vertex table
    )

    -- Then paint the INNER polygon with "Vacuum" to carve it out
    S:SetLayerPatternPolygon(
        "MetaLayer",
        "Vacuum",
        {0,0},
        0,
        inner_poly
    )

    -- 8) Plane-wave excitation
    S:SetExcitationPlanewave(
        {0,0},   -- normal incidence
        {1,0},   -- s-pol
        {0,0}    -- p-pol
    )

    -- 9) Set the frequency
    S:SetFrequency(freq)

    -- 10) Compute reflection & transmission
    local fwdAbove, backAbove = S:GetPoyntingFlux("AirAbove", 0)
    local R = backAbove
    local fwdBelow, backBelow = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdBelow

    -- Print results
    print(string.format(
        "freq=%.2f | n=%.2f, k=%.2f | R=%.4f, T=%.4f, R+T=%.4f",
        freq, n, k, R, T, R+T
    ))
end

