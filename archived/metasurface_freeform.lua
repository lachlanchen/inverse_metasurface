--------------------------------------------------------------------------------
-- metasurface_polygon.lua
--
-- For each frequency in [0.3..1.0], we:
--   1) Generate random (n,k).
--   2) Create a brand-new S4 simulation.
--   3) Define a "RandSilicon" layer with a free-form polygon shape.
--   4) Output Reflection (R) and Transmission (T).
--------------------------------------------------------------------------------

-- Optional: for reproducible randomness
-- math.randomseed(12345)

-- Helper: random n,k in some range
function random_nk()
    -- e.g. n in [1,3], k in [0,0.2]
    local n = 1.0 + 2.0*math.random()
    local k = 0.2 * math.random()
    return n, k
end

for freq = 0.3, 1.0, 0.1 do
    -- 1) Generate random (n,k)
    local n, k = random_nk()

    -- 2) Convert (n + i k) -> epsilon
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k

    -- Build new S4 simulation for each frequency
    S = S4.NewSimulation()

    -- Lattice, Fourier orders
    S:SetLattice({1,0}, {0,1})
    S:SetNumG(40)

    -- Define materials
    S:AddMaterial("Vacuum",       {1,0})
    S:AddMaterial("RandSilicon",  {eps_real, eps_imag})

    -- Layers
    S:AddLayer("AirAbove",   0,          "Vacuum")
    S:AddLayer("MetaLayer",  0.5,        "RandSilicon")
    S:AddLayerCopy("AirBelow", 0,        "AirAbove")

    -- Paint a free-form polygon in 'MetaLayer' filled with Vacuum 
    -- (carving out the polygon region). Or invert it, depending on your design.
    -- For instance, let's carve out a "Vacuum polygon" from RandSilicon background.
    local polygon_vertices = {
      0.3,  0.2,    -- (x1,y1)
      0.3,  0.3,    -- (x2,y2)
     -0.3,  0.3,    -- ...
     -0.3, -0.3,
      0.3, -0.3,
      0.3, -0.2,
     -0.2, -0.2,
     -0.2,  0.2
    }

    S:SetLayerPatternPolygon(
        "MetaLayer",
        "Vacuum",           -- set this shape to vacuum
        {0,0},             -- center
        0,                 -- tilt angle in degrees
        polygon_vertices   -- table of x,y coords
    )

    -- Incidence
    S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
    S:SetFrequency(freq)

    -- Reflection, Transmission
    local fwdAbove, backAbove = S:GetPoyntingFlux("AirAbove", 0)
    local R = backAbove
    local fwdBelow, backBelow = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdBelow

    -- Output
    print(string.format(
      "freq=%.2f | n=%.2f, k=%.2f | R=%.4f, T=%.4f, R+T=%.4f",
       freq, n, k, R, T, R+T
    ))
end

