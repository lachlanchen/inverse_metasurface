--------------------------------------------------------------------------------
-- metasurface4.lua
--
-- For each frequency in a sweep, we:
--   - Generate random (n,k)
--   - Build a new S4 simulation from scratch
--   - Use "RandSilicon" as the metasurface material
--   - A circular hole of vacuum in that layer
--   - Calculate reflection R and transmission T, then print them
--
-- NOTE: If your geometry is in 'arbitrary units', then 0.3..1.0 are
--       just dimensionless frequencies (e.g., freq = 1/lambda).
--------------------------------------------------------------------------------

-- Optional: Set a random seed for reproducibility
-- math.randomseed(12345)

-- Helper function to produce random (n,k)
-- For demonstration, n is in [1.0, 3.0], k is in [0.0, 0.2]
function random_nk()
    local n = 1.0 + 2.0 * math.random()
    local k = 0.2 * math.random()
    return n, k
end

-- Frequency sweep
for freq = 0.3, 1.0, 0.1 do

    -- 1) Generate random refractive index for this frequency
    local n, k = random_nk()

    -- 2) Convert (n, k) to permittivity: (n + i k)^2 = (n^2 - k^2) + i(2 n k)
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k

    ----------------------------------------------------------------------------
    -- 3) Create a NEW S4 simulation for this frequency
    ----------------------------------------------------------------------------
    S = S4.NewSimulation()

    -- 4) Set 2D lattice (period = 1 × 1) and number of Fourier orders
    S:SetLattice({1,0}, {0,1})
    S:SetNumG(50)

    -- 5) Define materials. 
    --    We MUST define "Vacuum" before referencing it in layers.
    S:AddMaterial("Vacuum",      {1,0})
    S:AddMaterial("RandSilicon", {eps_real, eps_imag})

    -- 6) Define the layer stack
    --    - AirAbove (semi-infinite)
    --    - MetaLayer with thickness=0.5, background = RandSilicon
    --    - AirBelow (another semi-infinite)
    S:AddLayer("AirAbove",   0,               "Vacuum")
    S:AddLayer("MetaLayer",  0.5,            "RandSilicon")
    S:AddLayerCopy("AirBelow", 0,            "AirAbove")

    -- 7) Within "MetaLayer", carve out a circular hole filled with Vacuum.
    --    (radius=0.2, centered at x=0, y=0)
    S:SetLayerPatternCircle(
        "MetaLayer",   -- which layer
        "Vacuum",      -- fill material
        {0, 0},        -- center
        0.2            -- radius
    )

    -- 8) Plane-wave excitation at normal incidence, s-polarized
    S:SetExcitationPlanewave(
        {0,0},   -- incidence angles (phi=0, theta=0)
        {1,0},   -- s-pol amplitude=1
        {0,0}    -- p-pol amplitude=0
    )

    -- 9) Set the frequency
    S:SetFrequency(freq)

    -- 10) Get reflection (R) from top side, transmission (T) from bottom side
    local fwdAbove, backAbove = S:GetPoyntingFlux("AirAbove", 0)
    local R = backAbove  -- reflection is backward flux in "AirAbove"
    local fwdBelow, backBelow = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdBelow   -- transmission is forward flux in "AirBelow"

    ----------------------------------------------------------------------------
    -- Print results for this frequency
    ----------------------------------------------------------------------------
    print(string.format(
        "freq=%.2f | n=%.2f, k=%.2f | R=%.4f, T=%.4f, R+T=%.4f",
         freq, n, k, R, T, R+T
    ))
end

