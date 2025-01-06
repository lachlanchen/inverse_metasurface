------------------------------------------------------------------------------
-- metasurface4.lua
-- Demonstration of:
--   - Nested loops (outer iteration and inner wavelength sweep)
--   - Randomly generated refractive index (n,k) each wavelength
--   - Overwriting one "RandomMaterial" in each loop iteration
--   - Simple 2D metasurface with a circle in a single layer
--   - Reflection/Transmission (Poynting flux)
------------------------------------------------------------------------------

-- 1) Create a new simulation
S = S4.NewSimulation()

-- 2) Define the 2D lattice (square, period=1 in S4 units).
S:SetLattice({1,0}, {0,1})

-- 3) Set number of Fourier orders
S:SetNumG(80)

-- 4) Define known material(s) before usage:
S:AddMaterial("Vacuum", {1,0})  -- Always define "Vacuum" first

-- 5) We'll add 'RandomMaterial' in the loop, so just note it here.

-- 6) Define a simple layer stack: air above, 0.5-thick layer, air below
S:AddLayer('AirAbove',  0,       'Vacuum')        -- thickness=0 => semi-infinite
S:AddLayer('MetaLayer', 0.5,     'Vacuum')        -- background of this layer
S:AddLayerCopy('AirBelow', 0,    'AirAbove')      -- another semi-infinite

-- 7) For demonstration, place a circle in 'MetaLayer'. We'll *overwrite* 
--    that circle's material each iteration in the loops below.
--
--    Initially, let's set it to vacuum (no effect). 
--    In the loops we re-paint it with "RandomMaterial".
S:SetLayerPatternCircle(
    'MetaLayer',
    'Vacuum',  -- temporary
    {0,0},     -- center
    0.2        -- radius
)

-- 8) Set up the plane-wave excitation at normal incidence, s-polarized.
S:SetExcitationPlanewave(
    {0,0},  -- incidence angles (phi=0, theta=0 => normal incidence)
    {1,0},  -- s-polarization amplitude=1
    {0,0}   -- p-polarization amplitude=0
)

------------------------------------------------------------------------------
-- HELPER: Convert wavelength (nm) to frequency in S4 units
-- Assuming your S4 geometry is in nm as well (period=1 means 1 nm).
-- If geometry is in µm, you'd do freq = 1/(lambda_nm/1000).
------------------------------------------------------------------------------
function nm_to_freq(lambda_nm)
    -- If 1 S4 unit = 1 nm, freq = 1 / lambda_nm
    return 1.0 / lambda_nm
end

------------------------------------------------------------------------------
-- HELPER: Return random (n, k). For demonstration only!
------------------------------------------------------------------------------
function random_nk()
    -- e.g., n in [1.0, 3.0], k in [0.0, 0.2]
    local n = 1.0 + 2.0 * math.random()  -- random in [1, 3]
    local k = 0.2 * math.random()        -- random in [0, 0.2]
    return n, k
end

-- Optional: set a seed for reproducible random numbers
-- math.randomseed(12345)

------------------------------------------------------------------------------
-- Outer loop (e.g. 3 scenarios)
------------------------------------------------------------------------------
for iteration = 1,3 do
    print("===================================================")
    print("Iteration #:\t", iteration)
    print("===================================================")
    
    ----------------------------------------------------------------------------
    -- Inner loop: wavelength from 400 nm to 700 nm in steps of 100 nm
    ----------------------------------------------------------------------------
    for lambda_nm = 400,700,100 do
        -- 1) Generate random n,k
        local n, k = random_nk()

        -- 2) Convert to permittivity: (n + i k)^2 = (n^2 - k^2) + i(2nk)
        local eps_real = n*n - k*k
        local eps_imag = 2*n*k

        -- 3) Add/overwrite 'RandomMaterial' with new epsilon
        S:AddMaterial("RandomMaterial", {eps_real, eps_imag})

        -- 4) Paint the circle region in 'MetaLayer' with 'RandomMaterial'
        S:SetLayerPatternCircle(
            'MetaLayer',
            'RandomMaterial',
            {0,0},
            0.2
        )

        -- 5) Convert wavelength to frequency and set S4 frequency
        local freq = nm_to_freq(lambda_nm)
        S:SetFrequency(freq)

        -- 6) Compute reflection (R) and transmission (T)
        local fwdA, backA = S:GetPoyntingFlux('AirAbove', 0)
        local R = backA   -- reflection is backward flux in AirAbove
        local fwdB, backB = S:GetPoyntingFlux('AirBelow', 0)
        local T = fwdB    -- transmission is forward flux in AirBelow

        -- 7) Print or store results
        print(string.format(
            " iteration=%d, lambda=%dnm => n=%.2f, k=%.2f;  freq=%.4f;  R=%.4f; T=%.4f; R+T=%.4f",
            iteration, lambda_nm, n, k, freq, R, T, R+T
        ))
    end  -- end wavelength loop
end  -- end iteration loop

