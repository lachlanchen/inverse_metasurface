--------------------------------------------------------------------------------
-- metasurface_wave.lua
--
-- A minimal S4 script demonstrating a metasurface:
--   - 2D square lattice (period 1 x 1)
--   - A silicon layer (thickness = 0.5)
--   - Circular holes (vacuum) of radius 0.2 in that layer
--   - Normal incidence, s-polarization
--   - Frequency sweep from 0.3 to 1.0 in steps of 0.1
--------------------------------------------------------------------------------

-- 1) Create a new simulation
S = S4.NewSimulation()

-- 2) Set the 2D lattice vectors (square lattice of period = 1)
S:SetLattice({1,0}, {0,1})

-- 3) Set the number of Fourier orders
S:SetNumG(80)

-- 4) Define the materials
S:AddMaterial("Vacuum",  {1,0})   -- must define "Vacuum" before using it in layers
S:AddMaterial("Silicon", {12,0})  -- example: silicon with epsilon_real=12, epsilon_imag=0

-- 5) Define the layer stack (z-direction)
--    - AirAbove layer (semi-infinite)
--    - MetaLayer (the metasurface layer, thickness=0.5)
--    - AirBelow layer (semi-infinite)
S:AddLayer('AirAbove',  0,     'Vacuum')     -- thickness=0 => semi-infinite
S:AddLayer('MetaLayer', 0.5,   'Silicon')    -- thickness=0.5
S:AddLayerCopy('AirBelow', 0,  'AirAbove')   -- another semi-infinite region

-- 6) Define the metasurface pattern inside 'MetaLayer'
--    Paint vacuum circles of radius 0.2 in the silicon layer
S:SetLayerPatternCircle(
    'MetaLayer',   -- which layer
    'Vacuum',      -- material in the circle
    {0,0},         -- center (x=0, y=0) in the unit cell
    0.2            -- radius
)

-- 7) Set up plane-wave excitation
--    Normal incidence (0,0) with purely s-polarized amplitude=1
S:SetExcitationPlanewave(
    {0,0},   -- incidence angles (phi=0, theta=0 => normal incidence)
    {1,0},   -- s-polarization amplitude=1, phase=0
    {0,0}    -- p-polarization amplitude=0
)

-- 8) Sweep over frequencies and get reflection/transmission
for freq = 0.3, 1.0, 0.1 do
    -- Set the frequency in S4
    S:SetFrequency(freq)

    -- Reflection is the "backward" flux at the top layer
    local forwardAbove, backwardAbove = S:GetPoyntingFlux('AirAbove', 0)
    local Reflect = backwardAbove

    -- Transmission is the "forward" flux at the bottom layer
    local forwardBelow, backwardBelow = S:GetPoyntingFlux('AirBelow', 0)
    local Transmit = forwardBelow

    -- Print results
    print(string.format(
        "freq=%.3f  R=%.6f  T=%.6f  R+T=%.6f", 
         freq, Reflect, Transmit, Reflect+Transmit))
end

