S = S4.NewSimulation()
S:SetLattice({1,0}, {0,1})   -- 2D square lattice, period = 1 in x and y
S:SetNumG(80)

-- Materials
-- 12 is the electrostatic constant, the square root of which is complex refractive index
S:AddMaterial('Silicon', {12,0}) -- it doesn't the wavelength dependence and the extinction coeff
S:AddMaterial('Vacuum',  {1,0})

-- Layers
S:AddLayer('AirAbove',  0,     'Vacuum')
S:AddLayer('MetaLayer', 0.5,   'Silicon')  -- thickness = 0.5
S:SetLayerPatternCircle('MetaLayer', 'Vacuum', {0,0}, 0.2)  -- carve out vacuum cylinder
S:AddLayerCopy('AirBelow', 0,  'AirAbove')

-- Excitation
S:SetExcitationPlanewave({0,0}, {1,0}, {0,0}) -- normal incidence, s-pol

-- Frequency sweep
for freq=1, 2.0, 0.01 do
    S:SetFrequency(freq)
    
    -- Get reflection (backward flux in AirAbove)
    refl_forw, refl_back = S:GetPoyntingFlux('AirAbove', 0)
    local R = refl_back
    
    -- Get transmission (forward flux in AirBelow)
    tran_forw, tran_back = S:GetPoyntingFlux('AirBelow', 0)
    local T = tran_forw
    
    print(freq, R, T, -R+T)
end

