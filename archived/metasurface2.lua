S = S4.NewSimulation()
-- (1) Set up your geometry, layers, and so forth as usual
S:SetLattice({1,0},{0,1})
S:SetNumG(80)
S:AddLayer('AirAbove',  0, 'Vacuum')
S:AddLayer('Slab', 0.5, 'TempMaterial') -- We'll re-assign 'TempMaterial' each iteration
S:AddLayerCopy('AirBelow', 0, 'AirAbove')

S:AddMaterial('Vacuum', {1,0})
-- We will ADD the 'TempMaterial' below, but re-init it for each wavelength.

-- (2) Convert nm -> frequency (in S4 units).
--     E.g., if you treat 1 length unit = 1 micrometer,
--     then freq = 1/lambda_in_um.  If you treat 1 length unit = 1 nm, do freq=1/lambda_nm.
--     Alternatively, freq = c / lambda if you use physical SI units, but S4 is typically unitless.
function nm_to_freq(lambda_nm)
  local lambda_units = lambda_nm -- if your S4 geometry is in nm
  return 1.0 / lambda_units
end

-- (3) Define the dispersion function (just an example).
function get_epsilon(lambda_nm)
  -- Suppose you load from a table or have a known function
  local n = 1.5  -- example
  local k = 0.05 -- example
  local eps_real = n*n - k*k
  local eps_imag = 2*n*k
  return eps_real, eps_imag
end

-- (4) Sweep over wavelength
for lambda_nm = 400, 800, 20 do
    local eps_real, eps_imag = get_epsilon(lambda_nm)

    -- Re-init or re-add the "TempMaterial" with the updated permittivity
    S:AddMaterial('TempMaterial', { eps_real, eps_imag })

    -- We could also do a remove/add approach, but S4 overwrites automatically
    -- each time you call `S:AddMaterial(...)` with the same name.

    local freq = nm_to_freq(lambda_nm)
    S:SetFrequency(freq)

    -- (5) Solve or get flux
    local forward, backward = S:GetPoyntingFlux('AirBelow', 0)
    local T = forward  -- transmission
    -- Reflection is in 'AirAbove' backward flux
    local fwdA, backA = S:GetPoyntingFlux('AirAbove', 0)
    local R = backA  -- reflection

    print(lambda_nm, R, T, R+T)
end

