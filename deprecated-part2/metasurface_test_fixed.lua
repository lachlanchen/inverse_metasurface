--------------------------------------------------------------------------------
-- metasurface_test_fixed.lua
-- Example script that uses a single, fixed polygon (no random offsets).
-- Always uses the same shape_str and c_value, ignoring external arguments.
--------------------------------------------------------------------------------

local shape_str = [[
0.91920647499265051,0.61119817068807369;
0.38880182931192636,0.91920647499265051;
0.080793525007349432,0.38880182931192642;
0.61119817068807358,0.080793525007349432
]]

-- Hard-coded c_value for the partial_crys_CX.csv file:
local c_value = 0.0   -- e.g. 0.0, or 0.5, or 0.9, etc.

------------------------------------------------------------------------------
-- Parse the shape_str into numeric coordinates
------------------------------------------------------------------------------
local function parse_polygon(str)
  local verts = {}
  for pair in str:gmatch("[^;]+") do
    local x,y = pair:match("([^,]+),([^,]+)")
    if x and y then
      local xx = tonumber(x)
      local yy = tonumber(y)
      table.insert(verts, xx)
      table.insert(verts, yy)
    end
  end
  return verts
end

local outer_polygon = parse_polygon(shape_str)

------------------------------------------------------------------------------
-- Load the partial_crys_C{c_value}.csv data
------------------------------------------------------------------------------
local function load_material_data(cval)
  -- For example, c_value = 0.0 => partial_crys_C0.0.csv
  -- c_value = 0.5 => partial_crys_C0.5.csv, etc.
  local fname = string.format("partial_crys_data/partial_crys_C%.1f.csv", cval)
  local f = io.open(fname, "r")
  if not f then
    error("Could not open file: "..fname)
  end

  local data = {}
  local header = true
  for line in f:lines() do
    if header then
      header = false
    else
      local lam_str, n_str, k_str = line:match("([^,]+),([^,]+),([^,]+)")
      if lam_str and n_str and k_str then
        local lam = tonumber(lam_str)
        local nr  = tonumber(n_str)
        local kr  = tonumber(k_str)
        if lam and nr and kr then
          table.insert(data, {lambda=lam, n=nr, k=kr})
        end
      end
    end
  end
  f:close()
  return data, fname
end

local mat_data, mat_fname = load_material_data(c_value)

------------------------------------------------------------------------------
-- Optionally store results to CSV
------------------------------------------------------------------------------
local store_data = true  -- change to false if you don't want output
local out_file
if store_data then
  local dt = os.date("%Y%m%d_%H%M%S")
  local outname = string.format("results/fixed_shape_c%.1f_%s.csv", c_value, dt)
  os.execute("mkdir -p results")
  out_file = io.open(outname, "w")
  out_file:write("c_value,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T\n")
  print("Saved to "..outname)
end

------------------------------------------------------------------------------
-- Run S4 for each row in mat_data
------------------------------------------------------------------------------
local S = nil
for i, row in ipairs(mat_data) do
  local lam = row.lambda
  local nr  = row.n
  local kr  = row.k
  local freq = 1.0 / lam

  local eps_real = nr*nr - kr*kr
  local eps_imag = 2*nr*kr

  -- Build S4 simulation
  S = S4.NewSimulation()
  S:SetLattice({1,0}, {0,1})   -- 1×1 period
  S:SetNumG(40)

  local matname = string.format("MatC%.1f", c_value)
  S:AddMaterial("Vacuum", {1,0})
  S:AddMaterial(matname, {eps_real, eps_imag})

  -- Layers
  S:AddLayer("AirAbove",   0,   "Vacuum")
  S:AddLayer("MetaLayer",  0.5, "Vacuum")
  S:AddLayerCopy("AirBelow", 0, "AirAbove")

  -- Hard-coded polygon
  S:SetLayerPatternPolygon("MetaLayer", matname, {0,0}, 0, outer_polygon)

  -- Plane wave incidence
  S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
  S:SetFrequency(freq)

  local fwdA, backA = S:GetPoyntingFlux("AirAbove", 0)
  local fwdB, backB = S:GetPoyntingFlux("AirBelow", 0)

  local R = -backA
  local T =  fwdB

  local rplus = R + T

  -- Print results to terminal
  print(string.format(
    "[i=%d/%d], λ=%.4f um, (n=%.3f,k=%.3f) => R=%.4f, T=%.4f, R+T=%.4f",
    i,#mat_data, lam, nr, kr, R, T, rplus
  ))

  -- Optionally write to CSV
  if out_file then
    out_file:write(string.format("%.1f,%.6f,%.6f,%.5f,%.5f,%.6f,%.6f,%.6f\n",
      c_value, lam, freq, nr, kr, R, T, rplus))
  end
end

if out_file then
  out_file:close()
end

print("Completed fixed-shape simulation for c="..c_value.." using polygon="..shape_str)

