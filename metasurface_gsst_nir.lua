--------------------------------------------------------------------------------
-- metasurface_gsst_nir.lua
--
-- Modified version for NIR spectrum testing with GSST 
-- Example usage:
--   ../build/S4 -a "0.2,0.0;0.0,0.2;-0.2,0.0;0.0,-0.2 -c 0.5 -t 500 -v -s" metasurface_gsst_nir.lua
--------------------------------------------------------------------------------
------------------------------------------------------------------------------
-- 0) Argument Parsing
------------------------------------------------------------------------------
local shape_str  = nil   -- e.g. "0.2,0.0;0.0,0.2;-0.2,0.0;0.0,-0.2"
local c_value    = 0.0   -- crystallinity (0.0 = amorphous, 1.0 = crystalline)
local thickness  = 500   -- thickness in nm (default = 500nm)
local verbose    = false
local store_data = false
local arg_str = S4.arg or ""
if #arg_str == 0 then
    error("No arguments supplied via -a. Expected: \"<vertices_str> -c 0.5 -t 500 -v -s\".")
end

-- Split the arg string into tokens
local tokens = {}
for tok in arg_str:gmatch("%S+") do
    table.insert(tokens, tok)
end

-- Parse tokens to find shape_str, c_value, thickness, flags, etc.
local i = 1
while i <= #tokens do
    local t = tokens[i]
    if t == "-c" then
        -- Next token should be a number for c_value
        if (i+1) <= #tokens then
            local ctest = tonumber(tokens[i+1])
            if ctest then
                c_value = ctest
                i = i + 2
            else
                error("Expected a numeric value after -c, but got '"..(tokens[i+1] or "").."'")
            end
        else
            error("Expected a numeric value after -c, none provided.")
        end
    elseif t == "-t" then
        -- Next token should be a number for thickness
        if (i+1) <= #tokens then
            local ttest = tonumber(tokens[i+1])
            if ttest then
                thickness = ttest
                i = i + 2
            else
                error("Expected a numeric value after -t, but got '"..(tokens[i+1] or "").."'")
            end
        else
            error("Expected a numeric value after -t, none provided.")
        end
    elseif t == "-v" then
        verbose = true
        i = i + 1
    elseif t == "-s" then
        store_data = true
        i = i + 1
    else
        -- Assume it's the shape string
        if shape_str then
            -- We already have a shape_str; treat the new token as an error or just ignore
            error("Received multiple shape strings? Already have shape='"..shape_str.."', new token='"..t.."'")
        else
            shape_str = t
        end
        i = i + 1
    end
end

if not shape_str then
    error("No shape string provided. Expected something like '0.2,0.0;0.0,0.2;...' etc.")
end

------------------------------------------------------------------------------
-- 1) Construct the polygon from the shape_str
------------------------------------------------------------------------------
-- shape_str is like "x1,y1;x2,y2;x3,y3;..."
local function parse_polygon(str)
    local verts = {}
    for pair in str:gmatch("[^;]+") do
        local x,y = pair:match("([^,]+),([^,]+)")
        if x and y then
            local xx = tonumber(x)
            local yy = tonumber(y)
            if xx and yy then
                table.insert(verts, xx)
                table.insert(verts, yy)
            else
                error("Invalid numeric pair in shape string: '"..pair.."'")
            end
        else
            error("Could not parse 'x,y' from substring: '"..pair.."'")
        end
    end
    
    -- Apply C4 symmetry to create points in all quadrants
    local original_points = {}
    for i = 1, #verts, 2 do
        table.insert(original_points, {verts[i], verts[i+1]})
    end
    
    local all_verts = {}
    -- First quadrant (original points)
    for _, pt in ipairs(original_points) do
        table.insert(all_verts, pt[1])
        table.insert(all_verts, pt[2])
    end
    
    -- Second quadrant: (-y, x)
    for _, pt in ipairs(original_points) do
        table.insert(all_verts, -pt[2])
        table.insert(all_verts, pt[1])
    end
    
    -- Third quadrant: (-x, -y)
    for _, pt in ipairs(original_points) do
        table.insert(all_verts, -pt[1])
        table.insert(all_verts, -pt[2])
    end
    
    -- Fourth quadrant: (y, -x)
    for _, pt in ipairs(original_points) do
        table.insert(all_verts, pt[2])
        table.insert(all_verts, -pt[1])
    end
    
    if verbose then
        print("Generated a C4 symmetric polygon with " .. (#all_verts/2) .. " total vertices")
    end
    
    return all_verts
end
local outer_polygon = parse_polygon(shape_str)

------------------------------------------------------------------------------
-- 2) Locate and read gsst_partial_crys_data/gsst_partial_crys_C{c_value}.csv
------------------------------------------------------------------------------
local function load_material_data(cval)
    -- Load from our new data directory and filename format
    local fname = string.format("gsst_partial_crys_data/gsst_partial_crys_C%.1f.csv", cval)
    local file = io.open(fname, "r")
    if not file then
        error("Could not open file '"..fname.."' for reading. Check that it exists.")
    end
    local data = {}
    local header = true
    for line in file:lines() do
        if header then
            header = false -- skip the first line
        else
            -- Each line: Wavelength_nm,n_eff,k_eff
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
    file:close()
    if #data == 0 then
        error("Material data file '"..fname.."' was read but had zero data rows.")
    end
    return data, fname
end
local mat_data, mat_fname = load_material_data(c_value)

------------------------------------------------------------------------------
-- 3) Prepare to store data (optional)
------------------------------------------------------------------------------
local out_file = nil
local outname = nil
if store_data then
    os.execute('mkdir -p "results-nir"')
    local dt = os.date("%Y%m%d_%H%M%S")
    outname = string.format("results-nir/gsst_nir_c%.1f_t%d_%s.csv", c_value, thickness, dt)
    out_file = io.open(outname, "w")
    if not out_file then
        error("Failed to open output CSV file for writing: "..(outname or "??"))
    end
    -- Write a header
    out_file:write("c_value,thickness_nm,wavelength_nm,n_eff,k_eff,R,T,R_plus_T\n")
end

------------------------------------------------------------------------------
-- 4) Main simulation loop
------------------------------------------------------------------------------
-- Progress bar function for verbose mode
local function progress_bar(current, total, width)
    local frac = current/total
    local fill = math.floor(frac*width)
    local line = string.rep("#", fill) .. string.rep("-", width - fill)
    io.write(string.format("\r[%s] %3d%% (%d/%d)", line, math.floor(frac*100+0.5), current, total))
    io.flush()
    if current == total then
        print("") -- new line on completion
    end
end

-- Start simulations
local total_rows = #mat_data
local S = nil
for i, row in ipairs(mat_data) do
    local lam = row.lambda    -- wavelength in nm
    local nr  = row.n         -- refractive index
    local kr  = row.k         -- extinction coefficient
    
    -- Set lattice size to 0.6nm less than the wavelength
    local lattice_size = 0.6
    
    -- Calculate the complex permittivity
    local eps_real = nr*nr - kr*kr
    local eps_imag = 2*nr*kr
    
    -- Create S4 simulation
    S = S4.NewSimulation()
    S:SetLattice({lattice_size, 0}, {0, lattice_size})  -- square lattice with size = wavelength-0.6nm
    S:SetNumG(40)  -- plane-wave expansions
    
    local matname = string.format("GSST_C%.1f", c_value)
    S:AddMaterial("Vacuum", {1,0})
    S:AddMaterial(matname, {eps_real, eps_imag})
    
    -- Convert thickness from nm to um for S4
    local thickness_um = thickness / 1000
    
    -- Add layers
    S:AddLayer("AirAbove", 0, "Vacuum")
    S:AddLayer("MetaLayer", thickness_um, "Vacuum")
    S:AddLayerCopy("AirBelow", 0, "AirAbove")
    
    -- Place the polygon
    S:SetLayerPatternPolygon("MetaLayer", matname, {0,0}, 0, outer_polygon)
    
    -- Normally-located plane wave incidence
    S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
    
    -- Set frequency (1/um) from wavelength (nm)
    local freq = 1000 / lam  -- 1/um
    S:SetFrequency(freq)
    
    -- Get reflection, transmission
    local fwdA, backA = S:GetPoyntingFlux("AirAbove", 0)  -- above
    local fwdB, backB = S:GetPoyntingFlux("AirBelow", 0)  -- below
    local R = -backA  -- reflect
    local T = fwdB    -- transmit
    
    if verbose then
        print(string.format(
            "λ=%.2f nm, (n=%.3f,k=%.3f) => R=%.4f, T=%.4f, R+T=%.4f",
            lam, nr, kr, R, T, (R+T)
        ))
    elseif verbose and (i % 10 == 0) then
        -- optionally some minimal logs every N steps
        print(string.format("Row %d of %d done...", i, total_rows))
    end
    
    if store_data and out_file then
        out_file:write(string.format("%.1f,%d,%.1f,%.5f,%.5f,%.6f,%.6f,%.6f\n",
            c_value, thickness, lam, nr, kr, R, T, (R + T)
        ))
    end
    
    if verbose then
        -- show progress bar
        progress_bar(i, total_rows, 50)
    end
end

-- Close file if we opened it
if store_data and out_file then
    out_file:close()
    print("Saved to " .. outname)
end

-- Done
if verbose then
    print(string.format("Completed all simulations for c=%.1f with thickness=%dnm", c_value, thickness))
end
