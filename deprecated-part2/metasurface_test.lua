--------------------------------------------------------------------------------
-- metasurface_test.lua
--
-- A simplified version of metasurface_progress_centered.lua, using user-supplied
-- polygon vertices (instead of random generation). The rest is kept similar:
--   * We read partial_crys_data/*.csv for optical constants
--   * We create an S4 simulation with a 1×1 lattice
--   * We measure R and T for each row of each CSV
--   * We optionally store results to a CSV
-- 
-- Usage:
--   S4 -a "x1,y1;x2,y2;x3,y3;... -c 0.0 -v -s" metasurface_test.lua
--
-- Where:
--   - The first token is the semicolon-separated vertex list: xN,yN
--   - '-c 0.0' is an optional argument that sets an extra uniform shift 
--        to both x and y. Defaults to 0.
--   - '-v' optional flag for verbose output.
--   - '-s' optional flag to store output CSV to "results/" folder.
--------------------------------------------------------------------------------

------------------------------------------------------------------------------
-- 0) Argument Parsing
------------------------------------------------------------------------------
local polygon_vertices = {}
local center_offset   = 0.0
local verbose         = false
local store_data      = false

-- Helper to parse a semicolon-separated string of "x,y" pairs.
local function parse_polygon(str)
    local verts = {}
    for pair in str:gmatch("[^;]+") do
        local x_str, y_str = pair:match("([^,]+),([^,]+)")
        if x_str and y_str then
            local x_val = tonumber(x_str)
            local y_val = tonumber(y_str)
            if x_val and y_val then
                table.insert(verts, x_val)
                table.insert(verts, y_val)
            end
        end
    end
    return verts
end

-- Process S4.arg
local arg_str = S4.arg
if arg_str then
    -- Break into tokens
    local tokens = {}
    for tok in arg_str:gmatch("%S+") do
        table.insert(tokens, tok)
    end

    -- 1) Attempt to interpret the first token as a polygon string "x1,y1;x2,y2;..."
    if tokens[1] then
        polygon_vertices = parse_polygon(tokens[1])
    end

    -- 2) Look for flags among subsequent tokens
    local i = 2
    while i <= #tokens do
        local t = tokens[i]
        if t == "-c" then
            -- Next token should be a center offset
            if tokens[i+1] then
                center_offset = tonumber(tokens[i+1]) or 0.0
                i = i + 1
            end
        elseif t == "-v" then
            verbose = true
        elseif t == "-s" then
            store_data = true
        end
        i = i + 1
    end
end

if #polygon_vertices < 6 then
    error("Not enough vertices passed in. Expecting at least three (x,y) pairs.")
end

-- Apply the same 0.5 shift from the original code + user-provided offset
for i = 1, #polygon_vertices, 2 do
    polygon_vertices[i]   = polygon_vertices[i]   -- + 0.5 + center_offset
    polygon_vertices[i+1] = polygon_vertices[i+1] -- + 0.5 + center_offset
end

print("Parsed polygon vertices (after shift):")
for i=1, #polygon_vertices, 2 do
    print(string.format("  (%.6f, %.6f)", polygon_vertices[i], polygon_vertices[i+1]))
end

print(string.format("center_offset = %.6f", center_offset))
print(string.format("verbose       = %s", tostring(verbose)))
print(string.format("store_data    = %s", tostring(store_data)))

------------------------------------------------------------------------------
-- 1) Helper: list_csv(folder)
------------------------------------------------------------------------------
function list_csv(folder)
    local t = {}
    local pipe = io.popen('ls "'..folder..'"/*.csv 2> /dev/null')
    if pipe then
        for line in pipe:lines() do
            table.insert(t, line)
        end
        pipe:close()
    end
    return t
end

------------------------------------------------------------------------------
-- 2) read_csv(csvfile)
------------------------------------------------------------------------------
function read_csv(csvfile)
    local data = {}
    local file, err = io.open(csvfile, "r")
    if not file then
        error("Could not open '" .. csvfile .. "': " .. (err or ""))
    end

    local header = true
    for line in file:lines() do
        if header then
            header = false -- skip the first line (header row)
        else
            local lam_str, n_str, k_str = line:match("([^,]+),([^,]+),([^,]+)")
            if lam_str and n_str and k_str then
                local lam = tonumber(lam_str)
                local n   = tonumber(n_str)
                local k   = tonumber(k_str)
                if lam and n and k then
                    table.insert(data, { wavelength=lam, n_eff=n, k_eff=k })
                end
            end
        end
    end

    file:close()
    return data
end

------------------------------------------------------------------------------
-- 3) Collect the CSV files
------------------------------------------------------------------------------
local folder = "partial_crys_data"
local all_csvs = list_csv(folder)
if #all_csvs == 0 then
    error("No CSV files found in '"..folder.."'. Check folder path or file extension.")
end

------------------------------------------------------------------------------
-- 4) Prepare subfolders / output
------------------------------------------------------------------------------
-- For reproducibility, fix a seed (same as before)
math.randomseed(88888)

-- If storing data, open a CSV file for results
local dt = os.date("%Y%m%d_%H%M%S")
local out_file
if store_data then
    os.execute('mkdir -p "results"')
    local outname = string.format("results/%s_test_output.csv", dt)
    out_file = io.open(outname, "w")
    if not out_file then
        error("Could not open results CSV for writing: "..outname)
    end
    -- Write CSV header
    out_file:write("csvfile,shape_idx,row_idx,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T\n")
end

------------------------------------------------------------------------------
-- 5) Determine total for progress bar
------------------------------------------------------------------------------
-- We'll sum the total row-count across all CSVs (we do only 1 shape here).
local total_iterations = 0
local row_counts = {}
for _, csvfile in ipairs(all_csvs) do
    local d = read_csv(csvfile)
    table.insert(row_counts, #d)
    total_iterations = total_iterations + #d
end

local current_count = 0
local bar_width = 50

------------------------------------------------------------------------------
-- 6) Main loop with progress bar
------------------------------------------------------------------------------
local shape_idx = 1  -- we have just one direct polygon

for csv_index, csvfile in ipairs(all_csvs) do
    local data = read_csv(csvfile)

    -- Attempt to parse material name from CSV filename
    local matname = "PartialGSST"
    do
        local c_str = csvfile:match("partial_crys_C([^/]+)%.csv")
        if c_str then
            matname = "PartialGSST_C"..c_str
        end
    end

    for i, row in ipairs(data) do
        local lam  = row.wavelength  -- micrometers
        local n    = row.n_eff
        local k    = row.k_eff
        local freq = 1.0 / lam

        -- Build the permittivity
        local eps_real = n*n - k*k
        local eps_imag = 2*n*k

        -- Create S4 simulation
        local S = S4.NewSimulation()
        -- Lattice size is explicitly 1×1
        S:SetLattice({1,0}, {0,1})
        S:SetNumG(40)

        -- Materials
        S:AddMaterial("Vacuum", {1, 0})
        S:AddMaterial(matname, {eps_real, eps_imag})

        -- Layers
        S:AddLayer("AirAbove",   0,   "Vacuum")
        S:AddLayer("MetaLayer",  0.5, "Vacuum")
        S:AddLayerCopy("AirBelow", 0, "AirAbove")

        -- Pattern
        -- (We directly use the polygon vertices from polygon_vertices)
        S:SetLayerPatternPolygon("MetaLayer", matname, {0,0}, 0, polygon_vertices)

        -- Source
        S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
        S:SetFrequency(freq)

        -- Reflection & Transmission
        local fwdA, backA = S:GetPoyntingFlux("AirAbove",  0)
        local fwdB, backB = S:GetPoyntingFlux("AirBelow",  0)
        local R = -backA  -- reflection from top
        local T =  fwdB   -- transmission to bottom

        if verbose then
            print(string.format(
                "%s | shape=%d, row=%d, λ=%.6f µm, freq=%.6f, n=%.6f, k=%.6f => R=%.6f, T=%.6f, R+T=%.6f",
                csvfile, shape_idx, i, lam, freq, n, k, R, T, (R + T)
            ))
        end

        if out_file then
            out_file:write(string.format(
                "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                csvfile, shape_idx, i, lam, freq, n, k, R, T, (R + T)
            ))
        end

        -- Progress bar update
        current_count = current_count + 1
        local fraction = current_count / total_iterations
        local fill = math.floor(fraction * bar_width)
        local line = string.rep("#", fill) .. string.rep("-", bar_width - fill)

        io.write(string.format(
            "\r[%s] %3d%% (csv=%d/%d, row=%d/%d)",
            line,
            math.floor(fraction*100 + 0.5),
            csv_index, #all_csvs,
            i, #data
        ))
        io.flush()
    end
end

print("")  -- Move to new line after finishing

if out_file then
    out_file:close()
end

