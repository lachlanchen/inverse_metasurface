--------------------------------------------------------------------------------
-- metasurface_progress_precise.lua
--
-- Corrected to:
--   1) Clarify the lattice size is 1×1.
--   2) Shift the generated polygon so its center is at (0.5, 0.5).
--------------------------------------------------------------------------------

------------------------------------------------------------------------------
-- 0) Argument Parsing
------------------------------------------------------------------------------
local default_N_quarter = 4   -- means 16 total vertices (4*N_quarter)
local default_num_shapes = 10
local verbose = false
local store_data = false

local N_quarter = default_N_quarter
local num_shapes = default_num_shapes

local arg_str = S4.arg
if arg_str then
    -- Break the S4.arg string into tokens
    local tokens = {}
    for tok in arg_str:gmatch("%S+") do
        table.insert(tokens, tok)
    end

    -- Try interpreting the first token as N_quarter
    if tokens[1] then
        local val = tonumber(tokens[1])
        if val then
            N_quarter = val
        end
    end

    -- Try interpreting the second token as num_shapes
    if tokens[2] then
        local val = tonumber(tokens[2])
        if val then
            num_shapes = val
        end
    end

    -- Any remaining tokens might be flags
    for i=3,#tokens do
        if tokens[i] == "-v" then
            verbose = true
        elseif tokens[i] == "-s" then
            store_data = true
        end
    end
end

-- Just to ensure N_quarter is positive:
if N_quarter < 1 then
    N_quarter = default_N_quarter
end

-- We compute total polygon vertices
local N_outer  = N_quarter * 4

-- Print out the final args
print("Parsed arguments:")
print(string.format("  N_quarter = %d  => total polygon vertices = %d", N_quarter, N_outer))
print(string.format("  num_shapes = %d", num_shapes))
print(string.format("  verbose = %s", tostring(verbose)))
print(string.format("  store_data = %s", tostring(store_data)))

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
-- 3) Generate C4-symmetric polygon
--    After generating the shape in the range ~ [-0.6..0.6], we shift it
--    so the overall shape is centered near (0.5, 0.5) in the 1×1 cell.
------------------------------------------------------------------------------
function generate_c4_polygon(N, base_radius, rand_amt)
    if (N % 4) ~= 0 then
        error("generate_c4_polygon: N must be divisible by 4 for perfect C4 symmetry.")
    end

    local verts = {}
    local two_pi = 2 * math.pi
    local quarter = N / 4

    -- Generate radius array for one quadrant
    local radii = {}
    for i=1, quarter do
        -- random radius around base_radius ± rand_amt
        local r = base_radius + rand_amt*(2*math.random()-1)
        -- range => [base_radius - rand_amt, base_radius + rand_amt]
        -- e.g. with base_radius=0.3, rand_amt=0.3 => range is [0..0.6]
        table.insert(radii, r)
    end

    -- Construct the entire polygon
    for i=0, N-1 do
        local angle = i * (two_pi / N)
        local idx   = (i % quarter) + 1
        local r     = radii[idx]
        local x     = r * math.cos(angle)
        local y     = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end

    -- Small random rotation offset in [0..(π / (2*N_quarter))], just to avoid
    -- always aligning vertices with the x-axis:
    local angle_offset = math.random() * (math.pi / (2*N / 4))
    local cosA = math.cos(angle_offset)
    local sinA = math.sin(angle_offset)
    for i=1, #verts, 2 do
        local x = verts[i]
        local y = verts[i+1]
        local rx = x*cosA - y*sinA
        local ry = x*sinA + y*cosA
        verts[i]   = rx
        verts[i+1] = ry
    end

    ----------------------------------------------------------------------------
    -- SHIFT the entire shape so that (0,0) → (0.5, 0.5).
    -- That places the shape near the center of a 1×1 cell.
    ----------------------------------------------------------------------------
    for i=1, #verts, 2 do
        verts[i]   = verts[i]   + 0.5
        verts[i+1] = verts[i+1] + 0.5
    end

    return verts
end

------------------------------------------------------------------------------
-- 4) Save polygon to file
------------------------------------------------------------------------------
function save_polygon_to_file(filename, polygon)
    local file = io.open(filename, "w")
    if not file then
        error("Could not open '" .. filename .. "' for writing.")
    end
    for i=1,#polygon,2 do
        file:write(string.format("%.6f,%.6f\n", polygon[i], polygon[i+1]))
    end
    file:close()
end

------------------------------------------------------------------------------
-- 5) Collect the CSV files
------------------------------------------------------------------------------
local folder = "partial_crys_data"
local all_csvs = list_csv(folder)
if #all_csvs == 0 then
    error("No CSV files found in '"..folder.."'. Check folder path or file extension.")
end

------------------------------------------------------------------------------
-- 6) Prepare subfolders / output
------------------------------------------------------------------------------
-- For reproducibility, fix a seed
math.randomseed(12345)

-- Setting up polygon radius parameters:
local base_outer = 0.30
local rand_outer = 0.30  -- so radius ∈ [0, 0.6]

local dt = os.date("%Y%m%d_%H%M%S")
-- Shapes folder includes N_quarter and num_shapes in the name
local shapes_subfolder = string.format("shapes/%s-poly-wo-hollow-nQ%d-nS%d", dt, N_quarter, num_shapes)
os.execute('mkdir -p "'..shapes_subfolder..'"')

-- If storing data, open a CSV file for results
local out_file
if store_data then
    os.execute('mkdir -p "results"')
    local outname = string.format("results/%s_output_nQ%d_nS%d.csv", dt, N_quarter, num_shapes)
    out_file = io.open(outname, "w")
    if not out_file then
        error("Could not open results CSV for writing: "..outname)
    end
    -- Write CSV header
    out_file:write("csvfile,shape_idx,row_idx,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T\n")
end

------------------------------------------------------------------------------
-- 7) Determine total for progress bar
------------------------------------------------------------------------------
-- We'll sum the total row-count across all CSVs, times the number of shapes.
local total_iterations = 0
local row_counts = {}
for _, csvfile in ipairs(all_csvs) do
    local d = read_csv(csvfile)
    table.insert(row_counts, #d)
    total_iterations = total_iterations + (#d * num_shapes)
end

local current_count = 0
local bar_width = 50

------------------------------------------------------------------------------
-- 8) Main loop with progress bar
------------------------------------------------------------------------------
for csv_index, csvfile in ipairs(all_csvs) do
    -- re-read this CSV's data
    local data = read_csv(csvfile)

    -- Attempt to parse some unique matname from CSV filename
    local matname = "PartialGSST"
    do
        local c_str = csvfile:match("partial_crys_C([^/]+)%.csv")
        if c_str then
            matname = "PartialGSST_C"..c_str
        end
    end

    for shape_idx = 1, num_shapes do
        -- Generate polygon (already centered at (0.5, 0.5))
        local outer_poly = generate_c4_polygon(N_outer, base_outer, rand_outer)
        -- Save polygon
        local shape_fname = string.format("%s/outer_shape%d.txt",
                                          shapes_subfolder, shape_idx)
        save_polygon_to_file(shape_fname, outer_poly)

        for i, row in ipairs(data) do
            local lam = row.wavelength  -- micrometers
            local n   = row.n_eff
            local k   = row.k_eff
            local freq = 1.0 / lam

            -- Build the permittivity
            local eps_real = n*n - k*k
            local eps_imag = 2*n*k

            -- -----------------------------
            -- Build the S4 simulation
            -- -----------------------------
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
            S:SetLayerPatternPolygon("MetaLayer", matname, {0,0}, 0, outer_poly)

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
                    "%s | shape=%d, row=%d, λ=%.17g µm, freq=%.17g, (n=%.17g, k=%.17g) => R=%.17g, T=%.17g, R+T=%.17g",
                    csvfile, shape_idx, i, lam, freq, n, k, R, T, (R + T)
                ))
            end

            if store_data then
                out_file:write(string.format(
                    "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    csvfile, shape_idx, i, lam, freq, n, k, R, T, (R + T)
                ))
            end

            -- Progress bar update
            current_count = current_count + 1
            local fraction = current_count / total_iterations
            local fill = math.floor(fraction * bar_width)
            local line = string.rep("#", fill)
                        .. string.rep("-", bar_width - fill)

            io.write(string.format(
                "\r[%s] %3d%% (csv=%d/%d, shape=%d/%d, row=%d/%d)",
                line,
                math.floor(fraction*100 + 0.5),
                csv_index, #all_csvs,
                shape_idx, num_shapes,
                i, #data
            ))
            io.flush()
        end
    end
end

print("")  -- Move to a new line after finishing

if out_file then
    out_file:close()
end

