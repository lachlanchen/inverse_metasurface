--------------------------------------------------------------------------------
-- metasurface_unique_shape.lua
--
-- Minimal update to accept a random seed as the 3rd argument.
-- Everything else remains the same.
--------------------------------------------------------------------------------

------------------------------------------------------------------------------
-- 0) Argument Parsing
------------------------------------------------------------------------------
local default_N_quarter = 4   -- means 16 total vertices (4*N_quarter)
local default_num_shapes = 10
local default_random_seed = 88888

local verbose = false
local store_data = false

local N_quarter = default_N_quarter
local num_shapes = default_num_shapes
local random_seed = default_random_seed

local arg_str = S4.arg
if arg_str then
    local tokens = {}
    for tok in arg_str:gmatch("%S+") do
        table.insert(tokens, tok)
    end

    -- 1) N_quarter
    if tokens[1] then
        local val = tonumber(tokens[1])
        if val then
            N_quarter = val
        end
    end

    -- 2) num_shapes
    if tokens[2] then
        local val = tonumber(tokens[2])
        if val then
            num_shapes = val
        end
    end

    -- 3) random_seed (optional)
    if tokens[3] then
        local val = tonumber(tokens[3])
        if val then
            random_seed = val
        end
    end

    -- 4) Any remaining tokens may be flags
    for i = 4, #tokens do
        if tokens[i] == "-v" then
            verbose = true
        elseif tokens[i] == "-s" then
            store_data = true
        end
    end
end

if N_quarter < 1 then
    N_quarter = default_N_quarter
end

local N_outer = N_quarter * 4

print("Parsed arguments:")
print(string.format("  N_quarter   = %d  => total polygon vertices = %d", N_quarter, N_outer))
print(string.format("  num_shapes  = %d", num_shapes))
print(string.format("  random_seed = %d", random_seed))
print(string.format("  verbose     = %s", tostring(verbose)))
print(string.format("  store_data  = %s", tostring(store_data)))

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
            header = false
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
--    After generating the shape in ~[-r..r], shift to (0.5, 0.5).
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
    for i = 1, quarter do
        local r = base_radius + rand_amt * (2 * math.random() - 1)
        table.insert(radii, r)
    end

    -- Construct the entire polygon
    for i = 0, N - 1 do
        local angle = i * (two_pi / N)
        local idx   = (i % quarter) + 1
        local r     = radii[idx]
        local x     = r * math.cos(angle)
        local y     = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end

    -- Small random rotation offset
    local angle_offset = math.random() * (math.pi / (2 * N / 4))
    local cosA = math.cos(angle_offset)
    local sinA = math.sin(angle_offset)

    for i = 1, #verts, 2 do
        local x = verts[i]
        local y = verts[i+1]
        local rx = x*cosA - y*sinA
        local ry = x*sinA + y*cosA
        verts[i]   = rx
        verts[i+1] = ry
    end

    -- Shift so that (0,0) -> (0.5, 0.5)
    for i = 1, #verts, 2 do
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
    for i = 1, #polygon, 2 do
        file:write(string.format("%.17g,%.17g\n", polygon[i], polygon[i+1]))
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
-- Set the user-chosen or default random seed
math.randomseed(random_seed)

-- New requested values:
local base_outer = 0.25
local rand_outer = 0.20

local dt = os.date("%Y%m%d_%H%M%S")
local shapes_subfolder = string.format("shapes/%s-poly-wo-hollow-nQ%d-nS%d", dt, N_quarter, num_shapes)
os.execute('mkdir -p "'..shapes_subfolder..'"')

local out_file
if store_data then
    os.execute('mkdir -p "results"')
    local outname = string.format("results/%s_output_nQ%d_nS%d.csv", dt, N_quarter, num_shapes)
    out_file = io.open(outname, "w")
    if not out_file then
        error("Could not open results CSV for writing: "..outname)
    end
    out_file:write("csvfile,shape_idx,row_idx,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T\n")
end

------------------------------------------------------------------------------
-- 7) Determine total for progress bar
------------------------------------------------------------------------------
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
-- 8) Main loop
--    A single shape per shape_idx is reused for ALL CSV files (c values).
------------------------------------------------------------------------------
for shape_idx = 1, num_shapes do
    -- Generate the polygon for this shape_idx
    local outer_poly = generate_c4_polygon(N_outer, base_outer, rand_outer)

    -- Save polygon to file
    local shape_fname = string.format("%s/outer_shape%d.txt", shapes_subfolder, shape_idx)
    save_polygon_to_file(shape_fname, outer_poly)

    -- For each CSV file, apply the same shape
    for csv_index, csvfile in ipairs(all_csvs) do
        local data = read_csv(csvfile)

        -- Attempt to parse a unique matname from CSV
        local matname = "PartialGSST"
        do
            local c_str = csvfile:match("partial_crys_C([^/]+)%.csv")
            if c_str then
                matname = "PartialGSST_C"..c_str
            end
        end

        for i, row in ipairs(data) do
            local lam = row.wavelength
            local n   = row.n_eff
            local k   = row.k_eff
            local freq = 1.0 / lam

            local eps_real = n*n - k*k
            local eps_imag = 2*n*k

            local S = S4.NewSimulation()

            -- Lattice size 1×1
            S:SetLattice({1,0}, {0,1})
            S:SetNumG(80)

            -- Materials
            S:AddMaterial("Vacuum", {1, 0})
            S:AddMaterial(matname, {eps_real, eps_imag})

            -- Layers
            S:AddLayer("AirAbove",   0,   "Vacuum")
            S:AddLayer("MetaLayer",  0.5, "Vacuum")
            S:AddLayerCopy("AirBelow", 0, "AirAbove")

            -- Polygon pattern
            S:SetLayerPatternPolygon("MetaLayer", matname, {0,0}, 0, outer_poly)

            -- Source
            S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
            S:SetFrequency(freq)

            local fwdA, backA = S:GetPoyntingFlux("AirAbove",  0)
            local fwdB, backB = S:GetPoyntingFlux("AirBelow",  0)
            local R = -backA
            local T =  fwdB

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

            current_count = current_count + 1
            local fraction = current_count / total_iterations
            local fill = math.floor(fraction * bar_width)
            local line = string.rep("#", fill) .. string.rep("-", bar_width - fill)

            io.write(string.format(
                "\r[%s] %3d%% (shape=%d/%d, csv=%d/%d, row=%d/%d)",
                line,
                math.floor(fraction*100 + 0.5),
                shape_idx, num_shapes,
                csv_index, #all_csvs,
                i, #data
            ))
            io.flush()
        end
    end
end

print("")
if out_file then
    out_file:close()
end

