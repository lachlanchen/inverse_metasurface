--------------------------------------------------------------------------------
-- metasurface_allargs_resume.lua
-- 
-- Accepts arguments in this order:
--   1) N_quarter
--   2) num_shapes
--   3) random_seed
--   4) prefix
--   5) num_g
--   6) base_outer
--   7) rand_outer
--   8+) optional flags (-v, -s)
--
-- Incorporates them all into the filename. If prefix == "", we use datetime
-- and do not resume. If prefix != "", we resume if a file with that name exists.
--------------------------------------------------------------------------------

------------------------------------------------------------------------------
-- 0) Argument Parsing
------------------------------------------------------------------------------
local default_N_quarter   = 4
local default_num_shapes  = 10
local default_random_seed = 88888
local default_prefix      = ""
local default_num_g       = 80
local default_base_outer  = 0.25
local default_rand_outer  = 0.20

local verbose     = false
local store_data  = false

local N_quarter   = default_N_quarter
local num_shapes  = default_num_shapes
local random_seed = default_random_seed
local prefix      = default_prefix
local num_g       = default_num_g
local base_outer  = default_base_outer
local rand_outer  = default_rand_outer

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

    -- 3) random_seed
    if tokens[3] then
        local val = tonumber(tokens[3])
        if val then
            random_seed = val
        end
    end

    -- 4) prefix
    if tokens[4] then
        prefix = tokens[4]
    end

    -- 5) num_g
    if tokens[5] then
        local val = tonumber(tokens[5])
        if val then
            num_g = val
        end
    end

    -- 6) base_outer
    if tokens[6] then
        local val = tonumber(tokens[6])
        if val then
            base_outer = val
        end
    end

    -- 7) rand_outer
    if tokens[7] then
        local val = tonumber(tokens[7])
        if val then
            rand_outer = val
        end
    end

    -- 8+) optional flags
    for i = 8, #tokens do
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
print(string.format("  prefix      = '%s'", prefix))
print(string.format("  num_g       = %d", num_g))
print(string.format("  base_outer  = %.4f", base_outer))
print(string.format("  rand_outer  = %.4f", rand_outer))
print(string.format("  verbose     = %s", tostring(verbose)))
print(string.format("  store_data  = %s", tostring(store_data)))

------------------------------------------------------------------------------
-- 1) CSV listing
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

local folder = "partial_crys_data"
local all_csvs = list_csv(folder)
table.sort(all_csvs)
if #all_csvs == 0 then
    error("No CSV files found in '"..folder.."'.")
end

------------------------------------------------------------------------------
-- 2) read_csv
------------------------------------------------------------------------------
function read_csv(csvfile)
    local data = {}
    local file, err = io.open(csvfile, "r")
    if not file then
        error("Could not open '"..csvfile.."': "..(err or ""))
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
                    table.insert(data, {
                        wavelength = lam,
                        n_eff      = n,
                        k_eff      = k
                    })
                end
            end
        end
    end
    file:close()
    return data
end

------------------------------------------------------------------------------
-- 3) randomseed
------------------------------------------------------------------------------
math.randomseed(random_seed)

------------------------------------------------------------------------------
-- 4) Generate polygon
------------------------------------------------------------------------------
function generate_c4_polygon(N, base_r, rand_amt)
    if (N % 4) ~= 0 then
        error("generate_c4_polygon: N must be divisible by 4 for perfect C4 symmetry.")
    end
    local verts = {}
    local two_pi = 2 * math.pi
    local quarter = N / 4

    local radii = {}
    for i=1, quarter do
        local r = base_r + rand_amt*(2*math.random() - 1)
        table.insert(radii, r)
    end

    for i=0, N-1 do
        local angle = i*(two_pi/N)
        local idx   = (i % quarter) + 1
        local rad   = radii[idx]
        local x     = rad * math.cos(angle)
        local y     = rad * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end

    local angle_offset = math.random()*(math.pi/(2*N/4))
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

    -- shift to center
    for i=1, #verts, 2 do
        verts[i]   = verts[i]   + 0.5
        verts[i+1] = verts[i+1] + 0.5
    end
    return verts
end

------------------------------------------------------------------------------
-- 5) Save polygon
------------------------------------------------------------------------------
function save_polygon_to_file(filename, polygon)
    local file = io.open(filename, "w")
    if not file then
        error("Could not open '"..filename.."' for writing.")
    end
    for i=1, #polygon, 2 do
        file:write(string.format("%.17g,%.17g\n", polygon[i], polygon[i+1]))
    end
    file:close()
end

------------------------------------------------------------------------------
-- 6) Decide final prefix (datetime if prefix == "")
------------------------------------------------------------------------------
local date_prefix = os.date("%Y%m%d_%H%M%S")
local do_resume = true

if prefix == "" then
    prefix = date_prefix  -- replace with datetime
    do_resume = false     -- brand new run
    print("No prefix => new run => using datetime prefix: "..prefix)
else
    print("Prefix given => attempt resume if file exists: "..prefix)
end

------------------------------------------------------------------------------
-- 7) Build final name that includes all parameters
------------------------------------------------------------------------------
-- e.g. "myrun_seed12345_g80_nQ4_nS100000_b0.25_r0.20"
-- We'll format base_outer, rand_outer with, say, 2 decimal places
local final_name = string.format(
    "%s_seed%d_g%d_nQ%d_nS%d_b%.2f_r%.2f",
    prefix, random_seed, num_g, N_quarter, num_shapes, base_outer, rand_outer
)

local shapes_subfolder = "shapes/"..final_name
local out_filename     = "results/"..final_name..".csv"

os.execute('mkdir -p "'..shapes_subfolder..'"')
os.execute('mkdir -p "results"')

------------------------------------------------------------------------------
-- 8) Attempt to parse resume if store_data=true and do_resume=true
------------------------------------------------------------------------------
local out_file = nil

local resume_info = nil
local resume_shape_idx = 1
local resume_csv_index = 1
local resume_row_index = 1

local csv_index_map = {}
for i, f in ipairs(all_csvs) do
    csv_index_map[f] = i
end

local function parse_resume_position(existing_file)
    local f = io.open(existing_file, "r")
    if not f then return nil end
    local last_valid
    for line in f:lines() do
        local count=0
        for _ in line:gmatch("[^,]+") do count=count+1 end
        if count == 10 then
            last_valid = line
        end
    end
    f:close()
    if not last_valid then return nil end

    local csvfile_str, shape_str, row_str =
        last_valid:match("([^,]+),([^,]+),([^,]+),[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+")
    if not (csvfile_str and shape_str and row_str) then
        return nil
    end
    return {
        csv_filename = csvfile_str,
        shape_idx    = tonumber(shape_str),
        row_idx      = tonumber(row_str)
    }
end

if store_data then
    if do_resume then
        local check_handle = io.open(out_filename, "r")
        if check_handle then
            check_handle:close()
            -- parse last line
            local info = parse_resume_position(out_filename)
            if info then
                resume_info = info
                out_file = io.open(out_filename, "a")
                print(string.format("Resuming from '%s': shape=%d, file=%s, row=%d",
                                    out_filename, info.shape_idx, info.csv_filename, info.row_idx))
            else
                -- no valid data, overwrite
                out_file = io.open(out_filename, "w")
                out_file:write("csvfile,shape_idx,row_idx,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T\n")
                print("Existing file but no valid data => overwriting: "..out_filename)
            end
        else
            -- new run
            out_file = io.open(out_filename, "w")
            out_file:write("csvfile,shape_idx,row_idx,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T\n")
            print("No existing file => new run => "..out_filename)
        end
    else
        -- do_resume=false => always new
        out_file = io.open(out_filename, "w")
        out_file:write("csvfile,shape_idx,row_idx,wavelength_um,freq_1perum,n_eff,k_eff,R,T,R_plus_T\n")
        print("Forcing new file => "..out_filename)
    end
else
    print("store_data=false => not generating CSV output.")
end

if resume_info then
    resume_shape_idx = resume_info.shape_idx
    resume_row_index = resume_info.row_idx + 1
    local idx = csv_index_map[resume_info.csv_filename]
    if idx then
        resume_csv_index = idx
    end
end

------------------------------------------------------------------------------
-- 9) Count total for progress bar
------------------------------------------------------------------------------
local total_iterations = 0
local data_cache = {}
for _, csvfile in ipairs(all_csvs) do
    local d = read_csv(csvfile)
    data_cache[csvfile] = d
    total_iterations = total_iterations + (#d * num_shapes)
end

local current_count = 0
local bar_width = 50

if resume_info and out_file then
    local lines_done = 0
    local f = io.open(out_filename, "r")
    if f then
        for line in f:lines() do
            local c=0
            for _ in line:gmatch("[^,]+") do c=c+1 end
            if c == 10 then
                lines_done = lines_done + 1
            end
        end
        f:close()
    end
    current_count = lines_done - 1  -- minus header
end

------------------------------------------------------------------------------
-- 10) Main loop
------------------------------------------------------------------------------
for shape_idx = resume_shape_idx, num_shapes do
    local outer_poly = generate_c4_polygon(N_outer, base_outer, rand_outer)
    local shape_fname = string.format("%s/outer_shape%d.txt", shapes_subfolder, shape_idx)
    save_polygon_to_file(shape_fname, outer_poly)

    local start_csv_index = 1
    if shape_idx == resume_shape_idx then
        start_csv_index = resume_csv_index
    end

    for csv_i = start_csv_index, #all_csvs do
        local csvfile = all_csvs[csv_i]
        local data = data_cache[csvfile]

        local matname = "PartialGSST"
        do
            local c_str = csvfile:match("partial_crys_C([^/]+)%.csv")
            if c_str then
                matname = "PartialGSST_C"..c_str
            end
        end

        local start_row = 1
        if shape_idx == resume_shape_idx and csv_i == resume_csv_index then
            start_row = resume_row_index
        end

        for i = start_row, #data do
            local lam = data[i].wavelength
            local n   = data[i].n_eff
            local k   = data[i].k_eff
            local freq = 1.0 / lam

            local eps_real = n*n - k*k
            local eps_imag = 2*n*k

            local S = S4.NewSimulation()
            S:SetLattice({1,0}, {0,1})
            S:SetNumG(num_g)

            S:AddMaterial("Vacuum", {1,0})
            S:AddMaterial(matname, {eps_real, eps_imag})

            S:AddLayer("AirAbove",   0,   "Vacuum")
            S:AddLayer("MetaLayer",  0.5, "Vacuum")
            S:AddLayerCopy("AirBelow", 0, "AirAbove")

            S:SetLayerPatternPolygon("MetaLayer", matname, {0,0}, 0, outer_poly)

            S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
            S:SetFrequency(freq)

            local fwdA, backA = S:GetPoyntingFlux("AirAbove",0)
            local fwdB, backB = S:GetPoyntingFlux("AirBelow",0)
            local R = -backA
            local T =  fwdB

            if verbose then
                print(string.format(
                    "%s | shape=%d, row=%d, λ=%.17g µm, freq=%.17g, (n=%.17g, k=%.17g) => R=%.17g, T=%.17g, R+T=%.17g",
                    csvfile, shape_idx, i, lam, freq, n, k, R, T, (R+T)
                ))
            end

            if store_data and out_file then
                out_file:write(string.format(
                    "%s,%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                    csvfile, shape_idx, i, lam, freq, n, k, R, T, (R + T)
                ))
            end

            current_count = current_count + 1
            local fraction = current_count / total_iterations
            local fill = math.floor(fraction * bar_width)
            local bar = string.rep("#", fill) .. string.rep("-", bar_width - fill)

            io.write(string.format(
                "\r[%s] %3d%% (%s, shape=%d/%d, csv=%d/%d, row=%d/%d)",
                bar,
                math.floor(fraction*100+0.5),
                final_name,
                shape_idx, num_shapes,
                csv_i, #all_csvs,
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

