--------------------------------------------------------------------------------
-- metasurface_batch_polygon_only.lua
-- Reads all *.csv files in partial_crys_data/ and for each row in each file,
-- builds an S4 simulation, measures R,T. 
--
-- We generate 10 random C4-symmetric "outer polygons" for each crystallization,
-- but no hollow/inner polygon this time. We save each shape in 
-- shapes/<datetime>-polygon-wo-hollow/outer_shapeX.txt
--
-- The final result is 11 CSVs × 10 shapes each = 110 shapes if you have 11 CSVs.
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- 0) Helper: List CSV files in partial_crys_data/
--------------------------------------------------------------------------------
function list_csv(folder)
    local t = {}
    -- Use a shell command "ls folder/*.csv" to list CSV filenames
    local pipe = io.popen('ls "'..folder..'"/*.csv 2> /dev/null')
    if pipe then
        for line in pipe:lines() do
            table.insert(t, line)
        end
        pipe:close()
    end
    return t
end

--------------------------------------------------------------------------------
-- 1) Helper: Read a single CSV
--------------------------------------------------------------------------------
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
            -- parse columns: lam_str, n_str, k_str
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

--------------------------------------------------------------------------------
-- 2) Generate C4-symmetric polygon
--    Suppose we want N to be a multiple of 4. We generate N/4 vertices 
--    in [0..π/2], replicate for the other 3 quadrants => perfect C4.
--------------------------------------------------------------------------------
function generate_c4_polygon(N, base_radius, rand_amt)
    if (N % 4) ~= 0 then
        error("generate_c4_polygon: N must be divisible by 4 for perfect C4 symmetry.")
    end

    local verts = {}
    local two_pi = 2 * math.pi
    local quarter = N / 4

    -- Generate radius array for quadrant 1 only
    local radii = {}
    for i=1, quarter do
        -- random radius around base_radius ± rand_amt
        local r = base_radius + rand_amt * (2*math.random() - 1)
        table.insert(radii, r)
    end

    -- We'll accumulate angles from 0..2π in steps of 2π/N
    for i=0, N-1 do
        local angle = i * (two_pi / N)
        -- index in [1..quarter]
        local idx = (i % quarter) + 1
        local r   = radii[idx]
        local x   = r * math.cos(angle)
        local y   = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end

    -- 3) Rotate the entire polygon by a random angle in [0..π/2]
    local angle_offset = math.random() * (math.pi / 2)  -- any angle from 0 to π/2
    local cosA = math.cos(angle_offset)
    local sinA = math.sin(angle_offset)

    for i=1, #verts, 2 do
        local x = verts[i]
        local y = verts[i+1]
        -- standard 2D rotation: (x, y) → (x cosA - y sinA, x sinA + y cosA)
        local rx = x*cosA - y*sinA
        local ry = x*sinA + y*cosA
        verts[i]   = rx
        verts[i+1] = ry
    end


    return verts
end

--------------------------------------------------------------------------------
-- 3) Helper: Save polygon to file
--------------------------------------------------------------------------------
function save_polygon_to_file(filename, polygon)
    local file = io.open(filename, "w")
    if not file then
        error("Could not open '" .. filename .. "' for writing.")
    end
    for i=1,#polygon,2 do
        local x = polygon[i]
        local y = polygon[i+1]
        file:write(string.format("%.6f,%.6f\n", x, y))
    end
    file:close()
end

--------------------------------------------------------------------------------
-- 4) Now find all CSV files
--------------------------------------------------------------------------------
local folder = "partial_crys_data"  -- The folder containing your CSVs
local all_csvs = list_csv(folder)
if #all_csvs == 0 then
    error("No CSV files found in '"..folder.."'. Check folder path or file extension.")
end

--------------------------------------------------------------------------------
-- We'll generate 10 random shapes for each crystallization (CSV).
-- For each shape, we do the entire frequency sweep from each CSV row.
--------------------------------------------------------------------------------

-- Current date/time for subfolder:
local dt = os.date("%Y%m%d_%H%M%S")

-- Create a subfolder like: shapes/20250104_120000-polygon-wo-hollow/
os.execute('mkdir -p "shapes/'..dt..'-polygon-wo-hollow"')

-- Outer polygon parameters:
-- local N_outer  = 12  -- must be multiple of 4
local N_outer  = 16  -- must be multiple of 4
local base_outer = 0.30
local rand_outer = 0.3

-- For reproducibility, pick some base seed if you want consistent shapes
math.randomseed(12345)

--------------------------------------------------------------------------------
-- 5) Loop over each CSV (each crystallization)
--------------------------------------------------------------------------------
for _, csvfile in ipairs(all_csvs) do
    print("Now processing CSV:", csvfile)
    local data = read_csv(csvfile)

    -- Build a unique material name from filename, e.g. "PartialGSST_C0.5"
    local matname = "PartialGSST"
    do
        local c_str = csvfile:match("partial_crys_C([^/]+)%.csv")
        if c_str then
            matname = "PartialGSST_C"..c_str
        end
    end

    ----------------------------------------------------------------------------
    -- For each CSV, generate 10 random shapes and run S4
    ----------------------------------------------------------------------------
    for shape_idx = 1,1000 do
        -- Generate new polygon for each shape
        local outer_poly = generate_c4_polygon(N_outer, base_outer, rand_outer)

        -- Save it
        local shape_fname = string.format("shapes/%s-polygon-wo-hollow/outer_shape%d.txt",
            dt, shape_idx)
        save_polygon_to_file(shape_fname, outer_poly)

        -- Now for each row in the CSV, run the simulation
        for i, row in ipairs(data) do
            local lam = row.wavelength  -- in micrometers
            local n   = row.n_eff
            local k   = row.k_eff

            -- S4 freq in 1/µm
            local freq = 1.0 / lam

            -- Build permittivity = (n + i k)^2
            local eps_real = n*n - k*k
            local eps_imag = 2 * n * k

            -- Create a new S4 simulation
            local S = S4.NewSimulation()
            S:SetLattice({1,0}, {0,1})
            S:SetNumG(40)

            -- Materials
            S:AddMaterial("Vacuum", {1, 0})
            S:AddMaterial(matname,  {eps_real, eps_imag})

            -- Layers
            S:AddLayer("AirAbove",   0,   "Vacuum")
            S:AddLayer("MetaLayer",  0.5, "Vacuum")
            S:AddLayerCopy("AirBelow", 0, "AirAbove")

            -- Paint the outer polygon with PartialGSST (no hollow region)
            S:SetLayerPatternPolygon("MetaLayer", matname, {0,0}, 0, outer_poly)

            -- Excitation
            S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
            S:SetFrequency(freq)

            -- Reflection & Transmission
            local fwdA, backA = S:GetPoyntingFlux("AirAbove",  0)
            local fwdB, backB = S:GetPoyntingFlux("AirBelow",  0)
            local R = backA
            local T = fwdB

            -- Print results
            print(string.format(
                "%s | shape=%d, row=%d, λ=%.3f µm, freq=%.3f, (n=%.3f, k=%.3f) => R=%.4f, T=%.4f, R+T=%.4f",
                csvfile, shape_idx, i, lam, freq, n, k, -R, T, -R+T
            ))
        end
    end
end
