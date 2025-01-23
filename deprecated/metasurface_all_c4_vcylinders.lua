--------------------------------------------------------------------------------
-- metasurface_batch_multi_shapes.lua
-- Reads all *.csv files in partial_crys_data/ and for each row in each file,
-- builds an S4 simulation, measures R,T. 
-- Now we generate 10 different random shapes for each crystallization (CSV),
-- but each shape is made of 1..3 random cylinders in the first quadrant,
-- replicated C4 around the cell center => 4 x that many circles in total.
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
            -- skip the first line (header row)
            header = false
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
-- 2) Generate random cylinders in the first quadrant, replicate via C4 
--    The cell is 1×1 from (0,0) to (1,1). We'll assume the "center" is (0.5,0.5).
--    The "first quadrant" for this rotation is actually the region x >= 0.5, 
--    y >= 0.5. Then we rotate 3 more times about (0.5,0.5).
--
--    Steps:
--      (1) random integer #circles in [1..3] for Q1
--      (2) for each circle, random center in [0.5..1, 0.5..1], 
--          random radius in [radius_min..radius_max].
--      (3) replicate center+radius into Q2, Q3, Q4 by 90°, 180°, 270° rotation 
--          about pivot=(0.5,0.5).
--      (4) return a table of { {cx, cy, r}, ... } for all circles in entire cell.
--------------------------------------------------------------------------------
function generate_c4_cylinders(radius_min, radius_max)
    -- how many circles in first quadrant?
    local num_Q1 = math.random(1,3)

    -- pivot for rotation:
    local px, py = 0.5, 0.5

    -- define a local helper to rotate a point (x,y) 90° about (px,py)
    local function rotate_90_deg(x, y, px, py)
        -- shift to pivot
        local dx = x - px
        local dy = y - py
        -- rotation by +90° in standard math is (x,y)->(-y,x), but let's confirm
        local rx = -dy
        local ry =  dx
        -- shift back
        return rx + px, ry + py
    end

    local circles = {}

    -- generate random circles in quadrant #1 => x in [0.5..1], y in [0.5..1]
    for i=1, num_Q1 do
        local cx_q1 = 0.5 + 0.5*math.random()
        local cy_q1 = 0.5 + 0.5*math.random()
        local r_q1  = radius_min + (radius_max - radius_min)*math.random()

        -- replicate into Q2, Q3, Q4
        local cx_q2, cy_q2 = rotate_90_deg(cx_q1, cy_q1, px, py)      -- +90
        local cx_q3, cy_q3 = rotate_90_deg(cx_q2, cy_q2, px, py)      -- +180
        local cx_q4, cy_q4 = rotate_90_deg(cx_q3, cy_q3, px, py)      -- +270

        -- store them (including the original Q1 circle)
        table.insert(circles, {cx_q1, cy_q1, r_q1})
        table.insert(circles, {cx_q2, cy_q2, r_q1})
        table.insert(circles, {cx_q3, cy_q3, r_q1})
        table.insert(circles, {cx_q4, cy_q4, r_q1})
    end

    return circles
end

--------------------------------------------------------------------------------
-- 3) Helper to optionally save circles if needed
--------------------------------------------------------------------------------
function save_circles_to_file(fname, circle_array)
    -- circle_array is a table of { {cx,cy,r}, {cx,cy,r}, ... }
    local file = io.open(fname, "w")
    if not file then
        error("Could not open '" .. fname .. "' for writing.")
    end
    for i, cinfo in ipairs(circle_array) do
        local cx = cinfo[1]
        local cy = cinfo[2]
        local r  = cinfo[3]
        file:write(string.format("%.6f,%.6f,%.6f\n", cx, cy, r))
    end
    file:close()
end

--------------------------------------------------------------------------------
-- 4) Find all CSV files in partial_crys_data/
--------------------------------------------------------------------------------
local folder = "partial_crys_data"  -- The folder containing your CSVs
local all_csvs = list_csv(folder)
if #all_csvs == 0 then
    error("No CSV files found in '"..folder.."'. Check folder path or file extension.")
end

--------------------------------------------------------------------------------
-- We'll generate 10 random shapes (arrangements of circles) for each CSV.
-- For each shape, we do the entire frequency sweep from each CSV row.
--------------------------------------------------------------------------------

-- Current date/time for filenames:
local dt = os.date("%Y%m%d_%H%M%S")

-- For reproducibility, pick some base seed or not:
math.randomseed(12345)

-- We'll define the min/max radius for circles
local radius_min = 0.05
local radius_max = 0.25

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
    for shape_idx = 1,10 do
        -- Generate new circle arrangement
        local circles = generate_c4_cylinders(radius_min, radius_max)

        -- Optionally save them
        local shape_fname = string.format("shapes/circles_%s_csv%s_shape%d.txt",
            dt, matname, shape_idx)
        save_circles_to_file(shape_fname, circles)

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
            S:AddMaterial("Vacuum",      {1, 0})
            S:AddMaterial(matname,       {eps_real, eps_imag})

            -- Layers
            S:AddLayer("AirAbove",   0,   "Vacuum")
            S:AddLayer("MetaLayer",  0.5, "Vacuum")
            S:AddLayerCopy("AirBelow", 0, "AirAbove")

            -- Place circles
            -- The background is Vacuum, so each circle is matname
            -- We'll do one S:SetLayerPatternCircle(...) per circle
            for _, cinfo in ipairs(circles) do
                local cx = cinfo[1]
                local cy = cinfo[2]
                local r  = cinfo[3]
                S:SetLayerPatternCircle("MetaLayer", matname, {cx, cy}, r)
            end

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
