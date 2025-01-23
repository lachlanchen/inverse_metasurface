--------------------------------------------------------------------------------
-- metasurface_batch_multi_shapes.lua
-- Reads all *.csv files in partial_crys_data/ and for each row in each file,
-- builds an S4 simulation, measures R,T. 
-- Now we generate 10 different random shapes for each crystallization (CSV),
-- each shape is a single cylinder in the 1st quadrant (Q1),
-- replicated by C4 symmetry => 4 total cylinders.
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
            header = false  -- skip the first line (header row)
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
-- 2) Generate a single cylinder in quadrant #1, replicate by C4 around (0.5,0.5)
--    Quadrant #1 is the region [0.5..1] in x, [0.5..1] in y.
--    We pick the radius in [r_min, r_max].
--    We pick the center so the circle doesn't overflow Q1.
--    Then replicate to Q2, Q3, Q4 by +90°, +180°, +270° rotations.
--------------------------------------------------------------------------------
function generate_c4_single_cylinder(r_min, r_max)
    -- pivot for rotation is the center of the cell
    local px, py = 0.5, 0.5

    -- define local helper to rotate (x,y) +90° about (px,py)
    local function rotate_90_deg(x, y, px, py)
        local dx = x - px
        local dy = y - py
        -- standard 2D rotation by +90°: (x,y)->(-y,x)
        local rx = -dy
        local ry = dx
        return rx + px, ry + py
    end

    -- random radius
    local r = r_min + (r_max - r_min)*math.random()

    -- quadrant #1 is the region [0.5..1] x [0.5..1], so width=0.5
    -- to avoid overflow, we place the center in:
    --   x in [0.5 + r, 1 - r]
    --   y in [0.5 + r, 1 - r]
    local min_xy = 0.5 + r
    local max_xy = 1.0 - r
    if min_xy > max_xy then
        -- if the random radius is so large that it can't fit, we'll clamp
        -- or you can forcibly reduce r to fit. Let's just clamp it:
        r = 0.5*(0.5) -- i.e. forced smaller, or just set r=0.25
        min_xy = 0.5 + r
        max_xy = 1.0 - r
    end

    local cx_q1 = min_xy + (max_xy - min_xy)*math.random()
    local cy_q1 = min_xy + (max_xy - min_xy)*math.random()

    -- replicate
    local cx_q2, cy_q2 = rotate_90_deg(cx_q1, cy_q1, px, py)
    local cx_q3, cy_q3 = rotate_90_deg(cx_q2, cy_q2, px, py) -- +180
    local cx_q4, cy_q4 = rotate_90_deg(cx_q3, cy_q3, px, py) -- +270

    -- return table of 4 circles {cx,cy,r}
    local circles = {}
    table.insert(circles, {cx_q1, cy_q1, r})
    table.insert(circles, {cx_q2, cy_q2, r})
    table.insert(circles, {cx_q3, cy_q3, r})
    table.insert(circles, {cx_q4, cy_q4, r})

    return circles
end

--------------------------------------------------------------------------------
-- 3) Helper to save circles in shapes/<dt>/circles_csv<mat>_shapeX.txt
--------------------------------------------------------------------------------
function save_circles_to_file(fname, circle_array)
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
-- We'll generate 10 shapes for each CSV. For each shape, we do the entire freq 
-- sweep from each CSV row.
--------------------------------------------------------------------------------

-- Current date/time for subfolder:
local dt = os.date("%Y%m%d_%H%M%S")

-- Create a subfolder in "shapes/" for this run
os.execute('mkdir -p "shapes/'..dt..'"')

-- For reproducibility or random:
math.randomseed(12345)

-- radius range for half-quadrant
local r_min = 0.125  -- 1/4 of 0.5
local r_max = 0.25   -- 1/2 of 0.5

--------------------------------------------------------------------------------
-- 5) Loop over each CSV (crystallization)
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
    -- For each CSV, generate 10 shapes and run S4
    ----------------------------------------------------------------------------
    for shape_idx = 1,10 do
        -- Generate the four circles
        local circles = generate_c4_single_cylinder(r_min, r_max)

        -- Save them in shapes/<dt>/
        local shape_fname = string.format("shapes/%s/circles_csv%s_shape%d.txt",
            dt, matname, shape_idx)
        save_circles_to_file(shape_fname, circles)

        -- Now for each row in the CSV, run the simulation
        for i, row in ipairs(data) do
            local lam = row.wavelength   -- in micrometers
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

            -- Place circles (the background is Vacuum)
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

            -- Print results for parsing:
            print(string.format(
                "%s | shape=%d, row=%d, λ=%.3f µm, freq=%.3f, (n=%.3f, k=%.3f) => R=%.4f, T=%.4f, R+T=%.4f",
                csvfile, shape_idx, i, lam, freq, n, k, -R, T, -R+T
            ))
        end
    end
end
