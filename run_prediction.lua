--------------------------------------------------------------------------------
-- run_polygon.lua
--
-- Usage example:
--   ../build/S4 -a "partial_crys_data/partial_crys_C0.55.csv" run_polygon.lua
--------------------------------------------------------------------------------

-- 1) Grab the CSV filename from S4.arg
local arg_str = S4.arg
if not arg_str then
    error("Please provide a CSV filename via S4.arg. Example:\n  ../build/S4 -a \"partial_crys_data/partial_crys_C0.55.csv\" run_polygon.lua")
end

local csvfile = arg_str

-- 2) Define the polygon from your pres>0.5 lines
--    We'll put them in a single array: {x1,y1, x2,y2, x3,y3, ...}
--    Based on your test_pred.txt:
local polygon_3 = {
    0.267, 0.069,
    0.189, 0.193,
    0.107, 0.271
    -- i=3 was pres=0.369 => skip
}

-- 3) Helper to read a CSV. We'll skip the header line.
function read_csv_rows(fname)
    local t = {}
    local f,err = io.open(fname, "r")
    if not f then
        error("Could not open CSV: "..fname.." => "..(err or ""))
    end
    local header = true
    for line in f:lines() do
        if header then
            header = false
        else
            local lam_str, n_str, k_str = line:match("([^,]+),([^,]+),([^,]+)")
            if lam_str and n_str and k_str then
                local lam = tonumber(lam_str)
                local nn  = tonumber(n_str)
                local kk  = tonumber(k_str)
                table.insert(t, {wavelength=lam, n=nn, k=kk})
            end
        end
    end
    f:close()
    return t
end

-- 4) Read the CSV
local data = read_csv_rows(csvfile)
print("Loaded", #data, "rows from", csvfile)

-- 5) We'll do a loop over each row, building a simulation, retrieving R & T
for i,row in ipairs(data) do
    local lam = row.wavelength  -- micrometers
    local n   = row.n
    local k   = row.k

    -- freq = 1/lambda
    local freq = 1.0/lam

    -- Build complex epsilon = (n + i k)^2 => Re/Im
    -- But in S4, we can pass {eps_real, eps_imag} directly.
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k

    -- Create a fresh simulation
    local S = S4.NewSimulation()
    S:SetLattice({1,0}, {0,1})
    S:SetNumG(40)

    -- Add materials
    S:AddMaterial("Vacuum", {1,0})
    S:AddMaterial("MyGSST", {eps_real, eps_imag})

    -- Layers
    S:AddLayer("AirAbove",   0, "Vacuum")
    S:AddLayer("Slab",       0.5, "Vacuum") 
    S:AddLayerCopy("AirBelow", 0, "AirAbove")

    -- Add polygon pattern to "Slab" layer
    S:SetLayerPatternPolygon("Slab",
                             "MyGSST",   -- fill material
                             {0,0},      -- center
                             0,          -- tilt angle
                             polygon_3)

    -- Incidence
    S:SetExcitationPlanewave(
        {0,0},  -- incidence angles
        {1,0},  -- s-polarization amplitude, phase
        {0,0})  -- p-polarization amplitude, phase

    -- Frequency
    S:SetFrequency(freq)

    -- Poynting flux
    local fwdA, backA = S:GetPoyntingFlux("AirAbove",  0)
    local fwdB, backB = S:GetPoyntingFlux("AirBelow",  0)

    local R = -backA
    local T = fwdB

    print(string.format(
        "Row=%d lam=%.3fµm freq=%.3f -> n=%.3f,k=%.3f => R=%.4f, T=%.4f, R+T=%.4f",
        i, lam, freq, n, k, R, T, R+T
    ))
end

