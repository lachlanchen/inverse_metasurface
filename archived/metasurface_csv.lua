--------------------------------------------------------------------------------
-- metasurface_C0p4.lua
-- Example: Reads partial_crys_C0.4.csv, which has columns:
--    Wavelength_um, n_eff, k_eff
-- Then, for each row, we do:
--   freq = 1 / (Wavelength_um)
--   Build S4, paint polygons, measure R,T.
--------------------------------------------------------------------------------

-- 1) Read CSV
local csvfile = "partial_crys_C0.4.csv"
local data = {}

do
    local file = io.open(csvfile, "r")
    if not file then error("Could not open '"..csvfile.."'") end
    local header = true
    for line in file:lines() do
        if header then
            -- skip first line
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
end

-- 2) Generate (or read) a single ring shape (same as metasurface_vis_fixed.lua)
function generate_random_polygon(N, base_radius, rand_amt)
    local verts = {}
    local two_pi = 2*math.pi
    for i=1,N do
        local angle = two_pi*(i-1)/N
        local r = base_radius + rand_amt*(2*math.random() - 1)
        local x = r * math.cos(angle)
        local y = r * math.sin(angle)
        table.insert(verts, x)
        table.insert(verts, y)
    end
    return verts
end

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

math.randomseed(12345)  -- to have consistent shape
local outer_poly = generate_random_polygon(100, 0.30, 0.03)
local inner_poly = generate_random_polygon(100, 0.15, 0.02)

-- Optionally save:
save_polygon_to_file("shapes/fixed_outer.txt", outer_poly)
save_polygon_to_file("shapes/fixed_inner.txt", inner_poly)

-- 3) Loop over each row in partial_crys_C0.4.csv
--    freq = 1/lambda, then n_eff, k_eff => eps
for i, row in ipairs(data) do
    local lam = row.wavelength
    local n   = row.n_eff
    local k   = row.k_eff
    local freq = 1.0 / lam  -- S4 uses freq=omega/(2*pi), so if lam is in µm, freq is in 1/µm

    -- Build permittivity
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k

    -- 4) Build S4 simulation
    local S = S4.NewSimulation()
    S:SetLattice({1,0}, {0,1})
    S:SetNumG(40)

    -- Materials
    S:AddMaterial("Vacuum",          {1,0})
    S:AddMaterial("PartialGSST_C0.4", {eps_real, eps_imag})

    -- Layers
    S:AddLayer("AirAbove",   0,       "Vacuum")
    S:AddLayer("MetaLayer",  0.5,     "Vacuum")
    S:AddLayerCopy("AirBelow", 0,     "AirAbove")

    -- Paint the ring
    S:SetLayerPatternPolygon(
        "MetaLayer",
        "PartialGSST_C0.4",
        {0,0},
        0,
        outer_poly
    )
    S:SetLayerPatternPolygon(
        "MetaLayer",
        "Vacuum",
        {0,0},
        0,
        inner_poly
    )

    -- Incidence
    S:SetExcitationPlanewave({0,0}, {1,0}, {0,0})
    S:SetFrequency(freq)

    -- Reflection & Transmission
    local fwdA, backA = S:GetPoyntingFlux("AirAbove", 0)
    local R = backA
    local fwdB, backB = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdB

    -- Print results
    print(string.format(
        "Row=%d, λ=%.3f µm, freq=%.3f, (n=%.3f, k=%.3f)  => R=%.4f, T=%.4f, R+T=%.4f",
        i, lam, freq, n, k, R, T, -R+T
    ))
end

