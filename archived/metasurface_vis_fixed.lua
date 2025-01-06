--------------------------------------------------------------------------------
-- metasurface_fixed_shape.lua
--
-- We want exactly one shape (outer + inner polygons), used for all frequencies.
-- At each frequency, only n,k changes. The shape does not change.
--
-- Steps:
--   1) Generate outer_poly and inner_poly once.
--   2) (Optionally) save those polygons to "shapes/fixed_outer.txt"
--      and "shapes/fixed_inner.txt".
--   3) For each freq in 0.3..1.0, randomize n,k, create new S4 simulation,
--      and paint the same polygons.
--   4) Compute reflection (R) and transmission (T).
--------------------------------------------------------------------------------

-- math.randomseed(12345)  -- optional

------------------------------------------------------------------------------
-- 1) Generate a single shape (outer + inner polygons) once
------------------------------------------------------------------------------
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

-- Outer ring polygon
local outer_poly = generate_random_polygon(100, 0.30, 0.03)

-- Inner hole polygon
local inner_poly = generate_random_polygon(100, 0.15, 0.02)

-- (Optionally) save them to files once (so you can visualize them later).
local out_outer = "shapes/fixed_outer.txt"
local out_inner = "shapes/fixed_inner.txt"
save_polygon_to_file(out_outer, outer_poly)
save_polygon_to_file(out_inner, inner_poly)

------------------------------------------------------------------------------
-- 2) Random n,k function (range: n in [1..3], k in [0..0.2])
------------------------------------------------------------------------------
function random_nk()
    local n = 1.0 + 2.0 * math.random()
    local k = 0.2 * math.random()
    return n, k
end

------------------------------------------------------------------------------
-- 3) Frequency loop. For each freq, only n,k changes. The shape is fixed.
------------------------------------------------------------------------------
for freq = 0.3,1.0,0.1 do
    
    -- (a) Random n,k
    local n, k = random_nk()
    local eps_real = n*n - k*k
    local eps_imag = 2*n*k
    
    -- (b) Make a NEW S4 simulation
    S = S4.NewSimulation()
    S:SetLattice({1,0}, {0,1})
    S:SetNumG(40)
    
    -- (c) Define materials
    S:AddMaterial("Vacuum",         {1,0})
    S:AddMaterial("RandomMaterial", {eps_real, eps_imag})
    
    -- (d) Layers
    S:AddLayer("AirAbove",   0,       "Vacuum")
    S:AddLayer("MetaLayer",  0.5,     "Vacuum") -- background=Vacuum
    S:AddLayerCopy("AirBelow", 0,     "AirAbove")
    
    -- (e) Paint the same ring shape each iteration
    --     Outer polygon => RandomMaterial
    S:SetLayerPatternPolygon(
        "MetaLayer",
        "RandomMaterial",
        {0,0},   -- center
        0,       -- tilt angle
        outer_poly
    )
    
    --     Inner polygon => Vacuum
    S:SetLayerPatternPolygon(
        "MetaLayer",
        "Vacuum",
        {0,0},
        0,
        inner_poly
    )
    
    -- (f) Incidence
    S:SetExcitationPlanewave({0,0},{1,0},{0,0})
    S:SetFrequency(freq)
    
    -- (g) Reflection & Transmission
    local fwdA, backA = S:GetPoyntingFlux("AirAbove", 0)
    local R = backA
    local fwdB, backB = S:GetPoyntingFlux("AirBelow", 0)
    local T = fwdB
    
    -- (h) Print
    print(string.format("freq=%.2f  n=%.2f  k=%.2f  R=%.4f  T=%.4f  R+T=%.4f",
        freq, n, k, R, T, R+T))
end

