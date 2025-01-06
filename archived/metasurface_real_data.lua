--------------------------------------------------------------------------------
-- metasurface_real_data.lua
--
-- Fix for: "NewInterpolator: Table must be of length 2 or more."
-- We ensure that the frequency array is in ascending order when adding points.
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- 0) Maxwell–Garnett partial crystallization helpers (unchanged)
--------------------------------------------------------------------------------
local function c_add(a, b)
    return { re = a.re + b.re, im = a.im + b.im }
end
local function c_sub(a, b)
    return { re = a.re - b.re, im = a.im - b.im }
end
local function c_mul(a, b)
    return {
        re = a.re*b.re - a.im*b.im,
        im = a.re*b.im + a.im*b.re
    }
end
local function c_div(a, b)
    local denom = b.re*b.re + b.im*b.im
    return {
        re = (a.re*b.re + a.im*b.im)/denom,
        im = (a.im*b.re - a.re*b.im)/denom
    }
end

local function mg_transform(eps)
    local one = {re=1, im=0}
    local two = {re=2, im=0}
    local top = c_sub(eps, one)   
    local bot = c_add(eps, two)   
    return c_div(top, bot)
end

local function mg_invtransform(L)
    local one = {re=1, im=0}
    local two = {re=2, im=0}
    local numerator   = c_add(one, c_mul(two, L))
    local denominator = c_sub(one, L)
    return c_div(numerator, denominator)
end

local function mg_effective(eps_a, eps_c, C)
    local La = mg_transform(eps_a)
    local Lc = mg_transform(eps_c)
    local sumL = {
        re = C*Lc.re + (1 - C)*La.re,
        im = C*Lc.im + (1 - C)*La.im
    }
    return mg_invtransform(sumL)
end

local function complex_sqrt(z)
    local mag = math.sqrt(z.re*z.re + z.im*z.im)
    local s   = math.sqrt(0.5*(mag + z.re))
    local halfDiff = 0.5*(mag - z.re)
    if halfDiff < 0 then halfDiff = 0 end
    local t   = math.sqrt(halfDiff)
    if z.im >= 0 then
        return { re = s, im = t }
    else
        return { re = s, im = -t }
    end
end

local function make_epsilon(n, k)
    -- eps = (n + i*k)^2 = (n^2 - k^2) + i(2nk)
    return { re = n*n - k*k, im = 2*n*k }
end

--------------------------------------------------------------------------------
-- 1) Real GSST n,k data in wavelength domain. Then sorted by freq ascending.
--------------------------------------------------------------------------------
-- We'll do the final "AddPoint()" calls only after sorting freq in ascending order.
--------------------------------------------------------------------------------

-- Example data arrays (shortened for clarity here). 
-- In practice, copy all your points.

local wavelengths_crys = {
    1.039549192, 1.075165591, 1.110781991, 1.14639839, 1.182014789, 1.217631188,
    1.253247588, 1.288863987, 1.324480386, 1.360096785
    -- ... (add all your points!)
}
local n_crys_vals = {
    3.588763302, 3.553766089, 3.518768876, 3.490771105, 3.462773335, 3.441775007,
    3.420776679, 3.399778351, 3.385779466, 3.37178058
    -- ... (matching length!)
}

-- For k_crys, we might need to re-interpolate if its λ array is slightly different.
local k_crys_wavelengths = {
    1.03598758, 1.071602851, 1.107218122, 1.142833393, 1.178448663, 1.214063934,
    1.249679205, 1.285294476, 1.320909747, 1.356525017
    -- ... 
}
local k_crys_vals = {
    1.110339521, 1.02657927,  0.947007032, 0.871622806, 0.808802618, 0.74598243,
    0.687350254, 0.632906091, 0.582649941, 0.536581803
    -- ...
}

local wavelengths_amor = {
    1.039549192, 1.075165591, 1.110781991, 1.14639839, 1.182014789, 1.217631188,
    1.253247588, 1.288863987, 1.324480386, 1.360096785
    -- ...
}
local n_amor_vals = {
    5.338623961, 5.331624519, 5.324625076, 5.310626191, 5.296627306, 5.28262842,
    5.268629535, 5.247631207, 5.226632879, 5.198635109
    -- ...
}

local k_amor_wavelengths = {
    0.996810782, 1.032426053, 1.068041324, 1.103656595, 1.139271866, 1.174887136,
    1.210502407, 1.246117678, 1.281732949, 1.317348219
    -- ...
}
local k_amor_vals = {
    0.075900423, 0.063336385, 0.050772347, 0.034020297, 0.025644272, 0.017268247,
    0.008892222, 0.008892222, 0.004704209, 0.000516197
    -- ...
}

--------------------------------------------------------------------------------
-- 2) We must ensure freq is in ascending order. So we:
--    (A) Build a table of {freq, val}
--    (B) Sort it
--    (C) Then add to S4.NewInterpolator
--------------------------------------------------------------------------------
function linear_interpolate(wv_in, kv_in, wv_target)
    if #wv_in ~= #kv_in then
        error("k data mismatch!")
    end
    local out = {}
    for _, wvt in ipairs(wv_target) do
        if wvt <= wv_in[1] then
            table.insert(out, kv_in[1])
        elseif wvt >= wv_in[#wv_in] then
            table.insert(out, kv_in[#kv_in])
        else
            local found = false
            for j=1,#wv_in-1 do
                local x1 = wv_in[j]
                local x2 = wv_in[j+1]
                if (wvt>=x1) and (wvt<=x2) then
                    local frac = (wvt - x1)/(x2 - x1)
                    local y1   = kv_in[j]
                    local y2   = kv_in[j+1]
                    local val  = y1 + frac*(y2 - y1)
                    table.insert(out, val)
                    found = true
                    break
                end
            end
            if not found then
                -- fallback
                table.insert(out, kv_in[#kv_in])
            end
        end
    end
    return out
end

-- Create arrays k_crys_aligned, k_amor_aligned
local k_crys_aligned = linear_interpolate(k_crys_wavelengths, k_crys_vals, wavelengths_crys)
local k_amor_aligned = linear_interpolate(k_amor_wavelengths, k_amor_vals, wavelengths_amor)

-- We'll gather them in ascending freq.
function build_sorted_data(wv_arr, n_arr, k_arr)
    local tmp = {}
    for i=1,#wv_arr do
        local lam  = wv_arr[i]
        local freq = 1.0 / lam  -- µm^-1
        table.insert(tmp, {
            freq = freq,
            nval = n_arr[i],
            kval = k_arr[i]
        })
    end
    -- sort by ascending freq
    table.sort(tmp, function(a,b) return a.freq < b.freq end)
    return tmp
end

local crys_data = build_sorted_data(wavelengths_crys, n_crys_vals, k_crys_aligned)
local amor_data = build_sorted_data(wavelengths_amor, n_amor_vals, k_amor_aligned)

-- Now we create S4 interpolators with the sorted data
nCrys = S4.NewInterpolator('cubic hermite spline', {})
kCrys = S4.NewInterpolator('cubic hermite spline', {})
nAmor = S4.NewInterpolator('cubic hermite spline', {})
kAmor = S4.NewInterpolator('cubic hermite spline', {})

for i, row in ipairs(crys_data) do
    nCrys:AddPoint(row.freq, {row.nval})
    kCrys:AddPoint(row.freq, {row.kval})
end
for i, row in ipairs(amor_data) do
    nAmor:AddPoint(row.freq, {row.nval})
    kAmor:AddPoint(row.freq, {row.kval})
end

--------------------------------------------------------------------------------
-- 3) Get partial n,k from freq
--------------------------------------------------------------------------------
local C_fixed = 0.5
function get_partial_nk(freq)
    local nc = nCrys:Get(freq)[1]
    local kc = kCrys:Get(freq)[1]
    local na = nAmor:Get(freq)[1]
    local ka = kAmor:Get(freq)[1]
    local eps_c = make_epsilon(nc, kc)
    local eps_a = make_epsilon(na, ka)
    local eps_eff = mg_effective(eps_a, eps_c, C_fixed)
    local sq     = complex_sqrt(eps_eff)
    return sq.re, sq.im
end

--------------------------------------------------------------------------------
-- 4) Same ring geometry as before
--------------------------------------------------------------------------------
function generate_random_polygon(N, base_radius, rand_amt)
    local verts = {}
    local two_pi = 2*math.pi
    for i=1,N do
        local angle = two_pi*(i-1)/N
        local r = base_radius + rand_amt*(2*math.random() - 1)
        local x = r*math.cos(angle)
        local y = r*math.sin(angle)
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
        file:write(string.format("%.6f,%.6f\n", polygon[i], polygon[i+1]))
    end
    file:close()
end

math.randomseed(1234)
local outer_poly = generate_random_polygon(100, 0.3, 0.03)
local inner_poly = generate_random_polygon(100, 0.15, 0.02)

save_polygon_to_file("shapes/fixed_outer.txt", outer_poly)
save_polygon_to_file("shapes/fixed_inner.txt", inner_poly)

--------------------------------------------------------------------------------
-- 5) For freq=0.3..1.0, do partial n,k and run S4
--------------------------------------------------------------------------------
for freq=0.3,1.0,0.1 do
    local n_eff, k_eff = get_partial_nk(freq)
    local eps_real = n_eff*n_eff - k_eff*k_eff
    local eps_imag = 2*n_eff*k_eff

    local S = S4.NewSimulation()
    S:SetLattice({1,0},{0,1})
    S:SetNumG(40)

    S:AddMaterial("Vacuum", {1,0})
    S:AddMaterial("PartialGSST", {eps_real, eps_imag})

    S:AddLayer("AirAbove",   0, "Vacuum")
    S:AddLayer("MetaLayer",  0.5, "Vacuum")
    S:AddLayerCopy("AirBelow", 0, "AirAbove")

    -- Paint
    S:SetLayerPatternPolygon("MetaLayer","PartialGSST",{0,0},0,outer_poly)
    S:SetLayerPatternPolygon("MetaLayer","Vacuum",{0,0},0,inner_poly)

    -- Incidence
    S:SetExcitationPlanewave({0,0},{1,0},{0,0})
    S:SetFrequency(freq)

    local fwdA, backA = S:GetPoyntingFlux("AirAbove",0)
    local R = backA
    local fwdB, backB = S:GetPoyntingFlux("AirBelow",0)
    local T = fwdB
    print(string.format("freq=%.3f  (n=%.3f, k=%.3f)  R=%.4f  T=%.4f  R+T=%.4f",
        freq, n_eff, k_eff, R, T, R+T))
end

