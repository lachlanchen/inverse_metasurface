--------------------------------------------------------------------------------
-- metasurface_strict_progress.lua
--
-- This script duplicates the geometry & steps of "metasurface_progress.lua":
--   - For each row of partial_crys_data/partial_crys_C{c}.csv, we:
--       1) Build a fresh S4 simulation
--       2) Use {1,0},{0,1} lattice, G=40
--       3) A 0.5 um-thick layer in vacuum, patterned by polygon
--       4) AddMaterial(...) with the row's (n, k)
--       5) Set normal-incidence planewave => measure R, T
--   - The same pattern: A half-micron of high index in air typically yields ~1 reflectivity.
--
-- USAGE EXAMPLE:
--   S4 -a "--vertices-str=0.162926,0.189418;-0.189418,0.162926;-0.162926,-0.189418;0.189418,-0.162926 -c=0.0 out_csv=myout.csv" metasurface_strict_progress.lua
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
-- 0) Parse S4.arg => shape_str, c_str, out_csv
--------------------------------------------------------------------------------
local arg_str = S4.arg or ""
local shape_str = nil
local c_str     = nil
local out_csv   = nil

for token in arg_str:gmatch("%S+") do
    local k,v = token:match("^([^=]+)=(.*)$")
    if k and v then
        if k=="--vertices-str" then
            shape_str = v
        elseif k=="-c" then
            c_str = v
        elseif k=="out_csv" then
            out_csv = v
        end
    end
end

if not shape_str or not c_str then
    io.stderr:write("[metasurface_strict_progress.lua] ERROR: Must pass --vertices-str=... and -c=...\n")
    io.stderr:write("Example: -a \"--vertices-str=0.1,0.2;0.2,0.1 -c=0.0 out_csv=foo.csv\"\n")
    return
end

if not out_csv or out_csv=="" then
    out_csv = string.format("spectrum_c%s.csv", c_str)
end

io.stderr:write(string.format("[metasurface_strict_progress.lua] shape_str='%s'\n", shape_str))
io.stderr:write(string.format("[metasurface_strict_progress.lua] c=%s => partial_crys_C%s.csv\n", c_str, c_str))
io.stderr:write(string.format("[metasurface_strict_progress.lua] out_csv='%s'\n", out_csv))

--------------------------------------------------------------------------------
-- 1) Load partial_crys_data/partial_crys_C{c_str}.csv
--------------------------------------------------------------------------------
local function read_csv(csvfile)
    local f,err = io.open(csvfile, "r")
    if not f then
        error("Cannot open CSV '"..csvfile.."': "..(err or ""))
    end

    local head = f:read("*l")
    if not head then error("Empty CSV: "..csvfile) end
    head = head:gsub("^[\239\187\191]",""):gsub("[\r\n]+$","")

    -- find columns "Wavelength_um", "n_eff", "k_eff"
    local idx_lam, idx_n, idx_k = nil,nil,nil
    do
        local cols = {}
        local i=1
        for c in head:gmatch("[^,]+") do
            cols[i] = c:lower()
            i=i+1
        end
        for i,name in ipairs(cols) do
            if     name:match("wavelength_um") then idx_lam=i
            elseif name=="n_eff" then idx_n=i
            elseif name=="k_eff" then idx_k=i
            end
        end
    end

    if not(idx_lam and idx_n and idx_k) then
        error("CSV must have 'Wavelength_um,n_eff,k_eff' columns. Found: "..head)
    end

    local lines={}
    while true do
        local line = f:read("*l")
        if not line then break end
        if line~="" then
            local arr = {}
            local i=1
            for c in line:gmatch("[^,]+") do
                arr[i] = c
                i=i+1
            end
            local lam_s = arr[idx_lam]
            local n_s   = arr[idx_n]
            local k_s   = arr[idx_k]
            local lam   = tonumber(lam_s)
            local nr    = tonumber(n_s)
            local ki    = tonumber(k_s)
            if lam and nr and ki then
                table.insert(lines, {lam=lam, n=nr, k=ki})
            end
        end
    end
    f:close()
    return lines
end

local csvpath = string.format("partial_crys_data/partial_crys_C%s.csv", c_str)
local data_rows = read_csv(csvpath)
io.stderr:write(string.format("[metasurface_strict_progress.lua] Loaded %d rows from %s\n",
    #data_rows, csvpath))

--------------------------------------------------------------------------------
-- 2) Parse shape_str => polygon array
--------------------------------------------------------------------------------
local shape_pts = {}
for seg in shape_str:gmatch("[^;]+") do
    local xs, ys = seg:match("([^,]+),([^,]+)")
    if xs and ys then
        local xx = tonumber(xs)
        local yy = tonumber(ys)
        if xx and yy then
            table.insert(shape_pts, {xx, yy})
        end
    end
end

io.stderr:write(string.format("[metasurface_strict_progress.lua] Polygon with %d vertices.\n",
    #shape_pts))

local polygon_xy = {}
for i,pt in ipairs(shape_pts) do
    table.insert(polygon_xy, pt[1])
    table.insert(polygon_xy, pt[2])
end

--------------------------------------------------------------------------------
-- 3) Open out_csv => columns: wavelength_um, R, T
--------------------------------------------------------------------------------
local fout,er2 = io.open(out_csv,"w")
if not fout then
    error("Cannot open out_csv='"..out_csv.."' for writing: "..tostring(er2))
end
fout:write("Wavelength_um,R,T\n")

--------------------------------------------------------------------------------
-- 4) For each row => build S4 => measure => write
--------------------------------------------------------------------------------
for i,row in ipairs(data_rows) do
    local lam  = row.lam
    local nr   = row.n
    local ki   = row.k
    local freq = 1.0 / lam

    -- Convert n,k => (eps_r, eps_i)
    local eps_r = nr*nr - ki*ki
    local eps_i = 2 * nr * ki

    ----------------------------------------------------------------------------
    -- EXACT geometry from metasurface_progress.lua:
    --   New S4 simulation each row
    --   Lattice(1,0;0,1), #G=40
    --   Material:
    --       "Vacuum"
    --       "MyMat"
    --   Layers: AirAbove(0 thickness), MetaLayer(0.5, Vacuum), AirBelow(0 thickness)
    --   SetLayerPatternPolygon( "MetaLayer", "MyMat", ... )
    --   normal incidence planewave => measure R,T
    ----------------------------------------------------------------------------
    local S = S4.NewSimulation()
    S:SetLattice({1,0},{0,1})
    S:SetNumG(40)

    S:AddMaterial("Vacuum",{1,0})
    S:AddMaterial("MyMat",{1,0})  -- placeholder

    S:AddLayer("AirAbove",  0,   "Vacuum")
    S:AddLayer("MetaLayer", 0.5, "Vacuum")
    S:AddLayerCopy("AirBelow", 0, "AirAbove")

    -- Pattern
    S:SetLayerPatternPolygon("MetaLayer", "MyMat", {0,0}, 0, polygon_xy)

    -- Overwrite "MyMat" with row's eps
    S:AddMaterial("MyMat",{eps_r, eps_i})

    -- Normal-incidence planewave
    S:SetExcitationPlanewave({0,0},{1,0},{0,0})

    S:SetFrequency(freq)

    local fwdA, backA = S:GetPoyntingFlux("AirAbove",  0)
    local fwdB, backB = S:GetPoyntingFlux("AirBelow",  0)

    local R = -backA
    local T = fwdB

    fout:write(string.format("%.6f,%.6g,%.6g\n", lam, R, T))

    -- Also print to console
    print(string.format(
      "Row=%3d => λ=%.6f µm  (n=%.4f, k=%.4f) => R=%.6g, T=%.6g",
      i, lam, nr, ki, R, T
    ))
end

fout:close()
io.stderr:write(string.format("[metasurface_strict_progress.lua] Done. Wrote %s\n", out_csv))

