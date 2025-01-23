--------------------------------------------------------------------------------
-- metasurface_check.lua
--
-- Usage example:
--   S4 -a "partial_csv=partial_crys_data/partial_crys_C0.0.csv shape_str=0.1,0.2;0.2,0.1 out_csv=out_c0.0.csv" metasurface_check.lua
--
-- This script:
--   1) Reads 'partial_csv' (a CSV with header "Wavelength_um,n_eff,k_eff").
--   2) Parses the polygon from 'shape_str' => "x1,y1;x2,y2;...".
--   3) Builds S4 simulation with that shape, with a placeholder material "MyMat".
--   4) For each line (lam,n,k), sets the permittivity => calls S4's GetPoyntingFlux 
--      => writes (wavelength_um, R, T) to 'out_csv'.
-- 
-- Debugging prints are emitted to stderr to help verify input data.
--------------------------------------------------------------------------------

-- 0) Parse S4.arg
local arg_str = S4.arg or ""
local partial_csv = nil
local shape_str   = nil
local out_csv     = nil

do
    local tokens = {}
    for w in arg_str:gmatch("%S+") do
        table.insert(tokens, w)
    end
    for _,token in ipairs(tokens) do
        local k,v = token:match("([^=]+)=(.*)")
        if k and v then
            if k=="partial_csv" then
                partial_csv = v
            elseif k=="shape_str" then
                shape_str   = v
            elseif k=="out_csv" then
                out_csv     = v
            end
        end
    end
end

if (not partial_csv) or (not shape_str) or (not out_csv) then
    io.stderr:write("[metasurface_check.lua] ERROR: usage requires partial_csv=..., shape_str=..., out_csv=...\n")
    return
end

io.stderr:write(string.format("[metasurface_check.lua] partial_csv='%s'\n", partial_csv))
io.stderr:write(string.format("[metasurface_check.lua] shape_str='%s'\n", shape_str))
io.stderr:write(string.format("[metasurface_check.lua] out_csv='%s'\n", out_csv))

--------------------------------------------------------------------------------
-- 1) Read partial_csv => data_lines
--------------------------------------------------------------------------------
local function read_partial_csv(csvfile)
    local f,er = io.open(csvfile, "r")
    if not f then
        error("Cannot open partial_csv="..csvfile.."; err="..tostring(er))
    end

    local header_line = f:read("*l")
    if not header_line then
        error("Empty CSV: "..csvfile)
    end
    -- remove possible BOM, trailing \r, etc.
    header_line = header_line:gsub("^[\239\187\191]", "")
    header_line = header_line:gsub("[\r\n]+$","")
    header_line = header_line:gsub("^%s+",""):gsub("%s+$","")

    -- parse header
    local parts = {}
    for w in header_line:gmatch("[^,]+") do
        table.insert(parts, w:lower())
    end

    local col_map = {}
    for i, name in ipairs(parts) do
        col_map[name] = i
    end

    if not (col_map["wavelength_um"] and col_map["n_eff"] and col_map["k_eff"]) then
        error("partial_csv must have columns: Wavelength_um,n_eff,k_eff. Found: "..header_line)
    end

    local lines = {}
    while true do
        local line = f:read("*l")
        if not line then break end
        line = line:gsub("^[\239\187\191]",""):gsub("[\r\n]+$",""):gsub("^%s+",""):gsub("%s+$","")
        if line ~= "" then
            local arr = {}
            for w in line:gmatch("[^,]+") do
                table.insert(arr, w)
            end
            local lam_s = arr[col_map["wavelength_um"]]
            local n_s   = arr[col_map["n_eff"]]
            local k_s   = arr[col_map["k_eff"]]
            local lam   = tonumber(lam_s)
            local nr    = tonumber(n_s)
            local ki    = tonumber(k_s)
            if lam and nr and ki then
                table.insert(lines, {lam=lam, n=nr, k=ki})
            end
        end
    end
    f:close()

    table.sort(lines, function(a,b) return a.lam < b.lam end)
    return lines
end

local data_lines = read_partial_csv(partial_csv)
io.stderr:write(string.format("[metasurface_check.lua] Loaded %d lines from %s\n", #data_lines, partial_csv))

--------------------------------------------------------------------------------
-- 2) Parse shape_str => polygon array
--------------------------------------------------------------------------------
local shape_xy = {}
do
    for seg in shape_str:gmatch("[^;]+") do
        local xs,ys = seg:match("([^,]+),([^,]+)")
        if xs and ys then
            local xx=tonumber(xs)
            local yy=tonumber(ys)
            if xx and yy then
                table.insert(shape_xy, {xx,yy})
            end
        end
    end
end
io.stderr:write(string.format("[metasurface_check.lua] shape_xy has %d vertices\n", #shape_xy))

local polygon_nums = {}
for _,pt in ipairs(shape_xy) do
    table.insert(polygon_nums, pt[1])
    table.insert(polygon_nums, pt[2])
end

--------------------------------------------------------------------------------
-- 3) Open out_csv for writing
--------------------------------------------------------------------------------
local fout,er2 = io.open(out_csv,"w")
if not fout then
    error("Cannot open out_csv="..out_csv.." for writing err="..tostring(er2))
end
fout:write("wavelength_um,R,T\n")

--------------------------------------------------------------------------------
-- 4) Build S4 simulation once
--------------------------------------------------------------------------------
local S = S4.NewSimulation()
S:SetLattice({1,0},{0,1})
S:SetNumG(40)

-- We define two materials: Vacuum and "MyMat" as a placeholder that we will re‐define.
S:AddMaterial("Vacuum",{1,0})
S:AddMaterial("MyMat",{1,0})  -- placeholder

S:AddLayer("AirAbove", 0, "Vacuum")
S:AddLayer("MetaLayer",0.5,"Vacuum")
S:AddLayerCopy("AirBelow",0,"AirAbove")

-- Insert the polygon pattern
S:SetLayerPatternPolygon("MetaLayer","MyMat",{0,0},0, polygon_nums)

-- Plane‐wave excitation from above
S:SetExcitationPlanewave({0,0},{1,0},{0,0})

--------------------------------------------------------------------------------
-- 5) For each (lam, n, k), update MyMat => measure => output
--------------------------------------------------------------------------------
local function update_MyMat(eps_r, eps_i)
    -- Overwrite MyMat with new permittivity
    S:AddMaterial("MyMat", {eps_r, eps_i})
end

for i,row in ipairs(data_lines) do
    local lam = row.lam
    local nr  = row.n
    local ki  = row.k
    local freq = 1/lam
    local eps_r = nr*nr - ki*ki
    local eps_i = 2*nr*ki

    update_MyMat(eps_r, eps_i)
    S:SetFrequency(freq)

    local fwdA, backA = S:GetPoyntingFlux("AirAbove",0)
    local fwdB, backB = S:GetPoyntingFlux("AirBelow",0)
    local R = -backA
    local T = fwdB

    -- Write to CSV
    fout:write(string.format("%.6f,%.6g,%.6g\n", lam, R, T))

    -- Print a few example lines to stderr
    if i <= 3 then
        io.stderr:write(string.format(
          "[metasurface_check.lua] row %d => lam=%.6f, n=%.4f, k=%.4f => R=%.6g, T=%.6g\n",
          i, lam, nr, ki, R, T
        ))
    end
end

fout:close()
io.stderr:write("[metasurface_check.lua] Done. Wrote "..out_csv.."\n")

