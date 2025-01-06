-- test.lua
if S4.arg == nil then
    print("No arguments passed")
else
    print("Raw S4.arg = " .. S4.arg)
    -- a quick parse if it's '--num-shapes <integer>'
    local _, _, val = string.find(S4.arg, "%-%-num%-shapes%s+(%d+)")
    if val then
        print("Parsed --num-shapes = " .. val)
    else
        print("Could not parse --num-shapes")
    end
end

