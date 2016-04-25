local utils = {}
-- string split
function utils.mysplit(inputstr, sep)
    if sep == nil then
        sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
        t[i] = str:lower()
        i = i + 1
    end
    return t
end

-- count number
function utils.len(t)
    local count = 0
    for _ in pairs(t) do count = count + 1 end
    return count
end

-- shallow copy
function utils.shallowcopy(orig)
    local orig_type = type(orig)
    local copy
    if orig_type == 'table' then
        copy = {}
        for orig_key, orig_value in pairs(orig) do
            copy[orig_key] = orig_value
        end
    else -- number, string, boolean, etc
        copy = orig
    end
    return copy
end

-- concat arrays
function utils.concat(t1, t2)
    local t = utils.shallowcopy(t1)
    for _, v in ipairs(t2) do
        table.insert(t, v)
    end
    return t
end

return utils