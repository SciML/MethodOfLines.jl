struct PeriodicMap{hasperiodic}
    map
end

function PeriodicMap(bmap, v)
    map = Dict([operation(u) => Dict([x => isperiodic(bmap, u, x) for x in all_ivs(v)]) for u in v.ū])
    vals = reduce(vcat, collect.(values.(collect(values(map)))))
    hasperiodic = Val(any(p -> p isa Val{true}, vals))
    return PeriodicMap{hasperiodic}(map)
end
