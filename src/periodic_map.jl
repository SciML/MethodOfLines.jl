struct PeriodicMap{hasperiodic}
    map::Any
end

function PeriodicMap(bmap, s)
    map = Dict([
        operation(u) => Dict([x => isperiodic(bmap, u, x) for x in s.x̄]) for u in s.ū
    ])
    vals = reduce(vcat, collect.(values.(collect(values(map)))))
    hasperiodic = Val(any(p -> p isa Val{true}, vals))
    return PeriodicMap{hasperiodic}(map)
end
