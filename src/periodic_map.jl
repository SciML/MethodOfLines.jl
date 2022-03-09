struct PeriodicMap{hasperiodic}
    map
end

function PeriodicMap(bmap, s)
    map = Dict([operation(u) => Dict([x => isperiodic(bmap, u, x) for x in params(u, s)]) for u in keys(bmap)])
    isperiodic = Val(any(p -> p isa Val{true}, reduce(vcat, reduce(vcat, collect.(values.(collect(values(map))))))))
    return PeriodicMap{isperiodic}(map)
end