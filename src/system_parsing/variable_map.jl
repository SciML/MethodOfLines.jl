struct VariableMap
    ū
    x̄
    time
    intervals
    args
    x2i
    i2x
end

function VariableMap(domain, ū, x̄, discretization)
    intervals = Dict(map(x̄) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        x => (DomainSets.infimum(xdomain.domain), DomainSets.supremum(xdomain.domain))
    end)
    time = discretization.time
    nspace = length(x̄)
    args = [operation(u) => arguments(u) for u in ū]
    x̄2dim = [x̄[i] => i for i in 1:nspace]
    dim2x̄ = [i => x̄[i] for i in 1:nspace]
    return VariableMap(ū, x̄, time, Dict(intervals), Dict(args), Dict(x̄2dim), Dict(dim2x̄))
end

params(u, v::VariableMap) = s.args[operation(u)]

all_ivs(v::VariableMap) = vcat(v.x̄, v.time)

depvar(u, v::VariableMap) = operation(u)(v.args[operation(u)]...)

x2i(v::VariableMap, u, x) = findfirst(isequal(x), remove(v.args[operation(u)], v.time))

@inline function axiesvals(v::DiscreteSpace{N,M,G}, u_, x_, I) where {N,M,G}
    u = depvar(u_, v)
    map(params(u, v)) do x
        if isequal(x, x_)
            x => (I[x2i(s, u, x)] == 1 ? v.intervals[x][1] : v.intervals[x][2])
        else
            x => s.grid[x][I[x2i(s, u, x)]]
        end
    end
end
