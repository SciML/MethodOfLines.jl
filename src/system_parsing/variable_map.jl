struct VariableMap
    ū
    x̄
    time
    intervals
    args
    depvar_ops
    x2i
    i2x
end

function VariableMap(eqs, indvars, depvars, domain, discretization)
    time = discretization.time

    depvar_ops = map(u -> operation(u isa Num ? u.val : u), depvars)
    # Get all dependent variables in the correct type
    alldepvars = get_all_depvars(eqs, depvar_ops)
    ū = filter(u -> !any(map(x -> x isa Number, arguments(u))), alldepvars)
    # Get all independent variables in the correct type
    x̄ = remove(collect(filter(x -> !(x isa Number), reduce(union, map(arguments, alldepvars)))), t)

    intervals = Dict(map(x̄) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        x => (DomainSets.infimum(xdomain.domain), DomainSets.supremum(xdomain.domain))
    end)
    time = discretization.time
    nspace = length(x̄)
    args = [operation(u) => arguments(u) for u in ū]
    x̄2dim = [x̄[i] => i for i in 1:nspace]
    dim2x̄ = [i => x̄[i] for i in 1:nspace]
    return VariableMap(depvars, indvars, time, Dict(intervals), Dict(args), depvar_ops, Dict(x̄2dim), Dict(dim2x̄))
end

function update_varmap!(v, newdv)
    push!(v.ū, newdv)
    merge!(v.args, Dict(operation(newdv) => arguments(newdv)))
    push!(v.depvar_ops, operation(unwrap(newdv)))
end


params(u, v::VariableMap) = s.args[operation(u)]

all_ivs(v::VariableMap) = v.time === nothing ? v.x̄ : v.x̄ ∪ [v.time]

depvar(u, v::VariableMap) = operation(u)(v.args[operation(u)]...)

x2i(v::VariableMap, u, x) = findfirst(isequal(x), remove(v.args[operation(u)], v.time))

@inline function axiesvals(v::VariableMap, u_, x_, I)
    u = depvar(u_, v)
    map(params(u, v)) do x
        x => (I[x2i(s, u, x)] == 1 ? v.intervals[x][1] : v.intervals[x][2])
    end
end
