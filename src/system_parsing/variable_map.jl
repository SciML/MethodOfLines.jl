struct VariableMap
    ū
    x̄
    ps
    time
    intervals
    args
    depvar_ops
    x2i
    i2x
end

function VariableMap(eqs, depvars, domain, time, ps = [])
    time = safe_unwrap(time)
    ps = map(ps) do p
        safe_unwrap(p.first)
    end
    depvar_ops = get_ops(depvars)
    # Get all dependent variables in the correct type
    alldepvars = get_all_depvars(eqs, depvar_ops)
    # Filter out boundaries
    ū = filter(u -> !any(x -> x isa Number, arguments(u)), alldepvars)
    # Get all independent variables in the correct type
    allivs = collect(filter(x -> !(x isa Number), reduce(union, map(arguments, alldepvars))))
    x̄ = remove(allivs, time)
    intervals = Dict(map(allivs) do x
        xdomain = domain[findfirst(d -> isequal(x, d.variables), domain)]
        x => (DomainSets.infimum(xdomain.domain), DomainSets.supremum(xdomain.domain))
    end)
    nspace = length(x̄)
    args = [operation(u) => arguments(u) for u in ū]
    x̄2dim = [x̄[i] => i for i in 1:nspace]
    dim2x̄ = [i => x̄[i] for i in 1:nspace]
    return VariableMap(ū, x̄, ps, time, Dict(intervals), Dict(args), depvar_ops, Dict(x̄2dim), Dict(dim2x̄))
end

VariableMap(pdesys::PDESystem, disc::MOLFiniteDifference) = VariableMap(pdesys.eqs, pdesys.dvs, pdesys.domain, disc.time)
VariableMap(pdesys::PDESystem, t) = VariableMap(pdesys.eqs, pdesys.dvs, pdesys.domain, t)

function update_varmap!(v, newdv)
    push!(v.ū, newdv)
    merge!(v.args, Dict(operation(newdv) => arguments(newdv)))
    push!(v.depvar_ops, operation(safe_unwrap(newdv)))
end


ivs(u, v::VariableMap) = remove(v.args[operation(u)], v.time)

Base.ndims(u, v::VariableMap) = length(ivs(u, v))

all_ivs(v::VariableMap) = v.time === nothing ? v.x̄ : v.x̄ ∪ [v.time]

all_ivs(u, v::VariableMap) = v.args[operation(u)]

depvar(u, v::VariableMap) = operation(u)(v.args[operation(u)]...)

x2i(v::VariableMap, u, x) = findfirst(isequal(x), remove(v.args[operation(u)], v.time))

@inline function axiesvals(v::VariableMap, u_, x_, I)
    u = depvar(u_, v)
    map(ivs(u, v)) do x
        x => (I[x2i(v, u, x)] == 1 ? v.intervals[x][1] : v.intervals[x][2])
    end
end
