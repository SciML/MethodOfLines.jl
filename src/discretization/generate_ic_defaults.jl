"""
    generate_ic_defaults(tconds, s::DiscreteSpace, ::MOLDiscretization)

The discrete initial conditions: one pair per initial condition, from the system unknown
(the array `u(t)[1:n, ...]`, or `Dt^k` of it for an initial condition on a time
derivative) to its values on the grid of `u`, staggered points included.
"""
function PDEBase.generate_ic_defaults(
        tconds, s::DiscreteSpace, ::MOLDiscretization
    )
    t = s.time
    t === nothing && return []
    return map(tconds) do ic
        if isupper(ic)
            throw(ArgumentError("Upper boundary condition $(ic.eq) on time variable is not supported, please use a change of variables `t => -τ` to make this an initial condition."))
        end
        u = depvar(ic.u, s)
        D = ic.order == 0 ? identity : (Differential(t)^ic.order)
        rhs = safe_unwrap(symbolic_linear_solve(ic.eq, D(ic.u)))
        vals = ic_values(rhs, s, u, ic.depvars)
        D(system_unknown(s, u)) => (ndims(vals) == 0 ? vals[] : vals)
    end
end

# The value of `rhs` at every point of the grid of `u`. A right-hand side of the spatial
# coordinates alone is compiled once and broadcast over the coordinates; one that also
# involves parameters or other variables is substituted point by point.
function ic_values(rhs, s::DiscreteSpace, u, depvars)
    xs = ivs(u, s)
    if !SymbolicUtils.issym(rhs) && !iscall(rhs)
        return fill(unwrap_const(rhs), size(s.Igrid[u]))
    end
    if all(v -> any(x -> isequal(v, x), xs), Symbolics.get_variables(rhs))
        f = Symbolics.build_function(rhs, xs...; expression = Val{false}, nanmath = false)
        coords = map(xs) do x
            # shifted per element, as `gridvals` does, so the points match it bit for bit
            c = collect(s.grid[x]) .+ staggered_shift(s, u, x)
            d = x2i(s, u, x)
            reshape(c, ntuple(i -> i == d ? length(c) : 1, length(xs)))
        end
        return f.(coords...)
    end
    indexmap = Dict(x => i for (i, x) in enumerate(xs))
    return map(s.Igrid[u]) do II
        symbolic_to_float(pde_substitute(rhs, Dict(valmaps(s, u, depvars, II, indexmap))))
    end
end
