"""
    Fornberg (2020) Hermite finite-difference rules.

Discretizes spatial derivatives using both function values f(xᵢ) and first-derivative
values f′(xᵢ) at the stencil nodes.  When a partner variable `v = Dx(u)` is mapped via
[`HermiteScheme`](@ref), every derivative of `u` is approximated as `D·u + E·v` where
`(D, E)` are the HFD weights from [`calculate_weights`](@ref) with `dfdx=true`.
Variables without a mapped partner fall back to standard central differences.
"""

"""
    _same_hermite_variable(a, b) -> Bool

Check whether `a` and `b` refer to the same dependent variable, handling the
mismatch between bare symbolic callable (`u`) and applied term (`u(t, x)`).
"""
@inline function _hermite_operation(a)
    av = unwrap(a)
    return iscall(av) ? operation(av) : av
end

@inline function _same_hermite_variable(a, b)::Bool
    return isequal(_hermite_operation(a), _hermite_operation(b))
end

"""
    _resolve_hermite_variable(v, s::DiscreteSpace)

Return the key in `s.discvars` that matches `v`, or `nothing`.
"""
@inline function _resolve_hermite_variable(v, s)
    haskey(s.discvars, v) && return v
    for candidate in keys(s.discvars)
        _same_hermite_variable(candidate, v) && return candidate
    end
    return nothing
end

"""
    _hermite_partner(scheme::HermiteScheme, u, x, s::DiscreteSpace)

Look up the derivative-partner variable for `u` in direction `x`.
Returns the resolved key into `s.discvars`, or `nothing` if `u` is not mapped.
"""
@inline function _hermite_partner(scheme::HermiteScheme, u, x, s)
    for (key, value) in scheme.deriv_var
        if key isa Tuple && length(key) == 2
            _same_hermite_variable(key[1], u) && isequal(key[2], x) || continue
            return _resolve_hermite_variable(value, s)
        elseif _same_hermite_variable(key, u)
            length(ivs(u, s)) == 1 || throw(ArgumentError(
                "HermiteScheme mapping $(key) => $(value) is ambiguous for " *
                "multidimensional variable $(u). Use direction-specific mappings " *
                "such as (u, x) => vx and (u, y) => vy."
            ))
            return _resolve_hermite_variable(value, s)
        end
    end
    return nothing
end

"""
    _hermite_weights_and_taps(D, H, II, s, bs, jx, u)

Compute the HFD weight vectors `(d_weights, e_weights)` and stencil tap indices
for the derivative operator `D` at grid index `II`.  Reuses
`central_difference_weights_and_stencil` for tap selection; interior uniform
points use the precomputed weights in `H`, boundary/nonuniform points recompute
via `calculate_weights`.
"""
@inline function _hermite_weights_and_taps(D, H, II, s, bs, jx, u)
    j, x = jx
    if !isempty(bs) && !(s.grid[x] isa StepRangeLen)
        throw(ArgumentError(
            "HermiteScheme does not support periodic or interface stencils " *
                "on a nonuniform grid in $x."
        ))
    end
    _, Itap = central_difference_weights_and_stencil(D, II, s, bs, jx, u)

    if any(I -> I isa RefCartesianIndex, Itap)
        self_periodic = !isempty(bs) && all(bs) do b
            isequal(b.x, b.x2) && _same_hermite_variable(b.u, u) &&
                _same_hermite_variable(b.u2, u)
        end
        self_periodic || throw(ArgumentError(
            "HermiteScheme does not yet support cross-variable interface stencils."
        ))

        dweights = H.d_weights
        eweights = H.e_weights
        if dweights === nothing || eweights === nothing
            error("Missing precomputed Hermite weights for uniform periodic stencil.")
        end
        periodic_taps = [I isa RefCartesianIndex ? I.I : I for I in Itap]
        return dweights, eweights, periodic_taps
    end

    x0 = s.grid[x][II[j]]
    dorder = D.derivative_order

    # Interior uniform stencils are translation-invariant → use precomputed weights.
    # Boundary and nonuniform stencils are recomputed at their actual coordinates.
    interior = s.grid[x] isa StepRangeLen &&
        II[j] > D.boundary_point_count &&
        II[j] <= length(s, x) - D.boundary_point_count &&
        length(Itap) == H.stencil_length
    dweights = H.d_weights
    eweights = H.e_weights
    if interior && dweights !== nothing && eweights !== nothing
        return dweights, eweights, Itap
    end
    coords = [s.grid[x][I[j]] for I in Itap]
    dweights, eweights = calculate_weights(dorder, x0, coords; dfdx = true)
    return dweights, eweights, Itap
end

"""
    hermite_central_difference(D, H, II, s, bs, jx, u, v)

Return the symbolic HFD approximation `sym_dot(D_w, u_taps) + sym_dot(E_w, v_taps)`
for the derivative described by operator `D` at index `II`.
"""
@inline function hermite_central_difference(D, H, II, s, bs, jx, u, v)
    dweights, eweights, Itap = _hermite_weights_and_taps(D, H, II, s, bs, jx, u)
    u_vals = [s.discvars[u][I] for I in Itap]
    v_vals = [s.discvars[v][I] for I in Itap]
    return sym_dot(dweights, u_vals) + sym_dot(eweights, v_vals)
end

"""
    generate_hermite_rules(II, s, depvars, derivweights, bcmap, indexmap, terms)

Generate replacement rules for all spatial derivatives at index `II`.

For each dependent variable and spatial direction:
  - If a partner is mapped via `HermiteScheme`, use HFD weights for all orders.
  - Otherwise, fall back to standard central differences.

This replaces `generate_cartesian_rules` when `advection_scheme isa HermiteScheme`.
"""
function generate_hermite_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms
    )
    scheme = derivweights.advection_scheme
    central_ufunc(u, I, _) = s.discvars[u][I]
    rules = Pair[]
    for u in depvars
        for x in ivs(u, s)
            partner = _hermite_partner(scheme, u, x, s)
            jx = (x2i(s, u, x), x)
            bs = filter_interfaces(bcmap[operation(u)][x])
            for d in derivweights.orders[x]
                lhs = (Differential(x)^d)(u)
                rhs = if partner !== nothing
                    hermite_central_difference(
                        derivweights.map[Differential(x)^d],
                        derivweights.hermitemap[Differential(x)^d],
                        Idx(II, s, u, indexmap), s, bs, jx, u, partner)
                else
                    central_difference(
                        derivweights.map[Differential(x)^d], Idx(II, s, u, indexmap),
                        s, bs, jx, u, central_ufunc)
                end
                push!(rules, lhs => rhs)
            end
        end
    end
    return rules
end
