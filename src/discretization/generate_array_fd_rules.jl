"""
Array-level finite difference rule generation for `ArrayDiscretization`.

For interior points on uniform grids, the stencil weights are identical at
every point.  This module pre-computes them and applies them directly,
bypassing the per-point scheme-detection logic used by the scalar path.

Near-boundary points use boundary stencils (shifted, one-sided) exactly as the
scalar code does in `central_difference_weights_and_stencil`.
"""

# ─── stencil pre-computation ─────────────────────────────────────────────────

"""
    StencilInfo

Pre-computed information for a particular derivative operator.
"""
struct StencilInfo
    D_op::DerivativeOperator     # full operator, needed for boundary stencils
    offsets::Vector{Int}         # half_range(stencil_length)
    is_uniform::Bool             # true if dx is a Number
end

"""
    precompute_stencils(s, depvars, derivweights)

Returns a `Dict` mapping `(u, x, d)` to a `StencilInfo` for every
(variable, spatial dim, even derivative order) triple.
"""
function precompute_stencils(s, depvars, derivweights)
    info = Dict{Any, StencilInfo}()
    for u in depvars
        for x in ivs(u, s)
            for d in derivweights.orders[x]
                iseven(d) || continue
                D_op = derivweights.map[Differential(x)^d]
                info[(u, x, d)] = StencilInfo(
                    D_op,
                    collect(half_range(D_op.stencil_length)),
                    D_op.dx isa Number
                )
            end
        end
    end
    return info
end

"""
    stencil_weights_and_taps(si, II, j, grid_len, haslower, hasupper)

Compute the stencil weights and tap-point offsets at index `II` in dimension
`j`.  Handles boundary proximity exactly like
`central_difference_weights_and_stencil` from the scalar path.

Returns `(weights, Itap)` where `Itap` is a vector of `CartesianIndex`.
"""
function stencil_weights_and_taps(si::StencilInfo, II, j, ndim, grid_len, haslower, hasupper)
    D = si.D_op
    I1 = unitindex(ndim, j)
    idx = II[j]

    if (idx <= D.boundary_point_count) & !haslower
        # Near lower boundary — use one-sided stencil
        weights = D.low_boundary_coefs[idx]
        offset = 1 - idx
        Itap = [II + (k + offset) * I1 for k in 0:(D.boundary_stencil_length - 1)]
    elseif (idx > (grid_len - D.boundary_point_count)) & !hasupper
        # Near upper boundary — use one-sided stencil
        weights = D.high_boundary_coefs[grid_len - idx + 1]
        offset = grid_len - idx
        Itap = [II + (k + offset) * I1 for k in (-D.boundary_stencil_length + 1):0]
    else
        # True interior — use centred stencil
        if si.is_uniform
            weights = D.stencil_coefs
        else
            weights = D.stencil_coefs[idx - D.boundary_point_count]
        end
        Itap = [II + off * I1 for off in si.offsets]
    end
    return weights, Itap
end

# ─── interior equation generation ────────────────────────────────────────────

"""
    generate_array_interior_eqs(s, depvars, pde, derivweights, bcmap, eqvar,
                                 indexmap, boundaryvalfuncs, interior_ranges)

Generate discretised interior equations.

For each interior grid point, centred-difference rules are built from
pre-computed stencil info (with boundary-proximity handling) and combined with
variable/grid maps.  `pde_substitute` is then applied to produce the equation.
"""
function generate_array_interior_eqs(
        s, depvars, pde, derivweights, bcmap, eqvar,
        indexmap, boundaryvalfuncs, interior_ranges
    )
    # --- pre-compute stencils once -------------------------------------------
    stencil_cache = precompute_stencils(s, depvars, derivweights)

    lo, hi = interior_ranges[1]           # Phase 1: 1D only
    central_ufunc(u, I, x) = _disc_gather(s.discvars[u], I)

    eqs = map(lo:hi) do i
        II = CartesianIndex(i)

        # --- FD rules from pre-computed stencils -----------------------------
        fd_rules = Pair[]
        for u in depvars
            for (dim_j, x) in enumerate(ivs(u, s))
                j = x2i(s, u, x)
                bs = filter_interfaces(bcmap[operation(u)][x])
                haslower, hasupper = haslowerupper(bs, x)
                ndim = ndims(u, s)
                gl = length(s, x)
                for d in derivweights.orders[x]
                    iseven(d) || continue
                    si = stencil_cache[(u, x, d)]
                    weights, Itap = stencil_weights_and_taps(
                        si, Idx(II, s, u, indexmap), j, ndim, gl, haslower, hasupper
                    )
                    expr = sym_dot(weights, central_ufunc(u, Itap, x))
                    push!(fd_rules, (Differential(x)^d)(u) => expr)
                end
            end
        end

        # --- boundary-value and variable/grid rules --------------------------
        boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs, init = [])
        var_rules = valmaps(s, eqvar, depvars, II, indexmap)

        # --- assemble and substitute -----------------------------------------
        rules = vcat(fd_rules, boundaryrules, var_rules)
        rdict = Dict(rules)
        expand_derivatives(pde_substitute(pde.lhs, rdict)) ~ pde_substitute(pde.rhs, rdict)
    end

    return collect(eqs)
end
