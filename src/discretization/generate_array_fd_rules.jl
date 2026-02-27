"""
Array-level finite difference rule generation for `ArrayDiscretization`.

For interior points on uniform grids, the stencil weights are identical at
every point.  This module pre-computes them once and builds a single template
expression using a symbolic ArrayOp index from `SymbolicUtils.idxs_for_arrayop`.
The template is then instantiated at each interior point via `pde_substitute`,
avoiding per-point scheme-detection and rule construction.

Near-boundary interior points (where centred stencils don't fit) fall back to
per-point stencil computation using `stencil_weights_and_taps`.
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

For the centred-stencil region, substitution rules (FD stencils and
variable/grid maps) are built once using a symbolic ArrayOp index variable
`_i`, producing a template equation.  This template is then instantiated at
each interior grid point via `pde_substitute`.

Near-boundary interior points that require non-centred stencils fall back to
per-point computation using `stencil_weights_and_taps`.
"""
function generate_array_interior_eqs(
        s, depvars, pde, derivweights, bcmap, eqvar,
        indexmap, boundaryvalfuncs, interior_ranges
    )
    stencil_cache = precompute_stencils(s, depvars, derivweights)

    lo, hi = interior_ranges[1]           # Phase 1: 1D only

    # ── determine centred-stencil region ─────────────────────────────────────
    # Find the grid length and the maximum boundary_point_count on each side
    # across all (variable, spatial dim, derivative order) triples.
    max_lower_bpc = 0
    max_upper_bpc = 0
    gl = 0
    for u in depvars
        for (dim_j, x) in enumerate(ivs(u, s))
            gl = length(s, x)
            bs = filter_interfaces(bcmap[operation(u)][x])
            haslower, hasupper = haslowerupper(bs, x)
            for d in derivweights.orders[x]
                iseven(d) || continue
                bpc = stencil_cache[(u, x, d)].D_op.boundary_point_count
                if !haslower
                    max_lower_bpc = max(max_lower_bpc, bpc)
                end
                if !hasupper
                    max_upper_bpc = max(max_upper_bpc, bpc)
                end
            end
        end
    end

    lo_centered = max(lo, max_lower_bpc + 1)
    hi_centered = min(hi, gl - max_upper_bpc)
    n_centered = max(0, hi_centered - lo_centered + 1)

    # The ArrayOp template requires uniform grids (shared stencil weights).
    # For non-uniform grids, fall back to per-point for all interior points.
    all_uniform = all(si.is_uniform for si in values(stencil_cache))

    if !all_uniform
        return _per_point_eqs(
            collect(lo:hi),
            s, depvars, pde, derivweights, stencil_cache, bcmap, eqvar,
            indexmap, boundaryvalfuncs, gl
        )
    end

    # ── per-point equations for boundary-proximity interior points ───────────
    eqs_lower = _per_point_eqs(
        collect(lo:(lo_centered - 1)),
        s, depvars, pde, derivweights, stencil_cache, bcmap, eqvar,
        indexmap, boundaryvalfuncs, gl
    )
    eqs_upper = _per_point_eqs(
        collect((hi_centered + 1):hi),
        s, depvars, pde, derivweights, stencil_cache, bcmap, eqvar,
        indexmap, boundaryvalfuncs, gl
    )

    # ── ArrayOp-based template for centred-stencil region ────────────────────
    eqs_centered = if n_centered > 0
        _build_centered_eqs(
            n_centered, lo_centered, s, depvars, pde, derivweights,
            stencil_cache, eqvar, indexmap
        )
    else
        Equation[]
    end

    return collect(vcat(eqs_lower, eqs_centered, eqs_upper))
end

# ─── per-point fallback for boundary-proximity interior points ───────────────

"""
    _per_point_eqs(indices, ...)

Build equations for specific grid indices using the per-point approach.
Used for interior points near boundaries where the centred stencil doesn't fit.
"""
function _per_point_eqs(
        indices, s, depvars, pde, derivweights, stencil_cache, bcmap, eqvar,
        indexmap, boundaryvalfuncs, gl
    )
    isempty(indices) && return Equation[]
    central_ufunc(u, I, x) = _disc_gather(s.discvars[u], I)

    map(indices) do i
        II = CartesianIndex(i)

        fd_rules = Pair[]
        for u in depvars
            for (dim_j, x) in enumerate(ivs(u, s))
                j = x2i(s, u, x)
                bs = filter_interfaces(bcmap[operation(u)][x])
                haslower, hasupper = haslowerupper(bs, x)
                ndim = ndims(u, s)
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

        boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs, init = [])
        var_rules = valmaps(s, eqvar, depvars, II, indexmap)
        rules = vcat(fd_rules, boundaryrules, var_rules)
        rdict = Dict(rules)
        expand_derivatives(pde_substitute(pde.lhs, rdict)) ~ pde_substitute(pde.rhs, rdict)
    end
end

# ─── ArrayOp template for centred-stencil region ────────────────────────────

"""
    _build_centered_eqs(n_centered, lo_centered, s, depvars, pde, derivweights,
                         stencil_cache, eqvar, indexmap)

Build centred-stencil equations using a symbolic index template.

Constructs FD and variable/grid substitution rules parameterised by a symbolic
index `_i` from `SymbolicUtils.idxs_for_arrayop(SymReal)`.  The PDE is
symbolically transformed once to produce a template, which is then
instantiated at each centred-stencil point via `pde_substitute`.
"""
function _build_centered_eqs(
        n_centered, lo_centered, s, depvars, pde, derivweights,
        stencil_cache, eqvar, indexmap
    )
    _i = SymbolicUtils.idxs_for_arrayop(SymbolicUtils.SymReal)[1]
    base = lo_centered - 1  # _i=1 maps to grid index lo_centered

    # ── FD rules using symbolic _i ───────────────────────────────────────────
    fd_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        for (dim_j, x) in enumerate(ivs(u, s))
            for d in derivweights.orders[x]
                iseven(d) || continue
                si = stencil_cache[(u, x, d)]
                weights = si.D_op.stencil_coefs  # uniform, centred
                taps = [Symbolics.wrap(u_c[_i + base + off]) for off in si.offsets]
                expr = sym_dot(weights, taps)
                push!(fd_rules, (Differential(x)^d)(u) => expr)
            end
        end
    end

    # ── Variable/grid rules using symbolic _i ────────────────────────────────
    var_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        push!(var_rules, u => Symbolics.wrap(u_c[_i + base]))
    end
    for x in ivs(eqvar, s)
        grid_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(collect(s.grid[x]))
        push!(var_rules, x => Symbolics.wrap(grid_c[_i + base]))
    end

    # ── Build template (once) ────────────────────────────────────────────────
    rdict = Dict(vcat(fd_rules, var_rules))
    template_lhs = expand_derivatives(pde_substitute(pde.lhs, rdict))
    template_rhs = pde_substitute(pde.rhs, rdict)

    # ── Instantiate at each centred-stencil point ────────────────────────────
    return map(1:n_centered) do k
        lhs_k = pde_substitute(template_lhs, Dict(_i => k))
        rhs_k = pde_substitute(template_rhs, Dict(_i => k))
        lhs_k ~ rhs_k
    end
end
