"""
Array-level finite difference rule generation for `ArrayDiscretization`.

For PDEs on uniform grids, stencil weights are identical at every interior
point in each dimension.  This module pre-computes them once and builds a
single ArrayOp expression using N symbolic index variables `_i1, _i2, ...`
(one per spatial dimension).  The ArrayOp is a genuine symbolic array operation
that, when ModelingToolkit supports native array compilation, will compile to a
single vectorized loop.  Until then, MTK's `flatten_equations` scalarizes it
into individual per-point equations, preserving correctness.

Supported ArrayOp patterns:
- Centred (even-order) derivatives on uniform grids
- Upwind (odd-order) derivatives with UpwindScheme on uniform grids
- Mixed cross-derivatives on uniform grids

All other cases -- non-uniform grids, nonlinear Laplacians, spherical
derivatives, WENO, FunctionalScheme, etc. -- fall back to per-point
computation via `discretize_equation_at_point` from the scalar path, which
supports ALL scheme types.
"""

# --- stencil pre-computation ------------------------------------------------

"""
    StencilInfo

Pre-computed information for a particular centred derivative operator.
"""
struct StencilInfo
    D_op::DerivativeOperator     # full operator, needed for boundary stencils
    offsets::Vector{Int}         # half_range(stencil_length)
    is_uniform::Bool             # true if dx is a Number
end

"""
    UpwindStencilInfo

Pre-computed information for upwind derivative operators (both wind directions).
"""
struct UpwindStencilInfo
    D_neg::DerivativeOperator    # negative-wind (offside=0)
    D_pos::DerivativeOperator    # positive-wind (offside=d+upwind_order-1)
    neg_offsets::Vector{Int}     # 0:(stencil_length-1) for neg
    pos_offsets::Vector{Int}     # (-stencil_length+1):0 for pos
    is_uniform::Bool
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
    precompute_upwind_stencils(s, depvars, derivweights)

Returns a `Dict` mapping `(u, x, d)` to an `UpwindStencilInfo` for every
(variable, spatial dim, odd derivative order) triple.  Only populated when
the advection scheme is `UpwindScheme` and windmap operators exist.
"""
function precompute_upwind_stencils(s, depvars, derivweights)
    info = Dict{Any, UpwindStencilInfo}()
    !(derivweights.advection_scheme isa UpwindScheme) && return info
    for u in depvars
        for x in ivs(u, s)
            for d in derivweights.orders[x]
                isodd(d) || continue
                Dx_d = Differential(x)^d
                haskey(derivweights.windmap[1], Dx_d) || continue
                D_neg = derivweights.windmap[1][Dx_d]  # offside=0
                D_pos = derivweights.windmap[2][Dx_d]  # offside=d+upwind_order-1
                info[(u, x, d)] = UpwindStencilInfo(
                    D_neg,
                    D_pos,
                    collect(0:(D_neg.stencil_length - 1)),
                    collect((-D_pos.stencil_length + 1):0),
                    D_neg.dx isa Number
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
        # Near lower boundary -- use one-sided stencil
        weights = D.low_boundary_coefs[idx]
        offset = 1 - idx
        Itap = [II + (k + offset) * I1 for k in 0:(D.boundary_stencil_length - 1)]
    elseif (idx > (grid_len - D.boundary_point_count)) & !hasupper
        # Near upper boundary -- use one-sided stencil
        weights = D.high_boundary_coefs[grid_len - idx + 1]
        offset = grid_len - idx
        Itap = [II + (k + offset) * I1 for k in (-D.boundary_stencil_length + 1):0]
    else
        # True interior -- use centred stencil
        if si.is_uniform
            weights = D.stencil_coefs
        else
            weights = D.stencil_coefs[idx - D.boundary_point_count]
        end
        Itap = [II + off * I1 for off in si.offsets]
    end
    return weights, Itap
end

# --- interior equation generation -------------------------------------------

"""
    generate_array_interior_eqs(s, depvars, pde, derivweights, bcmap, eqvar,
                                 indexmap, boundaryvalfuncs, interior_ranges)

Generate discretised interior equations.

For the interior region on N-D uniform grids, a single ArrayOp equation is
produced when possible.  Supported patterns:
- Centred (even-order) derivatives
- Upwind (odd-order) derivatives with UpwindScheme
- Mixed cross-derivatives

Boundary-proximity interior points (the "frame" around the centred region)
fall back to per-point computation via `discretize_equation_at_point`.

All other cases (non-uniform grids, nonlinear Laplacians, WENO,
FunctionalScheme, spherical, etc.) fall back entirely to per-point
computation, which supports ALL scheme types.
"""
function generate_array_interior_eqs(
        s, depvars, pde, derivweights, bcmap, eqvar,
        indexmap, boundaryvalfuncs, interior_ranges
    )
    stencil_cache = precompute_stencils(s, depvars, derivweights)
    upwind_cache = precompute_upwind_stencils(s, depvars, derivweights)

    # -- determine whether the ArrayOp path can handle this PDE ---------------
    all_uniform = isempty(stencil_cache) ? true :
        all(si.is_uniform for si in values(stencil_cache))
    if !isempty(upwind_cache)
        all_uniform = all_uniform && all(si.is_uniform for si in values(upwind_cache))
    end

    has_odd_orders = any(
        any(isodd(d) for d in derivweights.orders[x])
        for u in depvars for x in ivs(u, s)
    )
    can_upwind = has_odd_orders && derivweights.advection_scheme isa UpwindScheme
    can_template = all_uniform && (!has_odd_orders || can_upwind)

    ndim = length(interior_ranges)

    if !can_template
        # Full per-point fallback: delegate every interior point to the
        # scalar path's discretize_equation_at_point.
        cart_ranges = Tuple(r[1]:r[2] for r in interior_ranges)
        interior = CartesianIndices(cart_ranges)
        return collect(vec(map(interior) do II
            discretize_equation_at_point(
                II, s, depvars, pde, derivweights, bcmap,
                eqvar, indexmap, boundaryvalfuncs
            )
        end))
    end

    # -- N-D ArrayOp path (uniform grid) --------------------------------------
    lo_vec = [r[1] for r in interior_ranges]
    hi_vec = [r[2] for r in interior_ranges]
    eqvar_ivs = ivs(eqvar, s)
    gl_vec = [length(s, x) for x in eqvar_ivs]

    # Per-dimension maximum boundary_point_count on each side across all
    # (variable, spatial dim, derivative order) triples.
    max_lower_bpc = zeros(Int, ndim)
    max_upper_bpc = zeros(Int, ndim)
    for u in depvars
        for (_, x) in enumerate(ivs(u, s))
            eq_dim = indexmap[x]  # dimension index in eqvar's ordering
            bs = filter_interfaces(bcmap[operation(u)][x])
            haslower, hasupper = haslowerupper(bs, x)
            for d in derivweights.orders[x]
                if iseven(d)
                    bpc = stencil_cache[(u, x, d)].D_op.boundary_point_count
                    if !haslower
                        max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], bpc)
                    end
                    if !hasupper
                        max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], bpc)
                    end
                elseif isodd(d) && haskey(upwind_cache, (u, x, d))
                    usi = upwind_cache[(u, x, d)]
                    # Negative-wind (offside=0): stencil reaches forward
                    # → boundary proximity near upper end
                    neg_bpc = usi.D_neg.boundary_point_count
                    # Positive-wind (offside>0): stencil reaches backward
                    # → boundary proximity near lower end
                    pos_bpc = usi.D_pos.boundary_point_count
                    bpc = max(neg_bpc, pos_bpc)
                    if !haslower
                        max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], bpc)
                    end
                    if !hasupper
                        max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], bpc)
                    end
                end
            end
        end
    end

    lo_centered = [max(lo_vec[d], max_lower_bpc[d] + 1) for d in 1:ndim]
    hi_centered = [min(hi_vec[d], gl_vec[d] - max_upper_bpc[d]) for d in 1:ndim]
    n_centered  = [max(0, hi_centered[d] - lo_centered[d] + 1) for d in 1:ndim]

    # -- per-point equations for boundary-proximity interior points -----------
    # These are the "frame" around the centred region: all interior points
    # that are NOT in the centred rectangle.
    full_interior = CartesianIndices(Tuple(lo_vec[d]:hi_vec[d] for d in 1:ndim))
    centered_nonempty = all(lo_centered[d] <= hi_centered[d] for d in 1:ndim)

    eqs_boundary = Equation[]
    for II in full_interior
        in_centered = centered_nonempty &&
            all(lo_centered[d] <= II[d] <= hi_centered[d] for d in 1:ndim)
        in_centered && continue
        push!(eqs_boundary, discretize_equation_at_point(
            II, s, depvars, pde, derivweights, bcmap,
            eqvar, indexmap, boundaryvalfuncs
        ))
    end

    # -- ArrayOp equation for interior region ---------------------------------
    eqs_centered = if centered_nonempty && all(n_centered .> 0)
        candidate, eq_first = _build_interior_arrayop(
            n_centered, lo_centered, s, depvars, pde, derivweights,
            stencil_cache, upwind_cache, bcmap, eqvar, indexmap
        )
        # Validate: compare the first instantiated equation against the
        # scalar path for the same point.  This catches any mismatch from
        # unsupported derivative patterns (nonlinear Laplacian, spherical,
        # etc.) that the template cannot handle.
        II_check = CartesianIndex(Tuple(lo_centered))
        eq_scalar = discretize_equation_at_point(
            II_check, s, depvars, pde, derivweights, bcmap,
            eqvar, indexmap, boundaryvalfuncs
        )
        if isequal(eq_first.lhs, eq_scalar.lhs) &&
           isequal(eq_first.rhs, eq_scalar.rhs)
            candidate
        else
            # Template doesn't match scalar path -- fall back to per-point
            # for the centred region as well.
            centered_rect = CartesianIndices(
                Tuple(lo_centered[d]:hi_centered[d] for d in 1:ndim)
            )
            collect(vec(map(centered_rect) do II
                discretize_equation_at_point(
                    II, s, depvars, pde, derivweights, bcmap,
                    eqvar, indexmap, boundaryvalfuncs
                )
            end))
        end
    else
        Equation[]
    end

    return collect(vcat(eqs_boundary, eqs_centered))
end

# --- ArrayOp construction for interior region --------------------------------

"""
    _build_interior_arrayop(n_centered, lo_centered, s, depvars, pde,
                             derivweights, stencil_cache, upwind_cache,
                             bcmap, eqvar, indexmap)

Build a single ArrayOp equation for the interior region.

Handles centred (even-order), upwind (odd-order), and mixed cross-derivative
stencils using symbolic index variables.

Returns `(eqs, eq_first)` where `eqs` is a single-element vector containing
the ArrayOp equation, and `eq_first` is the scalar equation at the first
centred point (for validation against the scalar path).
"""
function _build_interior_arrayop(
        n_centered, lo_centered, s, depvars, pde, derivweights,
        stencil_cache, upwind_cache, bcmap, eqvar, indexmap
    )
    ndim = length(n_centered)
    _idxs_arr = SymbolicUtils.idxs_for_arrayop(SymbolicUtils.SymReal)
    _idxs = [_idxs_arr[d] for d in 1:ndim]
    bases = [lo_centered[d] - 1 for d in 1:ndim]

    # -- FD rules for centred (even-order) derivatives ------------------------
    fd_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        for (_, x) in enumerate(u_spatial)
            for d in derivweights.orders[x]
                iseven(d) || continue
                si = stencil_cache[(u, x, d)]
                weights = si.D_op.stencil_coefs  # uniform, centred
                taps = map(si.offsets) do off
                    idx_exprs = map(u_spatial) do xv
                        eq_d = indexmap[xv]
                        base_expr = _idxs[eq_d] + bases[eq_d]
                        isequal(xv, x) ? base_expr + off : base_expr
                    end
                    Symbolics.wrap(u_c[idx_exprs...])
                end
                expr = sym_dot(weights, taps)
                push!(fd_rules, (Differential(x)^d)(u) => expr)
            end
        end
    end

    # -- Variable/grid rules using symbolic indices ---------------------------
    var_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        idx_exprs = [_idxs[indexmap[xv]] + bases[indexmap[xv]] for xv in u_spatial]
        push!(var_rules, u => Symbolics.wrap(u_c[idx_exprs...]))
    end
    eqvar_ivs = ivs(eqvar, s)
    for x in eqvar_ivs
        grid_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(collect(s.grid[x]))
        dim = indexmap[x]
        push!(var_rules, x => Symbolics.wrap(grid_c[_idxs[dim] + bases[dim]]))
    end

    # -- Upwind (odd-order) rules with IfElse wind switching ------------------
    upwind_rules = Pair[]
    if !isempty(upwind_cache)
        upwind_rules = _build_upwind_rules(
            pde, s, depvars, derivweights, upwind_cache,
            bcmap, indexmap, _idxs, bases, var_rules
        )
    end

    # -- Mixed derivative rules -----------------------------------------------
    mixed_rules = _build_mixed_derivative_rules(
        s, depvars, derivweights, indexmap, _idxs, bases
    )

    # -- Build templates (once) -----------------------------------------------
    # Upwind rules are term-level substitutions (they replace entire additive
    # terms, not just derivative sub-expressions).  Apply them first on the
    # PDE expression, then apply FD + var rules on the result.
    all_fd_rules = vcat(fd_rules, mixed_rules)
    rdict = Dict(vcat(all_fd_rules, var_rules))

    if isempty(upwind_rules)
        template_lhs = expand_derivatives(pde_substitute(pde.lhs, rdict))
        template_rhs = pde_substitute(pde.rhs, rdict)
    else
        # Apply upwind term-level substitutions first, then FD+var rules
        upwind_dict = Dict(upwind_rules)
        lhs_upwinded = pde_substitute(pde.lhs, upwind_dict)
        rhs_upwinded = pde_substitute(pde.rhs, upwind_dict)
        template_lhs = expand_derivatives(pde_substitute(lhs_upwinded, rdict))
        template_rhs = pde_substitute(rhs_upwinded, rdict)
    end

    # -- Separate time derivative from spatial terms --------------------------
    eqvar_raw = Symbolics.unwrap(s.discvars[eqvar])
    eqvar_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(eqvar_raw)
    eqvar_idx_exprs = [_idxs[d] + bases[d] for d in 1:ndim]
    dt_template = Differential(s.time)(Symbolics.wrap(eqvar_c[eqvar_idx_exprs...]))

    spatial_rhs = dt_template - template_lhs + template_rhs

    # -- Wrap in ArrayOps -----------------------------------------------------
    ao_ranges = Dict(_idxs[d] => (1:1:n_centered[d]) for d in 1:ndim)

    u_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _idxs, eqvar_c[eqvar_idx_exprs...], +, nothing, ao_ranges
    )
    lhs_wrapped = Differential(s.time)(Symbolics.wrap(u_ao))

    rhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _idxs, Symbolics.unwrap(spatial_rhs), +, nothing, ao_ranges
    )

    arrayop_eq = lhs_wrapped ~ Symbolics.wrap(rhs_ao)

    # -- Also produce first scalar equation for validation --------------------
    sub_first = Dict(_idxs[d] => 1 for d in 1:ndim)
    lhs_first = pde_substitute(template_lhs, sub_first)
    rhs_first = pde_substitute(template_rhs, sub_first)
    eq_first = lhs_first ~ rhs_first

    return [arrayop_eq], eq_first
end

# --- Upwind ArrayOp rules ---------------------------------------------------

"""
    _build_upwind_rules(pde, s, depvars, derivweights, upwind_cache,
                         bcmap, indexmap, _idxs, bases, var_rules)

Build term-level upwind substitution rules for odd-order derivatives.

Uses the same pattern-matching approach as `generate_winding_rules` from the
scalar path, but substitutes ArrayOp-parameterized stencils instead of
concrete-index stencils.  The advection coefficient is expressed in terms of
symbolic grid indices via `var_rules`.

Returns a vector of `Pair{term => IfElse_expr}` for matched terms, plus
fallback rules for unmatched standalone derivatives.
"""
function _build_upwind_rules(
        pde, s, depvars, derivweights, upwind_cache,
        bcmap, indexmap, _idxs, bases, var_rules
    )
    # Helper: build stencil expression for a given variable, dimension, offsets, weights
    function _upwind_stencil_expr(u, x, offsets, weights, _idxs, bases, indexmap, s)
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        taps = map(offsets) do off
            idx_exprs = map(u_spatial) do xv
                eq_d = indexmap[xv]
                base_expr = _idxs[eq_d] + bases[eq_d]
                isequal(xv, x) ? base_expr + off : base_expr
            end
            Symbolics.wrap(u_c[idx_exprs...])
        end
        return sym_dot(weights, taps)
    end

    terms = split_terms(pde, s.x̄)
    vr_dict = Dict(var_rules)

    # Build @rule patterns for multiplication and division (same structure as
    # generate_winding_rules but returning ArrayOp-parameterized stencils).
    wind_rules = Pair[]

    for u in depvars
        u_spatial = ivs(u, s)
        for x in u_spatial
            odd_orders = filter(isodd, derivweights.orders[x])
            for d in odd_orders
                haskey(upwind_cache, (u, x, d)) || continue
                usi = upwind_cache[(u, x, d)]

                neg_expr = _upwind_stencil_expr(
                    u, x, usi.neg_offsets, usi.D_neg.stencil_coefs,
                    _idxs, bases, indexmap, s
                )
                pos_expr = _upwind_stencil_expr(
                    u, x, usi.pos_offsets, usi.D_pos.stencil_coefs,
                    _idxs, bases, indexmap, s
                )

                # Multiplication pattern: coeff * Dx^d(u)
                mul_rule = @rule *(
                    ~~a,
                    $(Differential(x)^d)(u),
                    ~~b
                ) => begin
                    coeff = *(~a..., ~b...)
                    coeff_subst = pde_substitute(coeff, vr_dict)
                    IfElse.ifelse(
                        coeff_subst > 0,
                        coeff_subst * pos_expr,
                        coeff_subst * neg_expr
                    )
                end

                # Division pattern: (coeff * Dx^d(u)) / denom
                div_rule = @rule /(
                    *(~~a, $(Differential(x)^d)(u), ~~b),
                    ~c
                ) => begin
                    coeff = *(~a..., ~b...) / ~c
                    coeff_subst = pde_substitute(coeff, vr_dict)
                    IfElse.ifelse(
                        coeff_subst > 0,
                        coeff_subst * pos_expr,
                        coeff_subst * neg_expr
                    )
                end

                # Apply rules to each term
                for t in terms
                    matched = mul_rule(t)
                    if matched !== nothing
                        push!(wind_rules, t => matched)
                        continue
                    end
                    matched = div_rule(t)
                    if matched !== nothing
                        push!(wind_rules, t => matched)
                    end
                end
            end
        end
    end

    # Fallback rules for standalone derivatives (no coefficient matched):
    # default to positive-wind direction (same as scalar path).
    fallback_rules = Pair[]
    for u in depvars
        u_spatial = ivs(u, s)
        for x in u_spatial
            odd_orders = filter(isodd, derivweights.orders[x])
            for d in odd_orders
                haskey(upwind_cache, (u, x, d)) || continue
                usi = upwind_cache[(u, x, d)]
                # Positive-wind stencil as default
                pos_expr = _upwind_stencil_expr(
                    u, x, usi.pos_offsets, usi.D_pos.stencil_coefs,
                    _idxs, bases, indexmap, s
                )
                push!(fallback_rules, (Differential(x)^d)(u) => pos_expr)
            end
        end
    end

    return vcat(wind_rules, fallback_rules)
end

# --- Mixed derivative ArrayOp rules -----------------------------------------

"""
    _build_mixed_derivative_rules(s, depvars, derivweights, indexmap, _idxs, bases)

Build FD rules for mixed cross-derivatives `(Dx * Dy)(u)` using the Cartesian
product of two 1D centred stencils.
"""
function _build_mixed_derivative_rules(s, depvars, derivweights, indexmap, _idxs, bases)
    mixed_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        for x in u_spatial
            # Need order-1 centred operator for this dimension
            haskey(derivweights.map, Differential(x)) || continue
            Dx_op = derivweights.map[Differential(x)]
            Dx_op.dx isa Number || continue  # uniform only
            x_weights = Dx_op.stencil_coefs
            x_offsets = collect(half_range(Dx_op.stencil_length))

            for y in u_spatial
                isequal(x, y) && continue
                haskey(derivweights.map, Differential(y)) || continue
                Dy_op = derivweights.map[Differential(y)]
                Dy_op.dx isa Number || continue  # uniform only
                y_weights = Dy_op.stencil_coefs
                y_offsets = collect(half_range(Dy_op.stencil_length))

                # Double sum: Σ_i Σ_j wx[i] * wy[j] * u[... + x_off[i] + y_off[j] ...]
                mixed_expr = sum(zip(x_weights, x_offsets)) do (wx, x_off)
                    sum(zip(y_weights, y_offsets)) do (wy, y_off)
                        idx_exprs = map(u_spatial) do xv
                            eq_d = indexmap[xv]
                            base_expr = _idxs[eq_d] + bases[eq_d]
                            if isequal(xv, x)
                                base_expr + x_off
                            elseif isequal(xv, y)
                                base_expr + y_off
                            else
                                base_expr
                            end
                        end
                        wx * wy * Symbolics.wrap(u_c[idx_exprs...])
                    end
                end
                push!(mixed_rules, (Differential(x) * Differential(y))(u) => mixed_expr)
            end
        end
    end
    return mixed_rules
end
