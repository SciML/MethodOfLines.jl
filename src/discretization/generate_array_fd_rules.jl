"""
Array-level finite difference rule generation for `ArrayDiscretization`.

For PDEs on uniform grids, stencil weights are identical at every interior
point in each dimension.  This module pre-computes them once and builds a
single ArrayOp expression using N symbolic index variables `_i1, _i2, ...`
(one per spatial dimension).  For non-uniform grids, per-point weights are
stored in Const-wrapped matrices indexed by symbolic grid indices.  The
ArrayOp is a genuine symbolic array operation that, when ModelingToolkit
supports native array compilation, will compile to a single vectorized loop.
Until then, MTK's `flatten_equations` scalarizes it into individual
per-point equations, preserving correctness.

Supported ArrayOp patterns:
- Centred (even-order) derivatives on uniform and non-uniform grids
- Upwind (odd-order) derivatives with UpwindScheme on uniform and non-uniform grids
- WENO (Jiang-Shu) first-order derivatives on uniform grids
- Mixed cross-derivatives on uniform and non-uniform grids
- Nonlinear Laplacian `Dx(expr * Dx(u))` on uniform and non-uniform grids
- Spherical Laplacian `r^{-2} * Dr(r^2 * Dr(u))` on uniform and non-uniform grids

Generic user-defined `FunctionalScheme` falls back to per-point computation
via `discretize_equation_at_point` from the scalar path, which supports ALL
scheme types.
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
    weight_matrix::Union{Nothing, Matrix{Float64}}  # non-uniform: stencil_length × num_interior
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
    neg_weight_matrix::Union{Nothing, Matrix{Float64}}  # non-uniform: stencil_length × num_interior
    pos_weight_matrix::Union{Nothing, Matrix{Float64}}  # non-uniform: stencil_length × num_interior
end

"""
    NonlinlapStencilInfo

Pre-computed stencil information for the nonlinear Laplacian `Dx(expr * Dx(u))`.
Contains the outer (half-offset) derivative, inner (half-offset) derivative,
and interpolation weights/offsets.

For uniform grids, weights are constant SVectors at every interior point.
For non-uniform grids, per-point weights are stored in weight matrices
(stencil_length × num_interior) indexed by symbolic grid position.
"""
struct NonlinlapStencilInfo
    outer_weights        # uniform: stencil_coefs of D_outer; non-uniform: nothing
    outer_offsets::Vector{Int}
    inner_weights        # uniform: stencil_coefs of D_inner; non-uniform: nothing
    inner_offsets::Vector{Int}
    interp_weights       # uniform: stencil_coefs of interp; non-uniform: nothing
    interp_offsets::Vector{Int}
    combined_lower_bpc::Int   # combined boundary point count, lower side
    combined_upper_bpc::Int   # combined boundary point count, upper side
    is_uniform::Bool
    # Non-uniform weight matrices (nothing for uniform grids)
    outer_weight_matrix::Union{Nothing, Matrix{Float64}}
    inner_weight_matrix::Union{Nothing, Matrix{Float64}}
    interp_weight_matrix::Union{Nothing, Matrix{Float64}}
    outer_bpc::Int       # D_outer.boundary_point_count
    inner_bpc::Int       # D_inner.boundary_point_count
    interp_bpc::Int      # interp.boundary_point_count
end

"""
    WENOStencilInfo

Pre-computed stencil information for WENO5 (Jiang-Shu) scheme.
The WENO scheme computes all substencil reconstructions at every point
and blends them with data-dependent nonlinear weights — no branching.
"""
struct WENOStencilInfo
    epsilon::Float64          # smoothness indicator regularization parameter
    offsets::Vector{Int}      # [-2, -1, 0, 1, 2] for 5-point stencil
    lower_bpc::Int            # boundary point count, lower side (= 2 for WENO5)
    upper_bpc::Int            # boundary point count, upper side (= 2 for WENO5)
    dx_val::Float64           # uniform grid spacing (WENO currently uniform only)
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
                is_uniform = D_op.dx isa Number
                wmat = if !is_uniform
                    # stencil_coefs is Vector{SVector{L,T}} — convert to L×N matrix
                    hcat(Vector.(D_op.stencil_coefs)...)
                else
                    nothing
                end
                info[(u, x, d)] = StencilInfo(
                    D_op,
                    collect(half_range(D_op.stencil_length)),
                    is_uniform,
                    wmat
                )
            end
        end
    end
    return info
end

"""
    precompute_upwind_stencils(s, depvars, derivweights)

Returns a `Dict` mapping `(u, x, d)` to an `UpwindStencilInfo` for every
(variable, spatial dim, odd derivative order) triple.  Populated when
the advection scheme is `UpwindScheme` (all odd orders) or
`FunctionalScheme`/WENO (odd orders >= 3, since WENO handles order 1
internally) and windmap operators exist.
"""
function precompute_upwind_stencils(s, depvars, derivweights)
    info = Dict{Any, UpwindStencilInfo}()
    # Upwind stencils are used for UpwindScheme (all odd orders) and for
    # FunctionalScheme/WENO (odd orders >= 3, since WENO handles order 1 internally).
    has_windmap = derivweights.advection_scheme isa UpwindScheme ||
                  (derivweights.advection_scheme isa FunctionalScheme &&
                   !isempty(derivweights.windmap[1]))
    !has_windmap && return info
    for u in depvars
        for x in ivs(u, s)
            for d in derivweights.orders[x]
                isodd(d) || continue
                Dx_d = Differential(x)^d
                haskey(derivweights.windmap[1], Dx_d) || continue
                D_neg = derivweights.windmap[1][Dx_d]  # offside=0
                D_pos = derivweights.windmap[2][Dx_d]  # offside=d+upwind_order-1
                is_uniform = D_neg.dx isa Number
                neg_wmat = !is_uniform ? hcat(Vector.(D_neg.stencil_coefs)...) : nothing
                pos_wmat = !is_uniform ? hcat(Vector.(D_pos.stencil_coefs)...) : nothing
                info[(u, x, d)] = UpwindStencilInfo(
                    D_neg,
                    D_pos,
                    collect(0:(D_neg.stencil_length - 1)),
                    collect((-D_pos.stencil_length + 1):0),
                    is_uniform,
                    neg_wmat, pos_wmat
                )
            end
        end
    end
    return info
end

"""
    precompute_nonlinlap_stencils(s, depvars, derivweights)

Returns a `Dict` mapping `(u, x)` to a `NonlinlapStencilInfo` for every
(variable, spatial dim) pair where the half-offset operators exist.
Supports both uniform and non-uniform grids.
"""
function precompute_nonlinlap_stencils(s, depvars, derivweights)
    info = Dict{Any, NonlinlapStencilInfo}()
    for u in depvars
        for x in ivs(u, s)
            haskey(derivweights.halfoffsetmap[1], Differential(x)) || continue
            haskey(derivweights.halfoffsetmap[2], Differential(x)) || continue
            haskey(derivweights.interpmap, x) || continue

            D_inner = derivweights.halfoffsetmap[1][Differential(x)]
            D_outer = derivweights.halfoffsetmap[2][Differential(x)]
            interp  = derivweights.interpmap[x]

            is_uniform = (D_inner.dx isa Number) &&
                         (D_outer.dx isa Number) &&
                         (interp.dx isa Number)

            outer_offsets  = collect((1 - div(D_outer.stencil_length, 2)):(div(D_outer.stencil_length, 2)))
            inner_offsets  = collect((1 - div(D_inner.stencil_length, 2)):(div(D_inner.stencil_length, 2)))
            interp_offsets = collect((1 - div(interp.stencil_length, 2)):(div(interp.stencil_length, 2)))

            bpc_outer  = D_outer.boundary_point_count
            bpc_inner  = D_inner.boundary_point_count
            bpc_interp = interp.boundary_point_count

            if is_uniform
                # Uniform: constant weights, tap-bounds-only BPC formula
                combined_lower_bpc = max(0,
                    1 - minimum(outer_offsets) - min(minimum(inner_offsets), minimum(interp_offsets)))
                combined_upper_bpc = max(0,
                    -1 + maximum(outer_offsets) + max(maximum(inner_offsets), maximum(interp_offsets)))

                info[(u, x)] = NonlinlapStencilInfo(
                    D_outer.stencil_coefs,
                    outer_offsets,
                    D_inner.stencil_coefs,
                    inner_offsets,
                    interp.stencil_coefs,
                    interp_offsets,
                    combined_lower_bpc,
                    combined_upper_bpc,
                    is_uniform,
                    nothing, nothing, nothing,
                    bpc_outer, bpc_inner, bpc_interp
                )
            else
                # Non-uniform: build weight matrices and use stronger BPC formula
                # that ensures all three operators' weight matrix columns are in range.
                outer_wmat  = hcat(Vector.(D_outer.stencil_coefs)...)
                inner_wmat  = hcat(Vector.(D_inner.stencil_coefs)...)
                interp_wmat = hcat(Vector.(interp.stencil_coefs)...)

                # Non-uniform combined BPC: must keep all weight matrix column indices valid
                # AND keep tap indices within [1, N].
                combined_lower_bpc = max(
                    bpc_outer + 1,                                          # outer wmat valid range
                    bpc_inner + 1 - minimum(outer_offsets),                  # inner wmat valid range
                    bpc_interp + 1 - minimum(outer_offsets),                 # interp wmat valid range
                    1 - minimum(outer_offsets) - min(minimum(inner_offsets), minimum(interp_offsets))  # tap bounds
                )
                combined_upper_bpc = max(
                    bpc_outer,                                               # outer wmat valid range
                    bpc_inner + maximum(outer_offsets) - 1,                   # inner wmat valid range
                    bpc_interp + maximum(outer_offsets) - 1,                  # interp wmat valid range
                    -1 + maximum(outer_offsets) + max(maximum(inner_offsets), maximum(interp_offsets))  # tap bounds
                )

                info[(u, x)] = NonlinlapStencilInfo(
                    nothing,
                    outer_offsets,
                    nothing,
                    inner_offsets,
                    nothing,
                    interp_offsets,
                    combined_lower_bpc,
                    combined_upper_bpc,
                    is_uniform,
                    outer_wmat, inner_wmat, interp_wmat,
                    bpc_outer, bpc_inner, bpc_interp
                )
            end
        end
    end
    return info
end

"""
    precompute_weno_stencils(s, depvars, derivweights)

Returns a `Dict` mapping `(u, x)` to a `WENOStencilInfo` for every
(variable, spatial dim) pair when the advection scheme is WENO.
Only populated for uniform grids (WENO is currently uniform-only).
"""
function precompute_weno_stencils(s, depvars, derivweights)
    info = Dict{Any, WENOStencilInfo}()
    !(derivweights.advection_scheme isa FunctionalScheme) && return info
    F = derivweights.advection_scheme
    F.name != "WENO" && return info   # Only handle known WENO, not generic FunctionalScheme

    for u in depvars
        for x in ivs(u, s)
            dx = s.dxs[x]
            dx isa Number || continue   # WENO uniform only (is_nonuniform = false)
            epsilon = isempty(F.ps) ? 1e-6 : F.ps[1]
            bpc = div(F.interior_points, 2)  # = 2 for WENO5
            info[(u, x)] = WENOStencilInfo(
                epsilon,
                collect(half_range(F.interior_points)),
                bpc, bpc,
                Float64(dx)
            )
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

# --- nonlinear Laplacian detection ------------------------------------------

"""
    _detect_nonlinlap_terms(pde, s, depvars, exclude_terms=Dict())

Scan PDE terms for nonlinear Laplacian patterns `Dx(expr * Dx(u))`.
Returns the set of matched terms (symbolic expressions).
Terms in `exclude_terms` (e.g., spherical-matched terms) are skipped.
"""
function _detect_nonlinlap_terms(pde, s, depvars, exclude_terms=Dict{Any, NamedTuple}())
    terms = split_terms(pde, s.x̄)
    matched = Set{Any}()
    for u in depvars
        for x in ivs(u, s)
            rules = [
                @rule(*(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => true),
                @rule($(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => true),
                @rule($(Differential(x))($(Differential(x))(u) / ~a) => true),
                @rule(*(~~b, $(Differential(x))($(Differential(x))(u) / ~a), ~~c) => true),
                @rule(/(*(~~b, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~d)), ~~c), ~e) => true),
            ]
            for t in terms
                haskey(exclude_terms, t) && continue
                for r in rules
                    if r(t) !== nothing
                        push!(matched, t)
                        break
                    end
                end
            end
        end
    end
    return matched
end

# --- spherical Laplacian detection ------------------------------------------

"""
    _detect_spherical_terms(pde, s, depvars)

Scan PDE terms for spherical Laplacian patterns `r^{-2} * Dr(r^2 * Dr(u))`.
Returns a `Dict` mapping each matched term to a `NamedTuple` with
`(u, r, innerexpr, outer_coeff)` for template building.
"""
function _detect_spherical_terms(pde, s, depvars)
    # Use split_additive_terms (NOT split_terms) to preserve the complete
    # spherical expression `1/r^2 * Dr(r^2 * Dr(u))` as a single term.
    # split_terms(pde, s.x̄) decomposes it into pieces that the patterns
    # cannot match.
    terms = split_additive_terms(pde)
    matched = Dict{Any, NamedTuple}()
    for u in depvars
        for r in ivs(u, s)
            # Pattern 1: *(~~a, 1/(r^2), Dr(*(~~c, r^2, ~~d, Dr(u), ~~e)), ~~b)
            rule1 = @rule *(
                ~~a,
                1 / (r^2),
                $(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e)),
                ~~b
            ) => (
                u = u, r = r,
                innerexpr = *(~c..., ~d..., ~e..., Num(1)),
                outer_coeff = *(~a..., ~b..., Num(1))
            )

            # Pattern 2: /(*(~~a, Dr(*(~~c, r^2, ~~d, Dr(u), ~~e)), ~~b), r^2)
            rule2 = @rule /(
                *(
                    ~~a, $(Differential(r))(
                        *(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e)
                    ), ~~b
                ),
                (r^2)
            ) => (
                u = u, r = r,
                innerexpr = *(~c..., ~d..., ~e..., Num(1)),
                outer_coeff = *(~a..., ~b..., Num(1))
            )

            # Pattern 3: /(Dr(*(~~c, r^2, ~~d, Dr(u), ~~e)), r^2)
            rule3 = @rule /(
                ($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))),
                (r^2)
            ) => (
                u = u, r = r,
                innerexpr = *(~c..., ~d..., ~e..., Num(1)),
                outer_coeff = Num(1)
            )

            rules = [rule1, rule2, rule3]
            for t in terms
                haskey(matched, t) && continue
                for rl in rules
                    result = rl(t)
                    if result !== nothing
                        matched[t] = result
                        break
                    end
                end
            end
        end
    end
    return matched
end

# --- equation comparison ----------------------------------------------------

"""
    _equations_match(eq_template, eq_scalar)

Compare two equations for equivalence.  First tries exact structural comparison
via `isequal`.  If that fails, falls back to numerical comparison by
substituting random values for all free symbolic variables.  This handles
mathematically equivalent expressions that differ only in symbolic form
(e.g., different sign distribution or term ordering).
"""
function _equations_match(eq_template, eq_scalar)
    # Fast path: exact structural match
    if isequal(eq_template.lhs, eq_scalar.lhs) &&
       isequal(eq_template.rhs, eq_scalar.rhs)
        return true
    end
    # Slow path: numerical comparison
    # The difference lhs1 - lhs2 (and rhs1 - rhs2) should be zero if equal.
    # Time derivatives cancel since they're structurally identical.
    diff_lhs = eq_template.lhs - eq_scalar.lhs
    diff_rhs = eq_template.rhs - eq_scalar.rhs
    diff_expr = diff_lhs - diff_rhs
    all_vars = Symbolics.get_variables(diff_expr)
    isempty(all_vars) && return isequal(Symbolics.value(diff_expr), 0)
    for _ in 1:3
        subs = Dict(v => 0.5 + rand() for v in all_vars)
        val = Symbolics.value(substitute(diff_expr, subs))
        if !(val isa Number) || abs(val) > 1e-8
            return false
        end
    end
    return true
end

# --- interior equation generation -------------------------------------------

"""
    generate_array_interior_eqs(s, depvars, pde, derivweights, bcmap, eqvar,
                                 indexmap, boundaryvalfuncs, interior_ranges)

Generate discretised interior equations.

For the interior region, a single ArrayOp equation is produced when possible.
Supported patterns:
- Centred (even-order) derivatives on uniform and non-uniform grids
- Upwind (odd-order) derivatives with UpwindScheme on uniform and non-uniform grids
- WENO (Jiang-Shu) first-order derivatives on uniform grids
- Mixed cross-derivatives on uniform and non-uniform grids
- Nonlinear Laplacian `Dx(expr * Dx(u))` on uniform and non-uniform grids
- Spherical Laplacian `r^{-2} * Dr(r^2 * Dr(u))` on uniform and non-uniform grids

Boundary-proximity interior points (the "frame" around the centred region)
fall back to per-point computation via `discretize_equation_at_point`.

Generic user-defined `FunctionalScheme` falls back entirely to per-point
computation, which supports ALL scheme types.
"""
function generate_array_interior_eqs(
        s, depvars, pde, derivweights, bcmap, eqvar,
        indexmap, boundaryvalfuncs, interior_ranges
    )
    stencil_cache = precompute_stencils(s, depvars, derivweights)
    upwind_cache = precompute_upwind_stencils(s, depvars, derivweights)
    nonlinlap_cache = precompute_nonlinlap_stencils(s, depvars, derivweights)
    weno_cache = precompute_weno_stencils(s, depvars, derivweights)

    # -- determine whether the ArrayOp path can handle this PDE ---------------
    has_odd_orders = any(
        any(isodd(d) for d in derivweights.orders[x])
        for u in depvars for x in ivs(u, s)
    )
    can_upwind = has_odd_orders && derivweights.advection_scheme isa UpwindScheme

    # Detect spherical Laplacian terms first (they take priority over nonlinlap).
    spherical_terms_info = _detect_spherical_terms(pde, s, depvars)
    has_spherical = !isempty(spherical_terms_info) && !isempty(nonlinlap_cache)

    # Detect nonlinear Laplacian terms -- their odd-order derivatives are
    # handled internally, so they don't block the template path.
    # Exclude spherical-matched terms to prevent double-matching.
    nonlinlap_terms = _detect_nonlinlap_terms(pde, s, depvars, spherical_terms_info)
    has_nonlinlap = !isempty(nonlinlap_terms) && !isempty(nonlinlap_cache)

    has_weno = !isempty(weno_cache)
    can_template = !has_odd_orders || can_upwind || has_nonlinlap || has_spherical || has_weno

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

    # -- N-D ArrayOp path ------------------------------------------------------
    lo_vec = [r[1] for r in interior_ranges]
    hi_vec = [r[2] for r in interior_ranges]
    eqvar_ivs = ivs(eqvar, s)
    gl_vec = [length(s, x) for x in eqvar_ivs]

    # Detect which dimensions are periodic: both lower and upper interface
    # boundaries are present, meaning the domain wraps around.
    is_periodic = map(eqvar_ivs) do x
        bs = filter_interfaces(bcmap[operation(eqvar)][x])
        hl, hu = haslowerupper(bs, x)
        hl && hu
    end

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
                    max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], bpc)
                    max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], bpc)
                elseif isodd(d) && haskey(upwind_cache, (u, x, d))
                    usi = upwind_cache[(u, x, d)]
                    lower_bpc = max(usi.D_neg.offside, usi.D_pos.offside)
                    upper_bpc = max(usi.D_neg.boundary_point_count, usi.D_pos.boundary_point_count)
                    max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], lower_bpc)
                    max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], upper_bpc)
                end
            end
            # Nonlinear Laplacian combined stencil extent
            if has_nonlinlap && haskey(nonlinlap_cache, (u, x))
                nsi = nonlinlap_cache[(u, x)]
                max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], nsi.combined_lower_bpc)
                max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], nsi.combined_upper_bpc)
            end
            # Spherical Laplacian stencil extent: combines D1, D2, and nonlinlap reach
            if has_spherical && haskey(nonlinlap_cache, (u, x))
                nsi = nonlinlap_cache[(u, x)]
                D1_op = derivweights.map[Differential(x)]
                D2_op = derivweights.map[Differential(x)^2]
                d1_bpc = D1_op.boundary_point_count
                d2_bpc = D2_op.boundary_point_count
                sph_lower = max(nsi.combined_lower_bpc, d1_bpc, d2_bpc)
                sph_upper = max(nsi.combined_upper_bpc, d1_bpc, d2_bpc)
                max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], sph_lower)
                max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], sph_upper)
            end
            # WENO stencil extent
            if has_weno && haskey(weno_cache, (u, x))
                wsi = weno_cache[(u, x)]
                max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], wsi.lower_bpc)
                max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], wsi.upper_bpc)
            end
        end
    end

    # For periodic dimensions: no boundary frame needed (stencil wraps around),
    # so the ArrayOp covers the full grid.
    lo_centered = [is_periodic[d] ? lo_vec[d] : max(lo_vec[d], max_lower_bpc[d] + 1) for d in 1:ndim]
    hi_centered = [is_periodic[d] ? hi_vec[d] : min(hi_vec[d], gl_vec[d] - max_upper_bpc[d]) for d in 1:ndim]
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
            stencil_cache, upwind_cache, nonlinlap_cache,
            spherical_terms_info, weno_cache, bcmap, eqvar, indexmap,
            is_periodic, gl_vec
        )
        # Validate: compare the first instantiated equation against the
        # scalar path for the same point.  This catches any mismatch from
        # unsupported derivative patterns that the template cannot handle.
        #
        # For periodic dimensions, the template uses IfElse.ifelse for index
        # wrapping, which doesn't simplify in the symbolic system even with
        # concrete condition values (e.g., `ifelse(1 <= 1, 5, 1)` stays as-is).
        # This causes structural and numerical comparison to fail against the
        # scalar path (which uses concrete wrapped indices).  We skip validation
        # for periodic problems and rely on the numerical Array-vs-Scalar tests.
        has_periodic = any(is_periodic)
        if !has_periodic
            II_check = CartesianIndex(Tuple(lo_centered))
            eq_scalar = discretize_equation_at_point(
                II_check, s, depvars, pde, derivweights, bcmap,
                eqvar, indexmap, boundaryvalfuncs
            )
        end
        if has_periodic || _equations_match(eq_first, eq_scalar)
            candidate
        else
            @debug "ArrayOp validation failed" eq_first eq_scalar
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

# --- Term-level + FD substitution helper ------------------------------------

"""
    _substitute_terms(expr, termlevel_dict, rdict, do_expand)

Process `expr` by splitting it into additive terms.  Terms that match a key
in `termlevel_dict` are replaced with the precomputed template value.  All
other terms are processed via `pde_substitute(term, rdict)`.

This avoids passing template values (which contain `Const`-wrapped arrays
with symbolic indices) through `pde_substitute`, whose `maketerm`
reconstruction would try to literally index concrete arrays.
"""
function _substitute_terms(expr, termlevel_dict, rdict, do_expand)
    uw = Symbolics.unwrap(expr)
    if SymbolicUtils.iscall(uw) && SymbolicUtils.operation(uw) == +
        additive_terms = SymbolicUtils.arguments(uw)
    else
        additive_terms = [uw]
    end
    processed = map(additive_terms) do term
        if haskey(termlevel_dict, term)
            # Already-discretized template — use directly, skip pde_substitute.
            Symbolics.unwrap(termlevel_dict[term])
        else
            t_wrapped = Symbolics.wrap(term)
            result = do_expand ?
                expand_derivatives(pde_substitute(t_wrapped, rdict)) :
                pde_substitute(t_wrapped, rdict)
            Symbolics.unwrap(result)
        end
    end
    return Symbolics.wrap(sum(Symbolics.wrap, processed))
end

# --- Periodic index wrapping helper ------------------------------------------

"""
    _wrap_periodic_idx(raw_idx, N)

Wrap `raw_idx` for periodic boundary conditions, mirroring the wrapping logic
in `_wrapperiodic` from `interface_boundary.jl` (lines 36-45).

For periodic grids, index 1 is the duplicate boundary point (same as index N).
Interior indices are `2:N`.  The wrapping maps:
- index ≤ 1 → index + (N-1)   (e.g., 1 → N, 0 → N-1)
- index > N → index - (N-1)   (e.g., N+1 → 2)

Uses `IfElse.ifelse` for symbolic compatibility.  Only handles a single wrap
(stencil extends at most one grid length past the boundary).
"""
function _wrap_periodic_idx(raw_idx, N)
    IfElse.ifelse(raw_idx <= 1, raw_idx + (N - 1),
        IfElse.ifelse(raw_idx > N, raw_idx - (N - 1), raw_idx))
end

"""
    _maybe_wrap(raw_idx, dim, is_periodic, gl_vec)

If dimension `dim` is periodic, wrap `raw_idx` into `[2, gl_vec[dim]]`.
Otherwise return `raw_idx` unchanged.
"""
_maybe_wrap(raw_idx, dim, is_periodic, gl_vec) =
    is_periodic[dim] ? _wrap_periodic_idx(raw_idx, gl_vec[dim]) : raw_idx

# --- ArrayOp construction for interior region --------------------------------

"""
    _build_interior_arrayop(n_centered, lo_centered, s, depvars, pde,
                             derivweights, stencil_cache, upwind_cache,
                             nonlinlap_cache, spherical_terms_info,
                             weno_cache, bcmap, eqvar, indexmap,
                             is_periodic, gl_vec)

Build a single ArrayOp equation for the interior region.

Handles centred (even-order), upwind (odd-order), WENO (1st-order), mixed
cross-derivative, nonlinear Laplacian, and spherical Laplacian stencils
using symbolic index variables.

For periodic dimensions, stencil indices are wrapped using `_wrap_periodic_idx`
so the ArrayOp covers the full grid without a boundary frame.

Returns `(eqs, eq_first)` where `eqs` is a single-element vector containing
the ArrayOp equation, and `eq_first` is the scalar equation at the first
centred point (for validation against the scalar path).
"""
function _build_interior_arrayop(
        n_centered, lo_centered, s, depvars, pde, derivweights,
        stencil_cache, upwind_cache, nonlinlap_cache,
        spherical_terms_info, weno_cache, bcmap, eqvar, indexmap,
        is_periodic=falses(length(n_centered)),
        gl_vec=zeros(Int, length(n_centered))
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
                taps = map(si.offsets) do off
                    idx_exprs = map(u_spatial) do xv
                        eq_d = indexmap[xv]
                        raw_idx = _idxs[eq_d] + bases[eq_d]
                        raw_idx = isequal(xv, x) ? raw_idx + off : raw_idx
                        _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
                    end
                    Symbolics.wrap(u_c[idx_exprs...])
                end
                if si.is_uniform
                    expr = sym_dot(si.D_op.stencil_coefs, taps)
                else
                    # Non-uniform: index into weight matrix with symbolic point index.
                    # Weight matrix column i = weights for grid point (bpc + i).
                    # ArrayOp point k is at grid position (k + bases[dim]).
                    # So weight column = k + bases[dim] - bpc.
                    wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(si.weight_matrix)
                    dim = indexmap[x]
                    bpc = si.D_op.boundary_point_count
                    point_idx = _idxs[dim] + bases[dim] - bpc
                    expr = sum(1:length(si.offsets)) do k
                        Symbolics.wrap(wmat_c[k, point_idx]) * taps[k]
                    end
                end
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
            bcmap, indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec
        )
    end

    # -- Mixed derivative rules -----------------------------------------------
    mixed_rules = _build_mixed_derivative_rules(
        s, depvars, derivweights, indexmap, _idxs, bases,
        is_periodic, gl_vec
    )

    # -- Nonlinear Laplacian rules --------------------------------------------
    nl_rules = Pair[]
    if !isempty(nonlinlap_cache)
        nl_rules = _build_nonlinlap_rules(
            pde, s, depvars, derivweights, nonlinlap_cache,
            indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec
        )
    end

    # -- Spherical Laplacian rules -------------------------------------------
    sph_rules = Pair[]
    if !isempty(spherical_terms_info) && !isempty(nonlinlap_cache)
        sph_rules = _build_spherical_rules(
            pde, s, depvars, derivweights, nonlinlap_cache,
            spherical_terms_info, indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec
        )
    end

    # -- WENO rules ----------------------------------------------------------
    weno_rules = Pair[]
    if !isempty(weno_cache)
        weno_rules = _build_weno_rules(
            pde, s, depvars, weno_cache,
            indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec
        )
    end

    # -- Build templates (once) -----------------------------------------------
    # Upwind, nonlinear Laplacian, spherical Laplacian, and WENO rules are
    # term-level substitutions (they replace entire additive terms, not just
    # derivative sub-expressions).  Apply them first on the PDE expression,
    # then apply FD + var rules.
    all_fd_rules = vcat(fd_rules, mixed_rules)
    rdict = Dict(vcat(all_fd_rules, var_rules))

    termlevel_rules = vcat(sph_rules, upwind_rules, nl_rules, weno_rules)
    if isempty(termlevel_rules)
        template_lhs = expand_derivatives(pde_substitute(pde.lhs, rdict))
        template_rhs = pde_substitute(pde.rhs, rdict)
    else
        # Process each additive term separately.  Term-level templates
        # (spherical, upwind, nonlinlap) contain Const-wrapped arrays with
        # symbolic indices that pde_substitute cannot safely traverse (its
        # maketerm reconstruction tries to literally index concrete arrays).
        # By processing matched and unmatched terms independently we avoid
        # passing templates through pde_substitute.
        termlevel_dict = Dict(Symbolics.unwrap(k) => v for (k, v) in termlevel_rules)
        template_lhs = _substitute_terms(pde.lhs, termlevel_dict, rdict, true)
        template_rhs = _substitute_terms(pde.rhs, termlevel_dict, rdict, false)
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
        bcmap, indexmap, _idxs, bases, var_rules,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases))
    )
    # Helper: build stencil expression for a given variable, dimension, offsets, weights.
    # For non-uniform grids, weight_matrix is a stencil_length × num_interior Matrix
    # and bpc is the offside (= low_boundary_point_count) used to align weight matrix
    # column indexing: stencil_coefs[j] corresponds to grid index (j + bpc).
    function _upwind_stencil_expr(u, x, offsets, weights, _idxs, bases, indexmap, s;
                                   weight_matrix=nothing, bpc=0)
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        taps = map(offsets) do off
            idx_exprs = map(u_spatial) do xv
                eq_d = indexmap[xv]
                raw_idx = _idxs[eq_d] + bases[eq_d]
                raw_idx = isequal(xv, x) ? raw_idx + off : raw_idx
                _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
            end
            Symbolics.wrap(u_c[idx_exprs...])
        end
        if weight_matrix === nothing
            # Uniform: constant weights
            return sym_dot(weights, taps)
        else
            # Non-uniform: index into weight matrix by interior point index
            wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(weight_matrix)
            dim = indexmap[x]
            point_idx = _idxs[dim] + bases[dim] - bpc
            return sum(1:length(offsets)) do k
                Symbolics.wrap(wmat_c[k, point_idx]) * taps[k]
            end
        end
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
                    _idxs, bases, indexmap, s;
                    weight_matrix=usi.neg_weight_matrix,
                    bpc=usi.D_neg.offside
                )
                pos_expr = _upwind_stencil_expr(
                    u, x, usi.pos_offsets, usi.D_pos.stencil_coefs,
                    _idxs, bases, indexmap, s;
                    weight_matrix=usi.pos_weight_matrix,
                    bpc=usi.D_pos.offside
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
                    _idxs, bases, indexmap, s;
                    weight_matrix=usi.pos_weight_matrix,
                    bpc=usi.D_pos.offside
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
function _build_mixed_derivative_rules(s, depvars, derivweights, indexmap, _idxs, bases,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases)))
    mixed_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        for x in u_spatial
            # Need order-1 centred operator for this dimension
            haskey(derivweights.map, Differential(x)) || continue
            Dx_op = derivweights.map[Differential(x)]
            x_is_uniform = Dx_op.dx isa Number
            x_offsets = collect(half_range(Dx_op.stencil_length))

            # For non-uniform: build weight matrix and Const-wrap it
            if x_is_uniform
                x_weights = Dx_op.stencil_coefs
            else
                x_wmat = hcat(Vector.(Dx_op.stencil_coefs)...)
                x_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(x_wmat)
                x_bpc = Dx_op.boundary_point_count
            end

            for y in u_spatial
                isequal(x, y) && continue
                haskey(derivweights.map, Differential(y)) || continue
                Dy_op = derivweights.map[Differential(y)]
                y_is_uniform = Dy_op.dx isa Number
                y_offsets = collect(half_range(Dy_op.stencil_length))

                if y_is_uniform
                    y_weights = Dy_op.stencil_coefs
                else
                    y_wmat = hcat(Vector.(Dy_op.stencil_coefs)...)
                    y_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(y_wmat)
                    y_bpc = Dy_op.boundary_point_count
                end

                dim_x = indexmap[x]
                dim_y = indexmap[y]

                # Double sum: Σ_i Σ_j wx[i] * wy[j] * u[... + x_off[i] + y_off[j] ...]
                mixed_expr = sum(enumerate(x_offsets)) do (kx, x_off)
                    sum(enumerate(y_offsets)) do (ky, y_off)
                        idx_exprs = map(u_spatial) do xv
                            eq_d = indexmap[xv]
                            raw_idx = _idxs[eq_d] + bases[eq_d]
                            if isequal(xv, x)
                                raw_idx = raw_idx + x_off
                            elseif isequal(xv, y)
                                raw_idx = raw_idx + y_off
                            end
                            _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
                        end
                        tap = Symbolics.wrap(u_c[idx_exprs...])

                        wx = if x_is_uniform
                            x_weights[kx]
                        else
                            Symbolics.wrap(x_wmat_c[kx, _idxs[dim_x] + bases[dim_x] - x_bpc])
                        end

                        wy = if y_is_uniform
                            y_weights[ky]
                        else
                            Symbolics.wrap(y_wmat_c[ky, _idxs[dim_y] + bases[dim_y] - y_bpc])
                        end

                        wx * wy * tap
                    end
                end
                push!(mixed_rules, (Differential(x) * Differential(y))(u) => mixed_expr)
            end
        end
    end
    return mixed_rules
end

# --- Nonlinear Laplacian ArrayOp rules --------------------------------------

"""
    _nonlinlap_template(expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases)

Build the ArrayOp-indexed discretization of `Dx(expr_sym * Dx(u))` using the
precomputed `NonlinlapStencilInfo`.

At each outer stencil half-point:
1. Interpolate all depvars and grid coordinates to the half-point
2. Compute the inner derivative of `u` at the half-point
3. Substitute rules into `expr_sym * Dx(u)` to get the inner expression

Then take the outer finite difference across the inner expressions.
"""
function _nonlinlap_template(expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases)))
    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
    u_spatial = ivs(u, s)
    dim = indexmap[x]

    # Pre-wrap Const weight matrices for non-uniform case (outside the loop)
    if !nsi.is_uniform
        interp_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(nsi.interp_weight_matrix)
        inner_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(nsi.inner_weight_matrix)
        outer_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(nsi.outer_weight_matrix)
    end

    inner_exprs = map(nsi.outer_offsets) do outer_off
        # --- Interpolation rules for variables at this half-point ---
        interp_var_rules = Pair[]
        for v in depvars
            v_raw = Symbolics.unwrap(s.discvars[v])
            v_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(v_raw)
            v_spatial = ivs(v, s)
            taps = map(nsi.interp_offsets) do ioff
                idx_exprs = map(v_spatial) do xv
                    eq_d = indexmap[xv]
                    raw_idx = _idxs[eq_d] + bases[eq_d]
                    # Offset: -1 for clipped grid, + outer_off, + interp offset
                    raw_idx = isequal(xv, x) ? raw_idx + outer_off + ioff - 1 : raw_idx
                    _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
                end
                Symbolics.wrap(v_c[idx_exprs...])
            end
            if nsi.is_uniform
                push!(interp_var_rules, v => sym_dot(nsi.interp_weights, taps))
            else
                interp_pt_idx = _idxs[dim] + bases[dim] + outer_off - 1 - nsi.interp_bpc
                interp_expr = sum(1:length(nsi.interp_offsets)) do k
                    Symbolics.wrap(interp_wmat_c[k, interp_pt_idx]) * taps[k]
                end
                push!(interp_var_rules, v => interp_expr)
            end
        end

        # --- Interpolation rules for grid coordinates ---
        interp_iv_rules = Pair[]
        for xv in s.x̄
            if isequal(xv, x)
                grid_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(collect(s.grid[x]))
                taps = map(nsi.interp_offsets) do ioff
                    raw_idx = _idxs[dim] + bases[dim] + outer_off + ioff - 1
                    Symbolics.wrap(grid_c[_maybe_wrap(raw_idx, dim, is_periodic, gl_vec)])
                end
                if nsi.is_uniform
                    push!(interp_iv_rules, x => sym_dot(nsi.interp_weights, taps))
                else
                    interp_pt_idx = _idxs[dim] + bases[dim] + outer_off - 1 - nsi.interp_bpc
                    interp_expr = sum(1:length(nsi.interp_offsets)) do k
                        Symbolics.wrap(interp_wmat_c[k, interp_pt_idx]) * taps[k]
                    end
                    push!(interp_iv_rules, x => interp_expr)
                end
            else
                haskey(indexmap, xv) || continue
                grid_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(collect(s.grid[xv]))
                xv_dim = indexmap[xv]
                push!(interp_iv_rules, xv => Symbolics.wrap(grid_c[_idxs[xv_dim] + bases[xv_dim]]))
            end
        end

        # --- Inner derivative of u at this half-point: Dx(u) ---
        inner_deriv_taps = map(nsi.inner_offsets) do ioff
            idx_exprs = map(u_spatial) do xv
                eq_d = indexmap[xv]
                raw_idx = _idxs[eq_d] + bases[eq_d]
                raw_idx = isequal(xv, x) ? raw_idx + outer_off + ioff - 1 : raw_idx
                _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
            end
            Symbolics.wrap(u_c[idx_exprs...])
        end
        if nsi.is_uniform
            inner_deriv = sym_dot(nsi.inner_weights, inner_deriv_taps)
        else
            inner_pt_idx = _idxs[dim] + bases[dim] + outer_off - 1 - nsi.inner_bpc
            inner_deriv = sum(1:length(nsi.inner_offsets)) do k
                Symbolics.wrap(inner_wmat_c[k, inner_pt_idx]) * inner_deriv_taps[k]
            end
        end

        # --- Substitute all rules into expr * Dx(u) ---
        deriv_rules = Pair[Differential(x)(u) => inner_deriv]
        all_rules = Dict(vcat(deriv_rules, interp_var_rules, interp_iv_rules))
        substitute(expr_sym * Differential(x)(u), all_rules)
    end

    # Apply outer weights to get the full nonlinear Laplacian
    if nsi.is_uniform
        return sym_dot(nsi.outer_weights, inner_exprs)
    else
        outer_pt_idx = _idxs[dim] + bases[dim] - 1 - nsi.outer_bpc
        return sum(1:length(nsi.outer_offsets)) do k
            Symbolics.wrap(outer_wmat_c[k, outer_pt_idx]) * inner_exprs[k]
        end
    end
end

"""
    _build_nonlinlap_rules(pde, s, depvars, derivweights, nonlinlap_cache,
                            indexmap, _idxs, bases, var_rules)

Build term-level substitution rules for nonlinear Laplacian patterns.

Uses the same pattern-matching approach as `generate_nonlinlap_rules` from the
scalar path, but substitutes ArrayOp-parameterized stencils.

Returns a vector of `Pair{term => discretized_expr}`.
"""
function _build_nonlinlap_rules(
        pde, s, depvars, derivweights, nonlinlap_cache,
        indexmap, _idxs, bases, var_rules,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases))
    )
    terms = split_terms(pde, s.x̄)
    vr_dict = Dict(var_rules)
    nonlinlap_rules = Pair[]

    for u in depvars
        for x in ivs(u, s)
            haskey(nonlinlap_cache, (u, x)) || continue
            nsi = nonlinlap_cache[(u, x)]

            # Pattern 1: *(~~c, Dx(*(~~a, Dx(u), ~~b)), ~~d)
            rule_mul = @rule *(
                ~~c,
                $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)),
                ~~d
            ) => begin
                expr_sym = *(~a..., ~b...)
                outer_coeff = *(~c..., ~d...)
                outer_coeff_subst = pde_substitute(outer_coeff, vr_dict)
                nlap = _nonlinlap_template(
                    expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases,
                    is_periodic, gl_vec
                )
                outer_coeff_subst * nlap
            end

            # Pattern 2: Dx(*(~~a, Dx(u), ~~b))
            rule_standalone = @rule $(Differential(x))(
                *(~~a, $(Differential(x))(u), ~~b)
            ) => begin
                expr_sym = *(~a..., ~b...)
                _nonlinlap_template(
                    expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases,
                    is_periodic, gl_vec
                )
            end

            # Pattern 3: Dx(Dx(u) / ~a)
            rule_div = @rule $(Differential(x))(
                $(Differential(x))(u) / ~a
            ) => begin
                expr_sym = 1 / ~a
                _nonlinlap_template(
                    expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases,
                    is_periodic, gl_vec
                )
            end

            # Pattern 4: *(~~b, Dx(Dx(u) / ~a), ~~c)
            rule_mul_div = @rule *(
                ~~b,
                $(Differential(x))($(Differential(x))(u) / ~a),
                ~~c
            ) => begin
                expr_sym = 1 / ~a
                outer_coeff = *(~b..., ~c...)
                outer_coeff_subst = pde_substitute(outer_coeff, vr_dict)
                nlap = _nonlinlap_template(
                    expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases,
                    is_periodic, gl_vec
                )
                outer_coeff_subst * nlap
            end

            # Pattern 5: /(*(~~b, Dx(*(~~a, Dx(u), ~~d)), ~~c), ~e)
            rule_full_div = @rule /(
                *(~~b, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~d)), ~~c),
                ~e
            ) => begin
                expr_sym = *(~a..., ~d...)
                outer_coeff = *(~b..., ~c...) / ~e
                outer_coeff_subst = pde_substitute(outer_coeff, vr_dict)
                nlap = _nonlinlap_template(
                    expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases,
                    is_periodic, gl_vec
                )
                outer_coeff_subst * nlap
            end

            # Try matching each term
            all_rules = [rule_mul, rule_standalone, rule_div, rule_mul_div, rule_full_div]
            for t in terms
                for r in all_rules
                    matched = r(t)
                    if matched !== nothing
                        push!(nonlinlap_rules, t => matched)
                        break
                    end
                end
            end
        end
    end

    return nonlinlap_rules
end

# --- Spherical Laplacian ArrayOp rules --------------------------------------

"""
    _spherical_template(info, nsi, s, depvars, derivweights,
                         indexmap, _idxs, bases, var_rules)

Build the ArrayOp-indexed discretization of the spherical Laplacian
`r^{-2} * Dr(r^2 * innerexpr * Dr(u))`.

At r ≈ 0:   `6 * innerexpr * D2(u)`   (L'Hôpital's rule)
At r ≠ 0:   `innerexpr * (D1(u)/r + nonlinlap(innerexpr, u, r))`

Uses `IfElse.ifelse` for the r ≈ 0 conditional (same pattern as upwind wind
switching).
"""
function _spherical_template(info, nsi, s, depvars, derivweights,
                              indexmap, _idxs, bases, var_rules,
                              is_periodic=falses(length(bases)),
                              gl_vec=zeros(Int, length(bases)))
    u = info.u
    r = info.r
    innerexpr = info.innerexpr

    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
    u_spatial = ivs(u, s)

    # Compute the grid coordinate at the current ArrayOp index.
    # Use Const-wrapped grid lookup (works for both uniform and non-uniform).
    # Safe because the result goes into termlevel_dict which bypasses pde_substitute.
    grid_r = collect(s.grid[r])
    dim = indexmap[r]
    grid_r_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(grid_r)
    r_at_i = Symbolics.wrap(grid_r_c[_idxs[dim] + bases[dim]])

    # The ArrayOp centred region never includes r = 0 (which is handled by
    # boundary conditions), so we always use the r ≠ 0 branch:
    #   innerexpr * (D1(u)/r + cartesian_nonlinear_laplacian(innerexpr, u, r))

    # --- Centered 1st derivative template ---
    D1_op = derivweights.map[Differential(r)]
    d1_is_uniform = D1_op.dx isa Number
    d1_offsets = collect(half_range(D1_op.stencil_length))
    d1_taps = map(d1_offsets) do off
        idx_exprs = map(u_spatial) do xv
            eq_d = indexmap[xv]
            raw_idx = _idxs[eq_d] + bases[eq_d]
            raw_idx = isequal(xv, r) ? raw_idx + off : raw_idx
            _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
        end
        Symbolics.wrap(u_c[idx_exprs...])
    end
    if d1_is_uniform
        D1_template = sym_dot(D1_op.stencil_coefs, d1_taps)
    else
        d1_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(
            hcat(Vector.(D1_op.stencil_coefs)...)
        )
        d1_pt_idx = _idxs[dim] + bases[dim] - D1_op.boundary_point_count
        D1_template = sum(1:length(d1_offsets)) do k
            Symbolics.wrap(d1_wmat_c[k, d1_pt_idx]) * d1_taps[k]
        end
    end

    # --- Nonlinear Laplacian template (reuse existing infrastructure) ---
    nlap_template = _nonlinlap_template(
        innerexpr, u, r, nsi, s, depvars, indexmap, _idxs, bases,
        is_periodic, gl_vec
    )

    # --- Substitute innerexpr variables at the current point ---
    vr_dict = Dict(var_rules)
    innerexpr_at_i = pde_substitute(innerexpr, vr_dict)

    # --- Combine: innerexpr * (D1/r + nonlinlap) ---
    return innerexpr_at_i * (D1_template / r_at_i + nlap_template)
end

"""
    _build_spherical_rules(pde, s, depvars, derivweights, nonlinlap_cache,
                            spherical_terms_info, indexmap, _idxs, bases, var_rules)

Build term-level substitution rules for spherical Laplacian patterns.

For each spherical-matched term, builds the ArrayOp-indexed discretization
using `_spherical_template` and multiplies by the outer coefficient.

Returns a vector of `Pair{term => discretized_expr}`.
"""
function _build_spherical_rules(
        pde, s, depvars, derivweights, nonlinlap_cache,
        spherical_terms_info, indexmap, _idxs, bases, var_rules,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases))
    )
    vr_dict = Dict(var_rules)
    spherical_rules = Pair[]

    for (term, info) in spherical_terms_info
        haskey(nonlinlap_cache, (info.u, info.r)) || continue
        nsi = nonlinlap_cache[(info.u, info.r)]

        sph_expr = _spherical_template(
            info, nsi, s, depvars, derivweights,
            indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec
        )

        # Substitute outer_coeff variables now (the template is self-contained,
        # so the second pde_substitute pass should not need to process it).
        outer_subst = pde_substitute(info.outer_coeff, vr_dict)
        push!(spherical_rules, term => outer_subst * sph_expr)
    end

    return spherical_rules
end

# --- WENO ArrayOp rules ----------------------------------------------------

"""
    _weno_template(u, x, wsi, s, indexmap, _idxs, bases)

Build the WENO5 (Jiang-Shu) formula as a symbolic ArrayOp expression.

Transcribes the `weno_f` function from `WENO.jl` using Const-wrapped array
taps instead of runtime values.  All coefficients are Float64 literals to
match the scalar path exactly (for `_equations_match` validation).
"""
function _weno_template(u, x, wsi, s, indexmap, _idxs, bases,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases)))
    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
    u_spatial = ivs(u, s)

    # Build the 5 shifted taps: u[i-2], u[i-1], u[i], u[i+1], u[i+2]
    taps = map(wsi.offsets) do off
        idx_exprs = map(u_spatial) do xv
            eq_d = indexmap[xv]
            raw_idx = _idxs[eq_d] + bases[eq_d]
            raw_idx = isequal(xv, x) ? raw_idx + off : raw_idx
            _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
        end
        Symbolics.wrap(u_c[idx_exprs...])
    end
    # Map to weno_f naming: u_m2, u_m1, u_0, u_p1, u_p2
    u_m2, u_m1, u_0, u_p1, u_p2 = taps

    ε = wsi.epsilon
    dx = wsi.dx_val

    # --- Smoothness indicators (β values) --- same for both L and R sides
    β1 = 13 * (u_0 - 2 * u_p1 + u_p2)^2 / 12 + (3 * u_0 - 4 * u_p1 + u_p2)^2 / 4
    β2 = 13 * (u_m1 - 2 * u_0 + u_p1)^2 / 12 + (u_m1 - u_p1)^2 / 4
    β3 = 13 * (u_m2 - 2 * u_m1 + u_0)^2 / 12 + (u_m2 - 4 * u_m1 + 3 * u_0)^2 / 4

    # --- Left-biased (minus) weights and reconstructions ---
    γm1 = 1 / 10
    γm2 = 3 / 5
    γm3 = 3 / 10

    ωm1 = γm1 / (ε + β1)^2
    ωm2 = γm2 / (ε + β2)^2
    ωm3 = γm3 / (ε + β3)^2
    wm_denom = ωm1 + ωm2 + ωm3
    wm1 = ωm1 / wm_denom
    wm2 = ωm2 / wm_denom
    wm3 = ωm3 / wm_denom

    hm1 = (11 * u_0 - 7 * u_p1 + 2 * u_p2) / 6
    hm2 = (5 * u_0 - u_p1 + 2 * u_m1) / 6
    hm3 = (2 * u_0 + 5 * u_m1 - u_m2) / 6
    hm = wm1 * hm1 + wm2 * hm2 + wm3 * hm3

    # --- Right-biased (plus) weights and reconstructions ---
    γp1 = 3 / 10
    γp2 = 3 / 5
    γp3 = 1 / 10

    ωp1 = γp1 / (ε + β1)^2
    ωp2 = γp2 / (ε + β2)^2
    ωp3 = γp3 / (ε + β3)^2
    wp_denom = ωp1 + ωp2 + ωp3
    wp1 = ωp1 / wp_denom
    wp2 = ωp2 / wp_denom
    wp3 = ωp3 / wp_denom

    hp1 = (2 * u_0 + 5 * u_p1 - u_p2) / 6
    hp2 = (5 * u_0 + 2 * u_p1 - u_m1) / 6
    hp3 = (11 * u_0 - 7 * u_m1 + 2 * u_m2) / 6
    hp = wp1 * hp1 + wp2 * hp2 + wp3 * hp3

    return (hp - hm) / dx
end

"""
    _build_weno_rules(pde, s, depvars, weno_cache, indexmap, _idxs, bases, var_rules)

Build term-level substitution rules for WENO 1st-order derivatives.

Unlike upwind schemes, WENO internally handles both flux directions (left-
and right-biased reconstructions), so no IfElse wind switching is needed.
The result is the numerical derivative itself; coefficients simply scale it.

Returns a vector of `Pair{term => discretized_expr}`.
"""
function _build_weno_rules(pde, s, depvars, weno_cache, indexmap, _idxs, bases, var_rules,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases)))
    terms = split_terms(pde, s.x̄)
    vr_dict = Dict(var_rules)
    weno_rules = Pair[]

    for u in depvars
        for x in ivs(u, s)
            haskey(weno_cache, (u, x)) || continue
            wsi = weno_cache[(u, x)]

            weno_expr = _weno_template(u, x, wsi, s, indexmap, _idxs, bases,
                is_periodic, gl_vec)

            # Pattern 1: *(~~a, Dx(u), ~~b) — coefficient-multiplied 1st-order
            mul_rule = @rule *(
                ~~a,
                $(Differential(x))(u),
                ~~b
            ) => begin
                coeff = *(~a..., ~b...)
                coeff_subst = pde_substitute(coeff, vr_dict)
                coeff_subst * weno_expr
            end

            # Pattern 2: /(*(~~a, Dx(u), ~~b), ~c) — divided coefficient
            div_rule = @rule /(
                *(~~a, $(Differential(x))(u), ~~b),
                ~c
            ) => begin
                coeff = *(~a..., ~b...) / ~c
                coeff_subst = pde_substitute(coeff, vr_dict)
                coeff_subst * weno_expr
            end

            for t in terms
                matched = mul_rule(t)
                if matched !== nothing
                    push!(weno_rules, t => matched)
                    continue
                end
                matched = div_rule(t)
                if matched !== nothing
                    push!(weno_rules, t => matched)
                end
            end
        end
    end

    # Fallback: bare Dx(u) with no coefficient
    fallback_rules = Pair[]
    for u in depvars
        for x in ivs(u, s)
            haskey(weno_cache, (u, x)) || continue
            wsi = weno_cache[(u, x)]
            weno_expr = _weno_template(u, x, wsi, s, indexmap, _idxs, bases,
                is_periodic, gl_vec)
            push!(fallback_rules, Differential(x)(u) => weno_expr)
        end
    end

    return vcat(weno_rules, fallback_rules)
end
