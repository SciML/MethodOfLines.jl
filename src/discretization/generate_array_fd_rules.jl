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
- Staggered grid (odd-order) derivatives on uniform grids
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
    StaggeredStencilInfo

Pre-computed stencil information for staggered grid odd-order derivatives.
On a staggered grid, variable alignment determines a fixed stencil offset:
`CenterAlignedVar` uses `[0, 1]` (forward half-shift) and `EdgeAlignedVar`
uses `[-1, 0]` (backward half-shift).  No wind-direction switching is needed.
Currently supported on uniform grids only.
"""
struct StaggeredStencilInfo
    alignment             # CenterAlignedVar or EdgeAlignedVar (Type)
    interior_offsets::Vector{Int}   # [0,1] for CenterAligned, [-1,0] for EdgeAligned
    D_wind::DerivativeOperator      # from windmap[1]
    is_uniform::Bool
    bpc::Int                        # boundary_point_count from derivweights.map
end

"""
    _periodic_stencil_positions(grid_x, g, offsets)

Compute the physical positions of stencil points around grid point `g`
for a periodic grid, wrapping indices that fall outside `[2, N]`.

`offsets` is a vector/range of integer offsets from `g` (e.g., `[-1, 0, 1]`
for a 3-point centered stencil).

For wrapped indices, positions are shifted by ±L where L = grid_x[N] - grid_x[1]
is the domain length.
"""
function _periodic_stencil_positions(grid_x, g, offsets)
    N = length(grid_x)
    L = grid_x[N] - grid_x[1]
    positions = Vector{Float64}(undef, length(offsets))
    for (j, off) in enumerate(offsets)
        raw = g + off
        if raw <= 1
            positions[j] = grid_x[raw + (N - 1)] - L
        elseif raw > N
            positions[j] = grid_x[raw - (N - 1)] + L
        else
            positions[j] = grid_x[raw]
        end
    end
    return positions
end

"""
    _build_periodic_centered_wmat(D_op, grid_x)

Build an extended `stencil_length × N` weight matrix for a non-uniform
periodic grid.  Interior points reuse the stencil coefficients from `D_op`;
boundary-proximity points use wrapped grid positions.
"""
function _build_periodic_centered_wmat(D_op, grid_x)
    N = length(grid_x)
    sl = D_op.stencil_length
    half_w = div(sl, 2)
    offsets = -half_w:half_w

    wmat = Matrix{Float64}(undef, sl, N)
    for g in 1:N
        positions = _periodic_stencil_positions(grid_x, g, offsets)
        wmat[:, g] = calculate_weights(D_op.derivative_order, grid_x[g], positions)
    end
    return wmat
end

"""
    _build_periodic_upwind_wmat(D_op, grid_x, offsets)

Build an extended `stencil_length × N` weight matrix for upwind stencils
on a non-uniform periodic grid.  `offsets` are the tap offsets relative to
the evaluation point (e.g., `0:(sl-1)` for negative-wind).
"""
function _build_periodic_upwind_wmat(D_op, grid_x, offsets)
    N = length(grid_x)
    sl = D_op.stencil_length

    wmat = Matrix{Float64}(undef, sl, N)
    for g in 1:N
        positions = _periodic_stencil_positions(grid_x, g, offsets)
        wmat[:, g] = calculate_weights(D_op.derivative_order, grid_x[g], positions)
    end
    return wmat
end

"""
    precompute_stencils(s, depvars, derivweights; spherical_vars=nothing)

Returns a `Dict` mapping `(u, x, d)` to a `StencilInfo` for every
(variable, spatial dim, even derivative order) triple.

When `spherical_vars` is a list of `(u, r)` pairs, also caches the D1
stencil for each pair so that `precompute_full_interior_stencils` can
build `FullInteriorStencilInfo` for the first derivative used by
`_spherical_template`.
"""
function precompute_stencils(s, depvars, derivweights; spherical_vars=nothing)
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
    # Add D1 stencils for spherical variables so that
    # precompute_full_interior_stencils can build FullInteriorStencilInfo
    # for the first derivative used by _spherical_template.
    if spherical_vars !== nothing
        for (u, r) in spherical_vars
            haskey(info, (u, r, 1)) && continue
            D_op = derivweights.map[Differential(r)]
            is_uniform = D_op.dx isa Number
            wmat = if !is_uniform
                hcat(Vector.(D_op.stencil_coefs)...)
            else
                nothing
            end
            info[(u, r, 1)] = StencilInfo(
                D_op,
                collect(half_range(D_op.stencil_length)),
                is_uniform,
                wmat
            )
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
    precompute_staggered_stencils(s, depvars, derivweights)

Returns a `Dict` mapping `(u, x, d)` to a `StaggeredStencilInfo` for every
(variable, spatial dim, odd derivative order) triple when the grid is staggered
and uniform.  Non-uniform staggered grids are not supported in the ArrayOp path
and will fall back to per-point scalar discretization.
"""
function precompute_staggered_stencils(s, depvars, derivweights)
    info = Dict{Any, StaggeredStencilInfo}()
    s.staggeredvars === nothing && return info
    for u in depvars
        for x in ivs(u, s)
            for d in derivweights.orders[x]
                isodd(d) || continue
                Dx_d = Differential(x)^d
                haskey(derivweights.windmap[1], Dx_d) || continue
                D_wind = derivweights.windmap[1][Dx_d]
                is_uniform = D_wind.dx isa Number
                !is_uniform && continue  # Non-uniform staggered not supported
                alignment = s.staggeredvars[operation(u)]
                interior_offsets = if alignment === CenterAlignedVar
                    collect(0:1)
                else  # EdgeAlignedVar
                    collect(-1:0)
                end
                bpc = derivweights.map[Dx_d].boundary_point_count
                info[(u, x, d)] = StaggeredStencilInfo(
                    alignment,
                    interior_offsets,
                    D_wind,
                    is_uniform,
                    bpc
                )
            end
        end
    end
    return info
end

# --- Full-interior stencil data structures ----------------------------------

"""
    FullInteriorStencilInfo

Weight and offset matrices covering ALL interior points (including boundary-
proximity frame points) for a single centered derivative.  Boundary stencils
from `D_op.low_boundary_coefs` / `D_op.high_boundary_coefs` are folded into
position-dependent rows so that a single ArrayOp covers the entire interior.
"""
struct FullInteriorStencilInfo
    weight_matrix::Matrix{Float64}   # padded_len × N_full_interior
    offset_matrix::Matrix{Int}       # padded_len × N_full_interior
    padded_len::Int                  # max(stencil_length, boundary_stencil_length)
end

"""
    FullInteriorUpwindStencilInfo

Weight and offset matrices covering ALL interior points for upwind derivatives.
Both wind directions get their own matrices.
"""
struct FullInteriorUpwindStencilInfo
    neg_weight_matrix::Matrix{Float64}
    neg_offset_matrix::Matrix{Int}
    padded_neg::Int
    pos_weight_matrix::Matrix{Float64}
    pos_offset_matrix::Matrix{Int}
    padded_pos::Int
end

"""
    precompute_full_interior_stencils(s, depvars, derivweights, stencil_cache,
                                       lo_vec, hi_vec, indexmap, eqvar;
                                       is_periodic=falses(length(lo_vec)))

Build `FullInteriorStencilInfo` for every `(u, x, d)` key in `stencil_cache`.
The matrices cover grid indices `lo_vec[dim]..hi_vec[dim]` (the full interior).

For periodic uniform dimensions, all columns use the interior stencil
(no boundary branches).
"""
function precompute_full_interior_stencils(s, depvars, derivweights, stencil_cache,
                                            lo_vec, hi_vec, indexmap, eqvar;
                                            is_periodic=falses(length(lo_vec)))
    info = Dict{Any, FullInteriorStencilInfo}()
    eqvar_ivs = ivs(eqvar, s)
    gl_vec = [length(s, x) for x in eqvar_ivs]

    for (key, si) in stencil_cache
        u, x, d = key
        dim = indexmap[x]
        gl = gl_vec[dim]
        N = hi_vec[dim] - lo_vec[dim] + 1
        D_op = si.D_op
        bpc = D_op.boundary_point_count
        sl = D_op.stencil_length
        bsl = D_op.boundary_stencil_length
        padded = max(sl, bsl)

        wmat = zeros(Float64, padded, N)
        omat = zeros(Int, padded, N)

        for k in 1:N
            g = lo_vec[dim] + k - 1  # absolute grid index

            if is_periodic[dim]
                # Periodic uniform: always use interior stencil (wrapping handled symbolically)
                weights = Vector{Float64}(D_op.stencil_coefs)
                offsets = collect(si.offsets)
            elseif g <= bpc
                # Lower frame: use low_boundary_coefs[g]
                weights = Vector{Float64}(D_op.low_boundary_coefs[g])
                # Taps are at grid indices 1:bsl, relative offsets from g
                offsets = collect((1 - g):(bsl - g))
            elseif g > gl - bpc
                # Upper frame: use high_boundary_coefs[gl - g + 1]
                weights = Vector{Float64}(D_op.high_boundary_coefs[gl - g + 1])
                # Taps are at grid indices (gl-bsl+1):gl
                offsets = collect((gl - bsl + 1 - g):(gl - g))
            else
                # Centered interior
                if si.is_uniform
                    weights = Vector{Float64}(D_op.stencil_coefs)
                else
                    # Non-uniform: stencil_coefs is indexed by interior position
                    weights = Vector{Float64}(D_op.stencil_coefs[g - bpc])
                end
                offsets = collect(si.offsets)
            end

            # Zero-pad to padded length
            nw = length(weights)
            for j in 1:nw
                wmat[j, k] = weights[j]
                omat[j, k] = offsets[j]
            end
            # Remaining entries stay 0 (zero weight = no contribution)
        end

        info[key] = FullInteriorStencilInfo(wmat, omat, padded)
    end
    return info
end

"""
    precompute_full_interior_upwind(s, depvars, derivweights, upwind_cache,
                                     lo_vec, hi_vec, indexmap, eqvar;
                                     is_periodic=falses(length(lo_vec)))

Build `FullInteriorUpwindStencilInfo` for every `(u, x, d)` key in `upwind_cache`.
"""
function precompute_full_interior_upwind(s, depvars, derivweights, upwind_cache,
                                          lo_vec, hi_vec, indexmap, eqvar;
                                          is_periodic=falses(length(lo_vec)))
    info = Dict{Any, FullInteriorUpwindStencilInfo}()
    eqvar_ivs = ivs(eqvar, s)
    gl_vec = [length(s, x) for x in eqvar_ivs]

    for (key, usi) in upwind_cache
        u, x, d = key
        dim = indexmap[x]
        gl = gl_vec[dim]
        N = hi_vec[dim] - lo_vec[dim] + 1

        # Process both neg and pos directions
        neg_wmat, neg_omat, padded_neg = _build_upwind_full_matrices(
            usi.D_neg, N, lo_vec[dim], gl, usi.neg_offsets, usi.is_uniform;
            dim_periodic=is_periodic[dim]
        )
        pos_wmat, pos_omat, padded_pos = _build_upwind_full_matrices(
            usi.D_pos, N, lo_vec[dim], gl, usi.pos_offsets, usi.is_uniform;
            dim_periodic=is_periodic[dim]
        )

        info[key] = FullInteriorUpwindStencilInfo(
            neg_wmat, neg_omat, padded_neg,
            pos_wmat, pos_omat, padded_pos
        )
    end
    return info
end

"""
    _build_upwind_full_matrices(D_op, N, lo, gl, interior_offsets, is_uniform;
                                 dim_periodic=false)

Build weight and offset matrices for a single upwind direction operator.
For periodic uniform dimensions, always use the interior stencil.
"""
function _build_upwind_full_matrices(D_op, N, lo, gl, interior_offsets, is_uniform;
                                      dim_periodic=false)
    bpc = D_op.boundary_point_count
    offside = D_op.offside
    sl = D_op.stencil_length
    bsl = D_op.boundary_stencil_length
    padded = max(sl, bsl)

    wmat = zeros(Float64, padded, N)
    omat = zeros(Int, padded, N)

    for k in 1:N
        g = lo + k - 1  # absolute grid index

        if dim_periodic
            # Periodic uniform: always use interior stencil (wrapping handled symbolically)
            weights = Vector{Float64}(D_op.stencil_coefs)
            offsets = collect(interior_offsets)
        elseif g <= offside
            # Lower frame: use low_boundary_coefs[g]
            weights = Vector{Float64}(D_op.low_boundary_coefs[g])
            # Taps at grid indices 1:bsl
            offsets = collect((1 - g):(bsl - g))
        elseif g > gl - bpc
            # Upper frame: use high_boundary_coefs[gl - g + 1]
            weights = Vector{Float64}(D_op.high_boundary_coefs[gl - g + 1])
            # Taps at grid indices (gl-bsl+1):gl
            offsets = collect((gl - bsl + 1 - g):(gl - g))
        else
            # Interior
            if is_uniform
                weights = Vector{Float64}(D_op.stencil_coefs)
            else
                # Non-uniform: stencil_coefs indexed by interior position
                weights = Vector{Float64}(D_op.stencil_coefs[g - offside])
            end
            offsets = collect(interior_offsets)
        end

        nw = length(weights)
        for j in 1:nw
            wmat[j, k] = weights[j]
            omat[j, k] = offsets[j]
        end
    end

    return wmat, omat, padded
end

# --- Periodic integer wrapping helpers for precompute-time ------------------

"""
    _wrap_grid_periodic(g, gl)

Wrap absolute grid index `g` into the periodic range `[2, gl]`.
Index 1 is the duplicate boundary point (same as index gl), so valid
interior indices are 2:gl.  The mapping is: `mod(g - 2, gl - 1) + 2`.
"""
_wrap_grid_periodic(g, gl) = mod(g - 2, gl - 1) + 2

"""
    _wrap_half_periodic(h, N_half)

Wrap half-point index `h` into the periodic range `[1, N_half]`.
"""
_wrap_half_periodic(h, N_half) = mod1(h, N_half)

# --- Full-interior nonlinlap data structures --------------------------------

"""
    FullNonlinlapInfo

Pre-expanded 3D weight+tap matrices for full-interior nonlinear Laplacian.
Uses single-level Const indexing: `Const(matrix_3d)[j_outer, j_inner, _i]`.

The outer derivative accesses half-points, and at each half-point the
inner/interp stencils access grid values.  Near boundaries, all three
operators may use boundary stencils.
"""
struct FullNonlinlapInfo
    # Outer derivative: 2D matrices indexed by (j_outer, _i)
    outer_weight_matrix::Matrix{Float64}   # padded_outer × N_full
    padded_outer::Int

    # Interpolation: 3D matrices indexed by (j_outer, j_interp, _i)
    interp_weight_3d::Array{Float64, 3}    # padded_outer × padded_interp × N_full
    interp_tap_3d::Array{Int, 3}           # padded_outer × padded_interp × N_full
    padded_interp::Int

    # Inner derivative: 3D matrices indexed by (j_outer, j_inner, _i)
    inner_weight_3d::Array{Float64, 3}     # padded_outer × padded_inner × N_full
    inner_tap_3d::Array{Int, 3}            # padded_outer × padded_inner × N_full
    padded_inner::Int
end

"""
    precompute_full_nonlinlap(s, depvars, derivweights, nonlinlap_cache,
                               lo_vec, hi_vec, indexmap, eqvar;
                               is_periodic=falses(length(lo_vec)))

Build `FullNonlinlapInfo` for every `(u, x)` key in `nonlinlap_cache`.
The matrices cover grid indices `lo_vec[dim]..hi_vec[dim]` (the full interior).

Half-point `h` (1-indexed) lies between grid points `h` and `h+1`.
There are `N_grid - 1` half-points total.

For periodic uniform dimensions, the helper functions skip boundary branches
and the returned tap indices are wrapped at precompute time using integer
modular arithmetic.
"""
function precompute_full_nonlinlap(s, depvars, derivweights, nonlinlap_cache,
                                    lo_vec, hi_vec, indexmap, eqvar;
                                    is_periodic=falses(length(lo_vec)))
    info = Dict{Any, FullNonlinlapInfo}()
    eqvar_ivs = ivs(eqvar, s)
    gl_vec = [length(s, x) for x in eqvar_ivs]

    for ((u, x), nsi) in nonlinlap_cache
        dim = indexmap[x]
        gl = gl_vec[dim]
        dim_periodic = is_periodic[dim]
        N = hi_vec[dim] - lo_vec[dim] + 1   # number of full-interior grid points

        D_inner = derivweights.halfoffsetmap[1][Differential(x)]
        D_outer = derivweights.halfoffsetmap[2][Differential(x)]
        interp  = derivweights.interpmap[x]

        N_half = gl - 1  # total number of half-points

        # Padded stencil lengths (max of interior and boundary)
        padded_outer  = max(D_outer.stencil_length, D_outer.boundary_stencil_length)
        padded_inner  = max(D_inner.stencil_length, D_inner.boundary_stencil_length)
        padded_interp = max(interp.stencil_length, interp.boundary_stencil_length)

        # For periodic: only interior stencils used, so padded = stencil_length
        if dim_periodic
            padded_outer  = D_outer.stencil_length
            padded_inner  = D_inner.stencil_length
            padded_interp = interp.stencil_length
        end

        # Allocate matrices
        outer_wmat = zeros(Float64, padded_outer, N)
        interp_w3d = zeros(Float64, padded_outer, padded_interp, N)
        interp_t3d = ones(Int, padded_outer, padded_interp, N)  # ones = safe default (index 1)
        inner_w3d  = zeros(Float64, padded_outer, padded_inner, N)
        inner_t3d  = ones(Int, padded_outer, padded_inner, N)

        bpc_outer  = D_outer.boundary_point_count
        bpc_inner  = D_inner.boundary_point_count
        bpc_interp = interp.boundary_point_count

        is_uniform = nsi.is_uniform

        for k in 1:N
            g = lo_vec[dim] + k - 1  # absolute grid index (1-indexed)

            # --- Outer operator at grid point g ---
            outer_weights, outer_half_points = _half_op_weights_and_taps(
                D_outer, g, gl, N_half, bpc_outer, nsi.outer_offsets, is_uniform;
                dim_periodic=dim_periodic
            )
            nw_outer = length(outer_weights)
            for j in 1:nw_outer
                outer_wmat[j, k] = outer_weights[j]
            end

            # Wrap outer half-points for periodic
            if dim_periodic
                outer_half_points = [_wrap_half_periodic(h, N_half) for h in outer_half_points]
            end

            # --- For each outer tap (half-point), compute inner and interp ---
            for j_outer in 1:nw_outer
                h = outer_half_points[j_outer]  # absolute half-point index (1-indexed)

                # Inner derivative at half-point h
                inner_weights, inner_taps = _half_inner_weights_and_taps(
                    D_inner, h, gl, N_half, bpc_inner, nsi.inner_offsets, is_uniform;
                    dim_periodic=dim_periodic
                )
                nw_inner = length(inner_weights)
                for j_inner in 1:nw_inner
                    inner_w3d[j_outer, j_inner, k] = inner_weights[j_inner]
                    tap = dim_periodic ? _wrap_grid_periodic(inner_taps[j_inner], gl) : inner_taps[j_inner]
                    inner_t3d[j_outer, j_inner, k] = tap
                end

                # Interpolation at half-point h
                interp_weights, interp_taps = _half_inner_weights_and_taps(
                    interp, h, gl, N_half, bpc_interp, nsi.interp_offsets, is_uniform;
                    dim_periodic=dim_periodic
                )
                nw_interp = length(interp_weights)
                for j_interp in 1:nw_interp
                    interp_w3d[j_outer, j_interp, k] = interp_weights[j_interp]
                    tap = dim_periodic ? _wrap_grid_periodic(interp_taps[j_interp], gl) : interp_taps[j_interp]
                    interp_t3d[j_outer, j_interp, k] = tap
                end
            end

            # For padded outer taps (j > nw_outer), the outer weight is zero
            # so their contribution is zero.  However, expr_sym may contain
            # negative powers of dependent variables (e.g. u^(-1)), which causes
            # 0/0 when both inner and interp weights are zero.  Avoid this by
            # setting the first interp weight to 1.0 for padded taps, making
            # the interpolation non-zero.  The product is still zero because
            # outer_weight = 0.
            for j_pad in (nw_outer + 1):padded_outer
                interp_w3d[j_pad, 1, k] = 1.0
            end
        end

        info[(u, x)] = FullNonlinlapInfo(
            outer_wmat, padded_outer,
            interp_w3d, interp_t3d, padded_interp,
            inner_w3d, inner_t3d, padded_inner
        )
    end
    return info
end

"""
    _half_op_weights_and_taps(D_op, g, gl, N_half, bpc, interior_offsets, is_uniform;
                               dim_periodic=false)

Compute outer operator weights and half-point tap positions at grid point `g`.
The outer operator maps grid points to half-points.

Returns `(weights, half_points)` where `half_points` are absolute 1-indexed
half-point positions.

For periodic uniform dimensions, always uses the interior stencil.
"""
function _half_op_weights_and_taps(D_op, g, gl, N_half, bpc, interior_offsets, is_uniform;
                                    dim_periodic=false)
    sl = D_op.stencil_length
    bsl = D_op.boundary_stencil_length

    # The outer operator is defined on half-points.
    # At grid point g, the interior stencil accesses half-points at
    # g + offset - 1 for each offset in interior_offsets.
    #
    # For non-uniform grids, the outer operator may be constructed on the
    # midpoint grid (length gl-1) rather than the full grid (length gl).
    # We compute the effective number of output positions from the operator
    # itself and map g to the operator's position space accordingly.

    if is_uniform
        # Uniform: stencil_coefs is a single SVector, no indexing needed
        if dim_periodic
            # Periodic: always use interior stencil (wrapping handled by caller)
            weights = Vector{Float64}(D_op.stencil_coefs)
            half_points = [g + off - 1 for off in interior_offsets]
        elseif g <= bpc
            weights = Vector{Float64}(D_op.low_boundary_coefs[g])
            half_points = collect(1:bsl)
        elseif g > gl - bpc
            weights = Vector{Float64}(D_op.high_boundary_coefs[gl - g + 1])
            half_points = collect((N_half - bsl + 1):N_half)
        else
            weights = Vector{Float64}(D_op.stencil_coefs)
            half_points = [g + off - 1 for off in interior_offsets]
        end
    else
        # Non-uniform: stencil_coefs is a Vector of SVectors.
        # Compute effective number of output positions from the operator.
        n_interior = length(D_op.stencil_coefs)
        N_eff = n_interior + 2 * bpc

        # Map grid point g to operator position p.
        # The operator covers N_eff positions; grid covers gl positions.
        # For outer operator (constructed on midpoint grid): N_eff = gl - 1, p = g - 1
        # For inner/interp (constructed on full grid): N_eff = gl, p = g
        p = g - (gl - N_eff)

        if p <= bpc
            weights = Vector{Float64}(D_op.low_boundary_coefs[p])
            half_points = collect(1:bsl)
        elseif p > N_eff - bpc
            weights = Vector{Float64}(D_op.high_boundary_coefs[N_eff - p + 1])
            half_points = collect((N_half - bsl + 1):N_half)
        else
            weights = Vector{Float64}(D_op.stencil_coefs[p - bpc])
            half_points = [g + off - 1 for off in interior_offsets]
        end
    end

    return weights, half_points
end

"""
    _half_inner_weights_and_taps(D_op, h, gl, N_half, bpc, interior_offsets, is_uniform;
                                  dim_periodic=false)

Compute inner/interp operator weights and grid-point tap positions at
half-point `h` (1-indexed, total `N_half` half-points).

The inner/interp operators are defined at half-points and access grid points.
At interior half-point `h`, the stencil accesses grid points at
`h + offset - 1 + 1 = h + offset` for the standard centered offsets
(the +1 accounts for the half-point lying between grid points h and h+1,
and the stencil_coefs being computed at position 0.5 relative to the stencil).

Returns `(weights, grid_taps)` where `grid_taps` are absolute 1-indexed
grid point positions.

For periodic uniform dimensions, always uses the interior stencil.
"""
function _half_inner_weights_and_taps(D_op, h, gl, N_half, bpc, interior_offsets, is_uniform;
                                       dim_periodic=false)
    sl = D_op.stencil_length
    bsl = D_op.boundary_stencil_length

    if dim_periodic
        # Periodic: always use interior stencil (tap wrapping handled by caller)
        weights = Vector{Float64}(D_op.stencil_coefs)
        grid_taps = [h + off for off in interior_offsets]
    elseif h <= bpc
        # Lower boundary: use low_boundary_coefs[h]
        weights = Vector{Float64}(D_op.low_boundary_coefs[h])
        # Boundary stencil taps at grid points 1:bsl
        grid_taps = collect(1:bsl)
    elseif h > N_half - bpc
        # Upper boundary: use high_boundary_coefs[N_half - h + 1]
        weights = Vector{Float64}(D_op.high_boundary_coefs[N_half - h + 1])
        # Boundary stencil taps at grid points (gl-bsl+1):gl
        grid_taps = collect((gl - bsl + 1):gl)
    else
        # Interior
        if is_uniform
            weights = Vector{Float64}(D_op.stencil_coefs)
        else
            # Non-uniform: stencil_coefs indexed by interior position
            weights = Vector{Float64}(D_op.stencil_coefs[h - bpc])
        end
        # Interior grid taps: h + offset for each offset in interior_offsets
        # (where offset is centered around 0, e.g. [0, 1] for 2-point stencil)
        grid_taps = [h + off for off in interior_offsets]
    end

    return weights, grid_taps
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
- Staggered grid (odd-order) derivatives on uniform grids
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
    upwind_cache = precompute_upwind_stencils(s, depvars, derivweights)
    nonlinlap_cache = precompute_nonlinlap_stencils(s, depvars, derivweights)
    weno_cache = precompute_weno_stencils(s, depvars, derivweights)
    staggered_cache = precompute_staggered_stencils(s, depvars, derivweights)

    # -- determine whether the ArrayOp path can handle this PDE ---------------
    has_odd_orders = any(
        any(isodd(d) for d in derivweights.orders[x])
        for u in depvars for x in ivs(u, s)
    )
    can_upwind = has_odd_orders && derivweights.advection_scheme isa UpwindScheme
    is_staggered = !isempty(staggered_cache)

    # Detect spherical Laplacian terms first (they take priority over nonlinlap).
    spherical_terms_info = _detect_spherical_terms(pde, s, depvars)
    has_spherical = !isempty(spherical_terms_info) && !isempty(nonlinlap_cache)

    # Precompute stencils, including D1 entries for spherical variables so that
    # precompute_full_interior_stencils can build full-interior D1 info.
    sph_vars = if has_spherical
        unique([(info.u, info.r) for info in values(spherical_terms_info)])
    else
        nothing
    end
    stencil_cache = precompute_stencils(s, depvars, derivweights; spherical_vars=sph_vars)

    # Detect nonlinear Laplacian terms -- their odd-order derivatives are
    # handled internally, so they don't block the template path.
    # Exclude spherical-matched terms to prevent double-matching.
    nonlinlap_terms = _detect_nonlinlap_terms(pde, s, depvars, spherical_terms_info)
    has_nonlinlap = !isempty(nonlinlap_terms) && !isempty(nonlinlap_cache)

    has_weno = !isempty(weno_cache)
    can_template = !has_odd_orders || can_upwind || has_nonlinlap || has_spherical || has_weno || is_staggered

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
                elseif isodd(d) && haskey(staggered_cache, (u, x, d))
                    ssi = staggered_cache[(u, x, d)]
                    max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], ssi.bpc)
                    max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], ssi.bpc)
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

    # -- Determine whether full-interior mode is possible ----------------------
    # Full-interior mode eliminates the boundary-proximity frame by using
    # position-dependent weight+offset matrices.  Supported for centered,
    # upwind, nonlinlap, and spherical Laplacian derivatives on uniform and
    # non-uniform grids.  WENO keeps the existing centered-ArrayOp +
    # frame-per-point behavior.  Periodic dimensions on uniform grids are
    # supported (interior stencil everywhere, indices wrapped symbolically);
    # periodic on non-uniform grids falls back to the standard path.
    # Staggered grids use the standard path (boundary frame + interior ArrayOp)
    # because their alignment-dependent boundary stencils are already handled
    # correctly by the per-point scalar fallback.
    periodic_dim_uniform = map(eqvar_ivs) do x
        s.dxs[x] isa Number
    end
    has_periodic_nonuniform = any(d -> is_periodic[d] && !periodic_dim_uniform[d], 1:ndim)
    all_full_interior = !has_weno && !has_periodic_nonuniform && !is_staggered

    if all_full_interior
        # Full-interior mode: ArrayOp covers lo_vec..hi_vec (entire interior)
        fi_centered = precompute_full_interior_stencils(
            s, depvars, derivweights, stencil_cache,
            lo_vec, hi_vec, indexmap, eqvar;
            is_periodic=is_periodic
        )
        fi_upwind = if !isempty(upwind_cache)
            precompute_full_interior_upwind(
                s, depvars, derivweights, upwind_cache,
                lo_vec, hi_vec, indexmap, eqvar;
                is_periodic=is_periodic
            )
        else
            nothing
        end
        fi_nonlinlap = if has_nonlinlap || has_spherical
            precompute_full_nonlinlap(
                s, depvars, derivweights, nonlinlap_cache,
                lo_vec, hi_vec, indexmap, eqvar;
                is_periodic=is_periodic
            )
        else
            nothing
        end

        n_region = [hi_vec[d] - lo_vec[d] + 1 for d in 1:ndim]
        eqs_boundary = Equation[]  # No frame loop needed

        eqs_centered = if all(n_region .> 0)
            result = _build_interior_arrayop(
                n_region, lo_vec, s, depvars, pde, derivweights,
                stencil_cache, upwind_cache, nonlinlap_cache,
                spherical_terms_info, weno_cache, bcmap, eqvar, indexmap,
                is_periodic, gl_vec;
                full_interior_centered_cache=fi_centered,
                full_interior_upwind_cache=fi_upwind,
                full_nonlinlap_cache=fi_nonlinlap,
                staggered_cache=staggered_cache
            )
            candidate, eq_first = result
            # Validate at the first point (which is a frame point in this mode).
            # Skip validation for periodic: the scalar path uses IfElse-based
            # wrapping which differs structurally from full-interior's Const-matrix
            # approach, so _equations_match would fail even though numerics match.
            has_any_periodic = any(is_periodic)
            if !has_any_periodic
                II_check = CartesianIndex(Tuple(lo_vec))
                eq_scalar = discretize_equation_at_point(
                    II_check, s, depvars, pde, derivweights, bcmap,
                    eqvar, indexmap, boundaryvalfuncs
                )
            end
            if has_any_periodic || _equations_match(eq_first, eq_scalar)
                candidate
            else
                @debug "Full-interior ArrayOp validation failed" eq_first eq_scalar
                # Fall back to standard centered + frame path
                collect(vec(map(CartesianIndices(Tuple(lo_vec[d]:hi_vec[d] for d in 1:ndim))) do II
                    discretize_equation_at_point(
                        II, s, depvars, pde, derivweights, bcmap,
                        eqvar, indexmap, boundaryvalfuncs
                    )
                end))
            end
        else
            Equation[]
        end
    else
        # Standard path: centered-only ArrayOp + frame per-point
        # For periodic dimensions: no boundary frame needed (stencil wraps around),
        # so the ArrayOp covers the full grid.
        lo_centered = [is_periodic[d] ? lo_vec[d] : max(lo_vec[d], max_lower_bpc[d] + 1) for d in 1:ndim]
        hi_centered = [is_periodic[d] ? hi_vec[d] : min(hi_vec[d], gl_vec[d] - max_upper_bpc[d]) for d in 1:ndim]
        n_centered  = [max(0, hi_centered[d] - lo_centered[d] + 1) for d in 1:ndim]

        # -- per-point equations for boundary-proximity interior points -----------
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
            result = _build_interior_arrayop(
                n_centered, lo_centered, s, depvars, pde, derivweights,
                stencil_cache, upwind_cache, nonlinlap_cache,
                spherical_terms_info, weno_cache, bcmap, eqvar, indexmap,
                is_periodic, gl_vec;
                staggered_cache=staggered_cache
            )
            candidate, eq_first = result
            begin
                # Validate: compare the first instantiated equation against the
                # scalar path for the same point.
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
            end
        else
            Equation[]
        end
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
                             is_periodic, gl_vec;
                             full_interior_centered_cache, full_interior_upwind_cache,
                             staggered_cache)

Build a single ArrayOp equation for the interior region.

Handles centred (even-order), upwind (odd-order), staggered (odd-order),
WENO (1st-order), mixed cross-derivative, nonlinear Laplacian, and
spherical Laplacian stencils using symbolic index variables.

For periodic dimensions, stencil indices are wrapped using `_wrap_periodic_idx`
so the ArrayOp covers the full grid without a boundary frame.

When `full_interior_centered_cache` and/or `full_interior_upwind_cache` are
provided, uses position-dependent weight+offset matrices to cover ALL interior
points (including boundary-proximity frame points) in a single ArrayOp.

Returns `(eqs, eq_first)` where `eqs` is a single-element vector containing
the ArrayOp equation, and `eq_first` is the scalar equation at the first
centred point (for validation against the scalar path).
"""
function _build_interior_arrayop(
        n_centered, lo_centered, s, depvars, pde, derivweights,
        stencil_cache, upwind_cache, nonlinlap_cache,
        spherical_terms_info, weno_cache, bcmap, eqvar, indexmap,
        is_periodic=falses(length(n_centered)),
        gl_vec=zeros(Int, length(n_centered));
        full_interior_centered_cache=nothing,
        full_interior_upwind_cache=nothing,
        full_nonlinlap_cache=nothing,
        staggered_cache=nothing
    )
    ndim = length(n_centered)
    _idxs_arr = SymbolicUtils.idxs_for_arrayop(SymbolicUtils.SymReal)
    _idxs = [_idxs_arr[d] for d in 1:ndim]
    bases = [lo_centered[d] - 1 for d in 1:ndim]

    # -- FD rules for centred (even-order) and staggered (odd-order) derivatives
    fd_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        for (_, x) in enumerate(u_spatial)
            for d in derivweights.orders[x]
                # Staggered odd-order derivatives: fixed offset per alignment
                if !iseven(d)
                    if staggered_cache !== nothing && haskey(staggered_cache, (u, x, d))
                        ssi = staggered_cache[(u, x, d)]
                        taps = map(ssi.interior_offsets) do off
                            idx_exprs = map(u_spatial) do xv
                                eq_d = indexmap[xv]
                                raw_idx = _idxs[eq_d] + bases[eq_d]
                                raw_idx = isequal(xv, x) ? raw_idx + off : raw_idx
                                _maybe_wrap(raw_idx, eq_d, is_periodic, gl_vec)
                            end
                            Symbolics.wrap(u_c[idx_exprs...])
                        end
                        expr = sym_dot(ssi.D_wind.stencil_coefs, taps)
                        push!(fd_rules, (Differential(x)^d)(u) => expr)
                    end
                    continue
                end
                si = stencil_cache[(u, x, d)]

                if full_interior_centered_cache !== nothing && haskey(full_interior_centered_cache, (u, x, d))
                    # Full-interior mode: position-dependent weight+offset matrices
                    fisi = full_interior_centered_cache[(u, x, d)]
                    wm_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(fisi.weight_matrix)
                    om_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(fisi.offset_matrix)
                    dim = indexmap[x]
                    expr = sum(1:fisi.padded_len) do j
                        w = Symbolics.wrap(wm_c[j, _idxs[dim]])
                        off_val = Symbolics.wrap(om_c[j, _idxs[dim]])
                        idx_exprs = map(u_spatial) do xv
                            eq_d = indexmap[xv]
                            raw_idx = _idxs[eq_d] + bases[eq_d]
                            combined = isequal(xv, x) ? raw_idx + off_val : raw_idx
                            _maybe_wrap(combined, eq_d, is_periodic, gl_vec)
                        end
                        w * Symbolics.wrap(u_c[idx_exprs...])
                    end
                else
                    # Standard centered-only mode
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
                        dim = indexmap[x]
                        bpc = si.D_op.boundary_point_count
                        if is_periodic[dim]
                            # Periodic non-uniform: extended N-column weight matrix
                            ext_wmat = _build_periodic_centered_wmat(si.D_op, collect(s.grid[x]))
                            wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(ext_wmat)
                            point_idx = _idxs[dim] + bases[dim]
                        else
                            wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(si.weight_matrix)
                            point_idx = _idxs[dim] + bases[dim] - bpc
                        end
                        expr = sum(1:length(si.offsets)) do k
                            Symbolics.wrap(wmat_c[k, point_idx]) * taps[k]
                        end
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
    # Staggered grids use alignment-based offsets instead of upwind switching,
    # matching the scalar path which sets advection_rules = [] for StaggeredGrid.
    upwind_rules = Pair[]
    if !isempty(upwind_cache) && (staggered_cache === nothing || isempty(staggered_cache))
        upwind_rules = _build_upwind_rules(
            pde, s, depvars, derivweights, upwind_cache,
            bcmap, indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec;
            full_interior_upwind_cache=full_interior_upwind_cache
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
            is_periodic, gl_vec;
            full_nonlinlap_cache=full_nonlinlap_cache
        )
    end

    # -- Spherical Laplacian rules -------------------------------------------
    sph_rules = Pair[]
    if !isempty(spherical_terms_info) && !isempty(nonlinlap_cache)
        sph_rules = _build_spherical_rules(
            pde, s, depvars, derivweights, nonlinlap_cache,
            spherical_terms_info, indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec;
            full_nonlinlap_cache=full_nonlinlap_cache,
            full_interior_centered_cache=full_interior_centered_cache
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

    # -- Detect algebraic equations (no time derivative of eqvar) ---------------
    # After PDE rearrangement, equations have the form `lhs ~ 0`.  For ODEs,
    # `lhs` contains `Dt(eqvar(...))`.  For algebraic equations it does not.
    function _contains_time_diff(expr_raw, time)
        SymbolicUtils.iscall(expr_raw) || return false
        op = SymbolicUtils.operation(expr_raw)
        if op isa Differential && isequal(op.x, time)
            return true
        end
        return any(a -> _contains_time_diff(a, time), SymbolicUtils.arguments(expr_raw))
    end
    # Check both sides — the time derivative could be on either side of the
    # rearranged equation (though typically lhs after rearrangement).
    is_algebraic = !_contains_time_diff(Symbolics.unwrap(pde.lhs), s.time) &&
                   !_contains_time_diff(Symbolics.unwrap(pde.rhs), s.time)

    # -- Separate time derivative from spatial terms --------------------------
    eqvar_raw = Symbolics.unwrap(s.discvars[eqvar])
    eqvar_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(eqvar_raw)
    eqvar_idx_exprs = [_idxs[d] + bases[d] for d in 1:ndim]

    if is_algebraic
        # Algebraic equation: no Dt term.  Wrap both sides directly as ArrayOps.
        ao_ranges = Dict(_idxs[d] => (1:1:n_centered[d]) for d in 1:ndim)

        lhs_raw = Symbolics.unwrap(template_lhs)
        rhs_raw = Symbolics.unwrap(template_rhs)
        # Handle numeric RHS (e.g., literal 0 after rearrangement)
        if !(lhs_raw isa SymbolicUtils.BasicSymbolic)
            lhs_raw = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(lhs_raw)
        end
        if !(rhs_raw isa SymbolicUtils.BasicSymbolic)
            rhs_raw = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(rhs_raw)
        end

        lhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
            _idxs, lhs_raw, +, nothing, ao_ranges
        )
        rhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
            _idxs, rhs_raw, +, nothing, ao_ranges
        )
        arrayop_eq = Symbolics.wrap(lhs_ao) ~ Symbolics.wrap(rhs_ao)

        # Validation equation at first point
        sub_first = Dict(_idxs[d] => 1 for d in 1:ndim)
        lhs_first = pde_substitute(template_lhs, sub_first)
        rhs_first = pde_substitute(template_rhs, sub_first)
        eq_first = lhs_first ~ rhs_first

        return [arrayop_eq], eq_first
    else
        # ODE equation: contains Dt(eqvar).  Isolate the spatial RHS.
        dt_template = Differential(s.time)(Symbolics.wrap(eqvar_c[eqvar_idx_exprs...]))

        # Try the standard formula first (works when Dt coefficient is +1,
        # which is the common case: `Dt(u) - spatial ~ 0`).
        spatial_rhs_candidate = dt_template - template_lhs + template_rhs

        if !_contains_time_diff(Symbolics.unwrap(spatial_rhs_candidate), s.time)
            # Standard case: Dt coefficient was +1, cleanly cancelled.
            spatial_rhs = spatial_rhs_candidate
        else
            # Non-standard Dt coefficient (e.g. `v - Dt(u) ~ 0` where
            # coefficient is -1).  Use pde_substitute to extract it.
            dt_key = Symbolics.unwrap(dt_template)
            f = pde_substitute(template_lhs, Dict(dt_key => 0))
            c_plus_f = pde_substitute(template_lhs, Dict(dt_key => 1))
            c = c_plus_f - f
            spatial_rhs = (template_rhs - f) / c
        end
    end

    # -- Wrap in ArrayOps -----------------------------------------------------
    ao_ranges = Dict(_idxs[d] => (1:1:n_centered[d]) for d in 1:ndim)

    rhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _idxs, Symbolics.unwrap(spatial_rhs), +, nothing, ao_ranges
    )

    # ODE: Dt(u_ao) ~ rhs_ao  (algebraic case already returned above)
    u_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _idxs, eqvar_c[eqvar_idx_exprs...], +, nothing, ao_ranges
    )
    lhs_wrapped = Differential(s.time)(Symbolics.wrap(u_ao))
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
        gl_vec=zeros(Int, length(bases));
        full_interior_upwind_cache=nothing
    )
    # Helper: build stencil expression for a given variable, dimension, offsets, weights.
    # For non-uniform grids, weight_matrix is a stencil_length × num_interior Matrix
    # and bpc is the offside (= low_boundary_point_count) used to align weight matrix
    # column indexing: stencil_coefs[j] corresponds to grid index (j + bpc).
    function _upwind_stencil_expr(u, x, offsets, weights, _idxs, bases, indexmap, s;
                                   weight_matrix=nothing, bpc=0, D_op=nothing)
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
            dim = indexmap[x]
            if is_periodic[dim]
                # Periodic non-uniform: extended N-column weight matrix
                ext_wmat = _build_periodic_upwind_wmat(D_op, collect(s.grid[x]), offsets)
                wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(ext_wmat)
                point_idx = _idxs[dim] + bases[dim]
            else
                wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(weight_matrix)
                point_idx = _idxs[dim] + bases[dim] - bpc
            end
            return sum(1:length(offsets)) do k
                Symbolics.wrap(wmat_c[k, point_idx]) * taps[k]
            end
        end
    end

    # Helper: build full-interior stencil expression using weight+offset matrices.
    function _upwind_full_interior_expr(u, x, wmat, omat, padded_len,
                                         _idxs, bases, indexmap, s,
                                         is_periodic, gl_vec)
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
        u_spatial = ivs(u, s)
        dim = indexmap[x]
        wm_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(wmat)
        om_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(omat)
        return sum(1:padded_len) do j
            w = Symbolics.wrap(wm_c[j, _idxs[dim]])
            off_val = Symbolics.wrap(om_c[j, _idxs[dim]])
            idx_exprs = map(u_spatial) do xv
                eq_d = indexmap[xv]
                raw_idx = _idxs[eq_d] + bases[eq_d]
                combined = isequal(xv, x) ? raw_idx + off_val : raw_idx
                _maybe_wrap(combined, eq_d, is_periodic, gl_vec)
            end
            w * Symbolics.wrap(u_c[idx_exprs...])
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

                if full_interior_upwind_cache !== nothing && haskey(full_interior_upwind_cache, (u, x, d))
                    fiusi = full_interior_upwind_cache[(u, x, d)]
                    neg_expr = _upwind_full_interior_expr(
                        u, x, fiusi.neg_weight_matrix, fiusi.neg_offset_matrix,
                        fiusi.padded_neg, _idxs, bases, indexmap, s,
                        is_periodic, gl_vec
                    )
                    pos_expr = _upwind_full_interior_expr(
                        u, x, fiusi.pos_weight_matrix, fiusi.pos_offset_matrix,
                        fiusi.padded_pos, _idxs, bases, indexmap, s,
                        is_periodic, gl_vec
                    )
                else
                    neg_expr = _upwind_stencil_expr(
                        u, x, usi.neg_offsets, usi.D_neg.stencil_coefs,
                        _idxs, bases, indexmap, s;
                        weight_matrix=usi.neg_weight_matrix,
                        bpc=usi.D_neg.offside,
                        D_op=usi.D_neg
                    )
                    pos_expr = _upwind_stencil_expr(
                        u, x, usi.pos_offsets, usi.D_pos.stencil_coefs,
                        _idxs, bases, indexmap, s;
                        weight_matrix=usi.pos_weight_matrix,
                        bpc=usi.D_pos.offside,
                        D_op=usi.D_pos
                    )
                end

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
                if full_interior_upwind_cache !== nothing && haskey(full_interior_upwind_cache, (u, x, d))
                    fiusi = full_interior_upwind_cache[(u, x, d)]
                    pos_expr = _upwind_full_interior_expr(
                        u, x, fiusi.pos_weight_matrix, fiusi.pos_offset_matrix,
                        fiusi.padded_pos, _idxs, bases, indexmap, s,
                        is_periodic, gl_vec
                    )
                else
                    pos_expr = _upwind_stencil_expr(
                        u, x, usi.pos_offsets, usi.D_pos.stencil_coefs,
                        _idxs, bases, indexmap, s;
                        weight_matrix=usi.pos_weight_matrix,
                        bpc=usi.D_pos.offside,
                        D_op=usi.D_pos
                    )
                end
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
            dim_x_local = indexmap[x]
            if x_is_uniform
                x_weights = Dx_op.stencil_coefs
            else
                x_bpc = Dx_op.boundary_point_count
                if is_periodic[dim_x_local]
                    x_wmat = _build_periodic_centered_wmat(Dx_op, collect(s.grid[x]))
                else
                    x_wmat = hcat(Vector.(Dx_op.stencil_coefs)...)
                end
                x_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(x_wmat)
            end

            for y in u_spatial
                isequal(x, y) && continue
                haskey(derivweights.map, Differential(y)) || continue
                Dy_op = derivweights.map[Differential(y)]
                y_is_uniform = Dy_op.dx isa Number
                y_offsets = collect(half_range(Dy_op.stencil_length))

                dim_y_local = indexmap[y]
                if y_is_uniform
                    y_weights = Dy_op.stencil_coefs
                else
                    y_bpc = Dy_op.boundary_point_count
                    if is_periodic[dim_y_local]
                        y_wmat = _build_periodic_centered_wmat(Dy_op, collect(s.grid[y]))
                    else
                        y_wmat = hcat(Vector.(Dy_op.stencil_coefs)...)
                    end
                    y_wmat_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(y_wmat)
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
                            x_pt = is_periodic[dim_x] ? _idxs[dim_x] + bases[dim_x] : _idxs[dim_x] + bases[dim_x] - x_bpc
                            Symbolics.wrap(x_wmat_c[kx, x_pt])
                        end

                        wy = if y_is_uniform
                            y_weights[ky]
                        else
                            y_pt = is_periodic[dim_y] ? _idxs[dim_y] + bases[dim_y] : _idxs[dim_y] + bases[dim_y] - y_bpc
                            Symbolics.wrap(y_wmat_c[ky, y_pt])
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
    _nonlinlap_full_template(expr_sym, u, x, nsi, fi_nlap, s, depvars, indexmap,
                              _idxs, bases_full)

Full-interior version of `_nonlinlap_template`.  Uses pre-expanded 3D
weight+tap matrices from `fi_nlap` (a `FullNonlinlapInfo`) so that a single
ArrayOp covers ALL interior points including boundary-proximity ones.

Tap positions are absolute (from `interp_tap_3d` and `inner_tap_3d`).
For periodic dimensions, tap indices are pre-wrapped at precompute time
(see `precompute_full_nonlinlap`), so no symbolic `_maybe_wrap` is needed.
"""
function _nonlinlap_full_template(expr_sym, u, x, nsi, fi_nlap, s, depvars, indexmap,
                                   _idxs, bases_full)
    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(u_raw)
    u_spatial = ivs(u, s)
    dim = indexmap[x]

    wrap = Symbolics.wrap
    ConstSR = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}

    # Const-wrap the precomputed matrices
    outer_wm_c   = ConstSR(fi_nlap.outer_weight_matrix)
    interp_w3d_c = ConstSR(fi_nlap.interp_weight_3d)
    interp_t3d_c = ConstSR(fi_nlap.interp_tap_3d)
    inner_w3d_c  = ConstSR(fi_nlap.inner_weight_3d)
    inner_t3d_c  = ConstSR(fi_nlap.inner_tap_3d)

    # NOTE: All Const indexing must use raw SymReal indices (not Num/wrapped).
    # Only wrap() the final product expressions.

    inner_exprs = map(1:fi_nlap.padded_outer) do j_outer
        # --- Interpolation rules for variables at this half-point ---
        interp_var_rules = Pair[]
        for v in depvars
            v_raw = Symbolics.unwrap(s.discvars[v])
            v_c = ConstSR(v_raw)
            v_spatial = ivs(v, s)
            taps = map(1:fi_nlap.padded_interp) do j_interp
                # Tap index (absolute grid position) — raw SymReal
                tap_idx = interp_t3d_c[j_outer, j_interp, _idxs[dim]]
                # Interpolation weight — raw SymReal
                iw = interp_w3d_c[j_outer, j_interp, _idxs[dim]]
                idx_exprs = map(v_spatial) do xv
                    eq_d = indexmap[xv]
                    if isequal(xv, x)
                        tap_idx
                    else
                        _idxs[eq_d] + bases_full[eq_d]
                    end
                end
                wrap(iw) * wrap(v_c[idx_exprs...])
            end
            push!(interp_var_rules, v => sum(taps))
        end

        # --- Interpolation rules for grid coordinates ---
        interp_iv_rules = Pair[]
        for xv in s.x̄
            if isequal(xv, x)
                grid_c = ConstSR(collect(s.grid[x]))
                taps = map(1:fi_nlap.padded_interp) do j_interp
                    iw = interp_w3d_c[j_outer, j_interp, _idxs[dim]]
                    tap_idx = interp_t3d_c[j_outer, j_interp, _idxs[dim]]
                    # grid_c[tap_idx]: tap_idx is raw SymReal (Int-typed), so this works
                    wrap(iw) * wrap(grid_c[tap_idx])
                end
                push!(interp_iv_rules, x => sum(taps))
            else
                haskey(indexmap, xv) || continue
                grid_c = ConstSR(collect(s.grid[xv]))
                xv_dim = indexmap[xv]
                push!(interp_iv_rules, xv => wrap(grid_c[_idxs[xv_dim] + bases_full[xv_dim]]))
            end
        end

        # --- Inner derivative of u at this half-point: Dx(u) ---
        inner_deriv_taps = map(1:fi_nlap.padded_inner) do j_inner
            tap_idx = inner_t3d_c[j_outer, j_inner, _idxs[dim]]
            iw = inner_w3d_c[j_outer, j_inner, _idxs[dim]]
            idx_exprs = map(u_spatial) do xv
                eq_d = indexmap[xv]
                if isequal(xv, x)
                    tap_idx
                else
                    _idxs[eq_d] + bases_full[eq_d]
                end
            end
            wrap(iw) * wrap(u_c[idx_exprs...])
        end
        inner_deriv = sum(inner_deriv_taps)

        # --- Substitute all rules into expr * Dx(u) ---
        deriv_rules = Pair[Differential(x)(u) => inner_deriv]
        all_rules = Dict(vcat(deriv_rules, interp_var_rules, interp_iv_rules))
        substitute(expr_sym * Differential(x)(u), all_rules)
    end

    # Apply outer weights to get the full nonlinear Laplacian
    return sum(1:fi_nlap.padded_outer) do j_outer
        wrap(outer_wm_c[j_outer, _idxs[dim]]) * inner_exprs[j_outer]
    end
end

"""
    _build_nonlinlap_rules(pde, s, depvars, derivweights, nonlinlap_cache,
                            indexmap, _idxs, bases, var_rules;
                            full_nonlinlap_cache=nothing)

Build term-level substitution rules for nonlinear Laplacian patterns.

Uses the same pattern-matching approach as `generate_nonlinlap_rules` from the
scalar path, but substitutes ArrayOp-parameterized stencils.

When `full_nonlinlap_cache` is provided, uses `_nonlinlap_full_template`
instead of `_nonlinlap_template` to eliminate boundary-proximity frame equations.

Returns a vector of `Pair{term => discretized_expr}`.
"""
function _build_nonlinlap_rules(
        pde, s, depvars, derivweights, nonlinlap_cache,
        indexmap, _idxs, bases, var_rules,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases));
        full_nonlinlap_cache=nothing
    )
    terms = split_terms(pde, s.x̄)
    vr_dict = Dict(var_rules)
    nonlinlap_rules = Pair[]

    for u in depvars
        for x in ivs(u, s)
            haskey(nonlinlap_cache, (u, x)) || continue
            nsi = nonlinlap_cache[(u, x)]

            # Local dispatch: full-interior template vs standard template
            fi_nlap = (full_nonlinlap_cache !== nothing && haskey(full_nonlinlap_cache, (u, x))) ?
                      full_nonlinlap_cache[(u, x)] : nothing
            _nlap(expr_sym) = if fi_nlap !== nothing
                _nonlinlap_full_template(
                    expr_sym, u, x, nsi, fi_nlap, s, depvars, indexmap, _idxs, bases
                )
            else
                _nonlinlap_template(
                    expr_sym, u, x, nsi, s, depvars, indexmap, _idxs, bases,
                    is_periodic, gl_vec
                )
            end

            # Pattern 1: *(~~c, Dx(*(~~a, Dx(u), ~~b)), ~~d)
            rule_mul = @rule *(
                ~~c,
                $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)),
                ~~d
            ) => begin
                expr_sym = *(~a..., ~b...)
                outer_coeff = *(~c..., ~d...)
                outer_coeff_subst = pde_substitute(outer_coeff, vr_dict)
                nlap = _nlap(expr_sym)
                outer_coeff_subst * nlap
            end

            # Pattern 2: Dx(*(~~a, Dx(u), ~~b))
            rule_standalone = @rule $(Differential(x))(
                *(~~a, $(Differential(x))(u), ~~b)
            ) => begin
                expr_sym = *(~a..., ~b...)
                _nlap(expr_sym)
            end

            # Pattern 3: Dx(Dx(u) / ~a)
            rule_div = @rule $(Differential(x))(
                $(Differential(x))(u) / ~a
            ) => begin
                expr_sym = 1 / ~a
                _nlap(expr_sym)
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
                nlap = _nlap(expr_sym)
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
                nlap = _nlap(expr_sym)
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
                         indexmap, _idxs, bases, var_rules;
                         full_nonlinlap_cache=nothing,
                         full_interior_centered_cache=nothing)

Build the ArrayOp-indexed discretization of the spherical Laplacian
`r^{-2} * Dr(r^2 * innerexpr * Dr(u))`.

At r ≈ 0:   `6 * innerexpr * D2(u)`   (L'Hôpital's rule)
At r ≠ 0:   `innerexpr * (D1(u)/r + nonlinlap(innerexpr, u, r))`

Uses `IfElse.ifelse` for the r ≈ 0 conditional (same pattern as upwind wind
switching).

When `full_nonlinlap_cache` is provided, uses `_nonlinlap_full_template` for
the nonlinlap term.  When `full_interior_centered_cache` is provided, uses
position-dependent weight+offset matrices for the D1 derivative.
"""
function _spherical_template(info, nsi, s, depvars, derivweights,
                              indexmap, _idxs, bases, var_rules,
                              is_periodic=falses(length(bases)),
                              gl_vec=zeros(Int, length(bases));
                              full_nonlinlap_cache=nothing,
                              full_interior_centered_cache=nothing)
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
    # Check if we have a full-interior centered cache for the D1(r) derivative
    fi_d1 = if full_interior_centered_cache !== nothing
        d1_key = (u, r, 1)
        haskey(full_interior_centered_cache, d1_key) ? full_interior_centered_cache[d1_key] : nothing
    else
        nothing
    end

    if fi_d1 !== nothing
        # Full-interior mode: position-dependent weight+offset matrices for D1
        wm_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(fi_d1.weight_matrix)
        om_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(fi_d1.offset_matrix)
        D1_template = sum(1:fi_d1.padded_len) do j
            w = Symbolics.wrap(wm_c[j, _idxs[dim]])
            off_val = Symbolics.wrap(om_c[j, _idxs[dim]])
            idx_exprs = map(u_spatial) do xv
                eq_d = indexmap[xv]
                raw_idx = _idxs[eq_d] + bases[eq_d]
                combined = isequal(xv, r) ? raw_idx + off_val : raw_idx
                _maybe_wrap(combined, eq_d, is_periodic, gl_vec)
            end
            w * Symbolics.wrap(u_c[idx_exprs...])
        end
    else
        # Standard centered D1 template
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
    end

    # --- Nonlinear Laplacian template (reuse existing infrastructure) ---
    fi_nlap = (full_nonlinlap_cache !== nothing && haskey(full_nonlinlap_cache, (u, r))) ?
              full_nonlinlap_cache[(u, r)] : nothing
    if fi_nlap !== nothing
        nlap_template = _nonlinlap_full_template(
            innerexpr, u, r, nsi, fi_nlap, s, depvars, indexmap, _idxs, bases
        )
    else
        nlap_template = _nonlinlap_template(
            innerexpr, u, r, nsi, s, depvars, indexmap, _idxs, bases,
            is_periodic, gl_vec
        )
    end

    # --- Substitute innerexpr variables at the current point ---
    vr_dict = Dict(var_rules)
    innerexpr_at_i = pde_substitute(innerexpr, vr_dict)

    # --- Combine: innerexpr * (D1/r + nonlinlap) ---
    return innerexpr_at_i * (D1_template / r_at_i + nlap_template)
end

"""
    _build_spherical_rules(pde, s, depvars, derivweights, nonlinlap_cache,
                            spherical_terms_info, indexmap, _idxs, bases, var_rules;
                            full_nonlinlap_cache=nothing,
                            full_interior_centered_cache=nothing)

Build term-level substitution rules for spherical Laplacian patterns.

For each spherical-matched term, builds the ArrayOp-indexed discretization
using `_spherical_template` and multiplies by the outer coefficient.

Returns a vector of `Pair{term => discretized_expr}`.
"""
function _build_spherical_rules(
        pde, s, depvars, derivweights, nonlinlap_cache,
        spherical_terms_info, indexmap, _idxs, bases, var_rules,
        is_periodic=falses(length(bases)),
        gl_vec=zeros(Int, length(bases));
        full_nonlinlap_cache=nothing,
        full_interior_centered_cache=nothing
    )
    vr_dict = Dict(var_rules)
    spherical_rules = Pair[]

    for (term, info) in spherical_terms_info
        haskey(nonlinlap_cache, (info.u, info.r)) || continue
        nsi = nonlinlap_cache[(info.u, info.r)]

        sph_expr = _spherical_template(
            info, nsi, s, depvars, derivweights,
            indexmap, _idxs, bases, var_rules,
            is_periodic, gl_vec;
            full_nonlinlap_cache=full_nonlinlap_cache,
            full_interior_centered_cache=full_interior_centered_cache
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
