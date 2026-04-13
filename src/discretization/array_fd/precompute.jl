
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
    T = eltype(grid_x)
    positions = Vector{T}(undef, length(offsets))
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
    _build_periodic_wmat(D_op, grid_x, offsets)

Build a `stencil_length × N` weight matrix for stencils on a non-uniform
periodic grid.  `offsets` are the tap offsets relative to each evaluation
point (e.g., `-half_w:half_w` for centered, `0:(sl-1)` for upwind).
"""
function _build_periodic_wmat(D_op, grid_x, offsets)
    N = length(grid_x)
    sl = D_op.stencil_length

    T = _op_eltype(D_op)
    wmat = Matrix{T}(undef, sl, N)
    for g in 1:N
        positions = _periodic_stencil_positions(grid_x, g, offsets)
        wmat[:, g] = calculate_weights(D_op.derivative_order, grid_x[g], positions)
    end
    return wmat
end

"""Centered-stencil convenience: offsets default to `-half_w:half_w`."""
function _build_periodic_wmat(D_op, grid_x)
    half_w = div(D_op.stencil_length, 2)
    return _build_periodic_wmat(D_op, grid_x, -half_w:half_w)
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
                D_op = derivweights.map[Differential(x)^d]
                is_uniform = D_op.dx isa Number
                wmat = if !is_uniform
                    # stencil_coefs is Vector{SVector{L,T}} — convert to L×N matrix
                    _stencil_coefs_to_matrix(D_op)
                else
                    nothing
                end
                info[(u, x, d)] = StencilInfo{_op_eltype(D_op)}(
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
                _stencil_coefs_to_matrix(D_op)
            else
                nothing
            end
            info[(u, r, 1)] = StencilInfo{_op_eltype(D_op)}(
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
                T = _op_eltype(D_neg)
                neg_wmat = !is_uniform ? _stencil_coefs_to_matrix(D_neg) : nothing
                pos_wmat = !is_uniform ? _stencil_coefs_to_matrix(D_pos) : nothing
                neg = StencilInfo{T}(D_neg,
                                      collect(0:(D_neg.stencil_length - 1)),
                                      is_uniform, neg_wmat)
                pos = StencilInfo{T}(D_pos,
                                      collect((-D_pos.stencil_length + 1):0),
                                      is_uniform, pos_wmat)
                info[(u, x, d)] = UpwindStencilInfo{T}(neg, pos)
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

                info[(u, x)] = NonlinlapStencilInfo{_op_eltype(D_inner)}(
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
                outer_wmat  = _stencil_coefs_to_matrix(D_outer)
                inner_wmat  = _stencil_coefs_to_matrix(D_inner)
                interp_wmat = _stencil_coefs_to_matrix(interp)

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

                info[(u, x)] = NonlinlapStencilInfo{_op_eltype(D_inner)}(
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
            T = typeof(float(dx))
            epsilon = convert(T, isempty(F.ps) ? 1e-6 : F.ps[1])
            bpc = div(F.interior_points, 2)  # = 2 for WENO5
            info[(u, x)] = WENOStencilInfo{T}(
                epsilon,
                collect(half_range(F.interior_points)),
                bpc, bpc,
                convert(T, dx)
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
struct FullInteriorStencilInfo{T<:Real}
    weight_matrix::Matrix{T}         # padded_len × N_full_interior
    offset_matrix::Matrix{Int}       # padded_len × N_full_interior
    padded_len::Int                  # max(stencil_length, boundary_stencil_length)
end

"""
    FullInteriorUpwindStencilInfo

Weight and offset matrices covering ALL interior points for upwind derivatives.
Both wind directions get their own matrices.
"""
struct FullInteriorUpwindStencilInfo{T<:Real}
    neg_weight_matrix::Matrix{T}
    neg_offset_matrix::Matrix{Int}
    padded_neg::Int
    pos_weight_matrix::Matrix{T}
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

        T = _op_eltype(D_op)
        wmat = zeros(T, padded, N)
        omat = zeros(Int, padded, N)

        # Pre-allocate reusable vectors for periodic/uniform interior cases
        interior_weights = collect(D_op.stencil_coefs)
        interior_offsets = collect(si.offsets)

        for k in 1:N
            g = lo_vec[dim] + k - 1  # absolute grid index

            if is_periodic[dim]
                # Periodic uniform: always use interior stencil (wrapping handled symbolically)
                weights = interior_weights
                offsets = interior_offsets
            elseif g <= bpc
                # Lower frame: use low_boundary_coefs[g]
                weights = collect(D_op.low_boundary_coefs[g])
                # Taps are at grid indices 1:bsl, relative offsets from g
                offsets = collect((1 - g):(bsl - g))
            elseif g > gl - bpc
                # Upper frame: use high_boundary_coefs[gl - g + 1]
                weights = collect(D_op.high_boundary_coefs[gl - g + 1])
                # Taps are at grid indices (gl-bsl+1):gl
                offsets = collect((gl - bsl + 1 - g):(gl - g))
            else
                # Centered interior
                if si.is_uniform
                    weights = interior_weights
                else
                    # Non-uniform: stencil_coefs is indexed by interior position
                    weights = collect(D_op.stencil_coefs[g - bpc])
                end
                offsets = interior_offsets
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
            usi.neg.D_op, N, lo_vec[dim], gl, usi.neg.offsets, usi.neg.is_uniform;
            dim_periodic=is_periodic[dim]
        )
        pos_wmat, pos_omat, padded_pos = _build_upwind_full_matrices(
            usi.pos.D_op, N, lo_vec[dim], gl, usi.pos.offsets, usi.pos.is_uniform;
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

    T = _op_eltype(D_op)
    wmat = zeros(T, padded, N)
    omat = zeros(Int, padded, N)

    # Pre-allocate reusable vectors for periodic/uniform interior cases
    int_weights = collect(D_op.stencil_coefs)
    int_offsets = collect(interior_offsets)

    for k in 1:N
        g = lo + k - 1  # absolute grid index

        if dim_periodic
            # Periodic uniform: always use interior stencil (wrapping handled symbolically)
            weights = int_weights
            offsets = int_offsets
        elseif g <= offside
            # Lower frame: use low_boundary_coefs[g]
            weights = collect(D_op.low_boundary_coefs[g])
            # Taps at grid indices 1:bsl
            offsets = collect((1 - g):(bsl - g))
        elseif g > gl - bpc
            # Upper frame: use high_boundary_coefs[gl - g + 1]
            weights = collect(D_op.high_boundary_coefs[gl - g + 1])
            # Taps at grid indices (gl-bsl+1):gl
            offsets = collect((gl - bsl + 1 - g):(gl - g))
        else
            # Interior
            if is_uniform
                weights = int_weights
            else
                # Non-uniform: stencil_coefs indexed by interior position
                weights = collect(D_op.stencil_coefs[g - offside])
            end
            offsets = int_offsets
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
struct FullNonlinlapInfo{T<:Real}
    # Outer derivative: 2D matrices indexed by (j_outer, _i)
    outer_weight_matrix::Matrix{T}         # padded_outer × N_full
    padded_outer::Int

    # Interpolation: 3D matrices indexed by (j_outer, j_interp, _i)
    interp_weight_3d::Array{T, 3}          # padded_outer × padded_interp × N_full
    interp_tap_3d::Array{Int, 3}           # padded_outer × padded_interp × N_full
    padded_interp::Int

    # Inner derivative: 3D matrices indexed by (j_outer, j_inner, _i)
    inner_weight_3d::Array{T, 3}           # padded_outer × padded_inner × N_full
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
        T = _op_eltype(D_inner)
        outer_wmat = zeros(T, padded_outer, N)
        interp_w3d = zeros(T, padded_outer, padded_interp, N)
        interp_t3d = ones(Int, padded_outer, padded_interp, N)  # ones = safe default (index 1)
        inner_w3d  = zeros(T, padded_outer, padded_inner, N)
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
                interp_w3d[j_pad, 1, k] = one(T)
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
            weights = collect(D_op.stencil_coefs)
            half_points = [g + off - 1 for off in interior_offsets]
        elseif g <= bpc
            weights = collect(D_op.low_boundary_coefs[g])
            half_points = collect(1:bsl)
        elseif g > gl - bpc
            weights = collect(D_op.high_boundary_coefs[gl - g + 1])
            half_points = collect((N_half - bsl + 1):N_half)
        else
            weights = collect(D_op.stencil_coefs)
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
            weights = collect(D_op.low_boundary_coefs[p])
            half_points = collect(1:bsl)
        elseif p > N_eff - bpc
            weights = collect(D_op.high_boundary_coefs[N_eff - p + 1])
            half_points = collect((N_half - bsl + 1):N_half)
        else
            weights = collect(D_op.stencil_coefs[p - bpc])
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
        weights = collect(D_op.stencil_coefs)
        grid_taps = [h + off for off in interior_offsets]
    elseif h <= bpc
        # Lower boundary: use low_boundary_coefs[h]
        weights = collect(D_op.low_boundary_coefs[h])
        # Boundary stencil taps at grid points 1:bsl
        grid_taps = collect(1:bsl)
    elseif h > N_half - bpc
        # Upper boundary: use high_boundary_coefs[N_half - h + 1]
        weights = collect(D_op.high_boundary_coefs[N_half - h + 1])
        # Boundary stencil taps at grid points (gl-bsl+1):gl
        grid_taps = collect((gl - bsl + 1):gl)
    else
        # Interior
        if is_uniform
            weights = collect(D_op.stencil_coefs)
        else
            # Non-uniform: stencil_coefs indexed by interior position
            weights = collect(D_op.stencil_coefs[h - bpc])
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

