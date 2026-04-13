# --- stencil pre-computation ------------------------------------------------

"""
    StencilInfo{T}

Pre-computed information for a single directional finite-difference stencil
(one per derivative order).  Reused for both centred even-order stencils and
the individual wind directions of an upwind odd-order operator.

- `D_op`: the full `DerivativeOperator`, kept around for boundary stencils.
- `offsets`: the signed grid shifts of each tap.  For centred stencils this
  is `half_range(stencil_length)`; for negative-wind upwind it is
  `0:(stencil_length-1)`; for positive-wind upwind it is
  `(-stencil_length+1):0`.
- `is_uniform`: `true` if `D_op.dx isa Number`.
- `weight_matrix`: `nothing` for uniform grids.  For non-uniform grids,
  `stencil_length × num_interior` matrix of per-point weights.
"""
struct StencilInfo{T<:Real}
    D_op::DerivativeOperator     # full operator, needed for boundary stencils
    offsets::Vector{Int}         # signed grid shifts of each tap
    is_uniform::Bool             # true if dx is a Number
    weight_matrix::Union{Nothing, Matrix{T}}  # non-uniform: stencil_length × num_interior
end

"""
    UpwindStencilInfo{T}

Pre-computed information for upwind derivative operators, one `StencilInfo`
per wind direction (`neg` with `offside=0`, `pos` with `offside=d+upwind_order-1`).

Replaces the previous layout that flattened seven parallel fields
(`D_neg`, `D_pos`, `neg_offsets`, …) into a single struct.  Each half is
shape-identical to a centred `StencilInfo`, so the tap-building helpers
can treat them uniformly.
"""
struct UpwindStencilInfo{T<:Real}
    neg::StencilInfo{T}
    pos::StencilInfo{T}
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
struct NonlinlapStencilInfo{T<:Real}
    outer_weights::Any   # uniform: stencil_coefs of D_outer; non-uniform: nothing
    outer_offsets::Vector{Int}
    inner_weights::Any   # uniform: stencil_coefs of D_inner; non-uniform: nothing
    inner_offsets::Vector{Int}
    interp_weights::Any  # uniform: stencil_coefs of interp; non-uniform: nothing
    interp_offsets::Vector{Int}
    combined_lower_bpc::Int   # combined boundary point count, lower side
    combined_upper_bpc::Int   # combined boundary point count, upper side
    is_uniform::Bool
    # Non-uniform weight matrices (nothing for uniform grids)
    outer_weight_matrix::Union{Nothing, Matrix{T}}
    inner_weight_matrix::Union{Nothing, Matrix{T}}
    interp_weight_matrix::Union{Nothing, Matrix{T}}
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
struct WENOStencilInfo{T<:Real}
    epsilon::T                # smoothness indicator regularization parameter
    offsets::Vector{Int}      # [-2, -1, 0, 1, 2] for 5-point stencil
    lower_bpc::Int            # boundary point count, lower side (= 2 for WENO5)
    upper_bpc::Int            # boundary point count, upper side (= 2 for WENO5)
    dx_val::T                 # uniform grid spacing (WENO currently uniform only)
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
    alignment::Type{<:AbstractVarAlign}   # CenterAlignedVar or EdgeAlignedVar
    interior_offsets::Vector{Int}   # [0,1] for CenterAligned, [-1,0] for EdgeAligned
    D_wind::DerivativeOperator      # from windmap[1]
    is_uniform::Bool
    bpc::Int                        # boundary_point_count from derivweights.map
end

