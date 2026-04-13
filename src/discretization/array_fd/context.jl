# --- module-level aliases ---------------------------------------------------

"""
Alias for the `Const` wrapper type used throughout ArrayOp construction.

`SymbolicUtils.Const{SymReal}` is the top-level alias for `BSImpl.Const{SymReal}`
exposed via re-export in `Symbolics.jl`.  Reaching through `SymbolicUtils.Const`
rather than `SymbolicUtils.BSImpl.Const` avoids a dependency on the internal
Moshi `@data` module path, which is the more likely identifier to move in a
SymbolicUtils minor rewrite.
"""
const _ConstSR = SymbolicUtils.Const{SymbolicUtils.SymReal}

"""
    ArrayOpContext

Bundle of state threaded through every `_build_*_rules` helper.  Exists so
rule-builder signatures don't each have to repeat the 8-tuple
`(s, depvars, derivweights, indexmap, idxs, bases, is_periodic, gl_vec)`.

- `idxs`: per-dimension symbolic ArrayOp index variables (`_i1`, `_i2`, …)
- `bases`: `lo_local[d] - 1` — absolute grid index = local index + base
- `is_periodic`: one bool per spatial dim
- `gl_vec`: full grid length per spatial dim (used for periodic wrapping)
"""
struct ArrayOpContext{S, DV, DW, IM}
    s::S
    depvars::DV
    derivweights::DW
    indexmap::IM
    idxs::Vector
    bases::Vector{Int}
    is_periodic::Vector{Bool}
    gl_vec::Vector{Int}
end

"""
    ArrayOpContext(n_local, lo_local, s, depvars, derivweights, indexmap;
                   is_periodic, gl_vec)

Construct an `ArrayOpContext` for an ArrayOp region of shape `n_local`
whose first local index maps to absolute grid index `lo_local[d]`.
Allocates the symbolic index variables and computes `bases`.
"""
function ArrayOpContext(n_local, lo_local, s, depvars, derivweights, indexmap;
                        is_periodic = falses(length(n_local)),
                        gl_vec = zeros(Int, length(n_local)))
    ndim = length(n_local)
    _idxs_arr = SymbolicUtils.idxs_for_arrayop(SymbolicUtils.SymReal)
    idxs = [_idxs_arr[d] for d in 1:ndim]
    bases = [lo_local[d] - 1 for d in 1:ndim]
    return ArrayOpContext(s, depvars, derivweights, indexmap,
                          idxs, bases, collect(is_periodic), collect(gl_vec))
end

"""
    StencilCaches

Bundle of the eight precomputed-stencil dictionaries threaded through the
ArrayOp build pipeline.  Unused slots hold empty dicts or `nothing` so we
don't need a separate "present / absent" field per cache.
"""
struct StencilCaches
    centered::Dict{Any, Any}
    upwind::Dict{Any, Any}
    nonlinlap::Dict{Any, Any}
    weno::Dict{Any, Any}
    staggered::Dict{Any, Any}
    full_centered::Any     # Dict or nothing
    full_upwind::Any       # Dict or nothing
    full_nonlinlap::Any    # Dict or nothing
    spherical_terms::Dict{Any, Any}
end

"""Default empty `StencilCaches` for PDE paths with no spatial derivatives."""
function StencilCaches(; centered = Dict{Any,Any}(), upwind = Dict{Any,Any}(),
                         nonlinlap = Dict{Any,Any}(), weno = Dict{Any,Any}(),
                         staggered = Dict{Any,Any}(),
                         full_centered = nothing, full_upwind = nothing,
                         full_nonlinlap = nothing,
                         spherical_terms = Dict{Any,Any}())
    return StencilCaches(centered, upwind, nonlinlap, weno, staggered,
                         full_centered, full_upwind, full_nonlinlap,
                         spherical_terms)
end

"""
    _tap_expr(ctx, u_c, u_spatial)              # at-point (no shift, no wrap)
    _tap_expr(ctx, u_c, u_spatial, x, off)      # single-dim shift, wrapped
    _tap_expr(ctx, u_c, u_spatial, shifts::Dict)# multi-dim shifts, wrapped

Build the symbolic tap expression `u[base + local_idx + shift]` for the
grid-function array `u_c` indexed by `u_spatial` spatial variables.

- The no-shift variant is used for `var_rules` (depvar value at the ArrayOp
  point itself).  It **does not** thread through `_maybe_wrap`: in periodic
  mode the ArrayOp range already covers every physical point, and wrapping
  the at-point index with `IfElse.ifelse` causes `pde_substitute`'s
  `maketerm` reconstruction to choke on nested `getindex` branches.
- The single-dim variant is used by every centered / upwind / staggered
  stencil builder — one `off` integer applied to the named `x` dimension.
  Shifted indices need periodic wrapping since stencil taps can reach past
  the ArrayOp boundary.
- The multi-dim variant is used by mixed cross-derivatives (`Dxy`), where
  different offsets apply along different dimensions simultaneously.
"""
function _tap_expr(ctx::ArrayOpContext, u_c, u_spatial)
    idx_exprs = [ctx.idxs[ctx.indexmap[xv]] + ctx.bases[ctx.indexmap[xv]]
                 for xv in u_spatial]
    return Symbolics.wrap(u_c[idx_exprs...])
end

function _tap_expr(ctx::ArrayOpContext, u_c, u_spatial, x, off)
    idx_exprs = map(u_spatial) do xv
        eq_d = ctx.indexmap[xv]
        raw_idx = ctx.idxs[eq_d] + ctx.bases[eq_d]
        if isequal(xv, x)
            raw_idx = raw_idx + off
        end
        _maybe_wrap(raw_idx, eq_d, ctx.is_periodic, ctx.gl_vec)
    end
    return Symbolics.wrap(u_c[idx_exprs...])
end

function _tap_expr(ctx::ArrayOpContext, u_c, u_spatial, shifts::AbstractDict)
    idx_exprs = map(u_spatial) do xv
        eq_d = ctx.indexmap[xv]
        raw_idx = ctx.idxs[eq_d] + ctx.bases[eq_d]
        if haskey(shifts, xv)
            raw_idx = raw_idx + shifts[xv]
        end
        _maybe_wrap(raw_idx, eq_d, ctx.is_periodic, ctx.gl_vec)
    end
    return Symbolics.wrap(u_c[idx_exprs...])
end

"""Extract the element type `T` from a `DerivativeOperator{T, ...}`."""
_op_eltype(::DerivativeOperator{T}) where {T} = T

"""
    _stencil_coefs_to_matrix(D_op)

Convert a DerivativeOperator's stencil coefficients (Vector{SVector}) to a Matrix.
"""
_stencil_coefs_to_matrix(D_op) = reduce(hcat, D_op.stencil_coefs)
