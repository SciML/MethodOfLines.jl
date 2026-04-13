# --- equation comparison ----------------------------------------------------

"""
    _equations_match(eq_template, eq_scalar; atol = 1e-8)

Compare two equations for equivalence.  First tries exact structural comparison
via `isequal`.  If that fails, falls back to numerical comparison by
substituting deterministic values for all free symbolic variables.  This handles
mathematically equivalent expressions that differ only in symbolic form
(e.g., different sign distribution or term ordering).

The numerical comparison uses three irrational-valued substitution points and
requires `|lhs1 - rhs1 - (lhs2 - rhs2)| ≤ atol` at each.  The default
`atol = 1e-8` is calibrated for double-precision stencil coefficients and is
intentionally a mixed (absolute + relative) tolerance: both sides are built
from the same symbolic PDE so their magnitudes cancel almost entirely, and the
residual only reflects floating-point noise in the weight products.  Tighter
tolerances cause spurious fallbacks on higher-order stencils; looser
tolerances mask real bugs.

This check is advisory — if it returns `false` the caller falls back to the
per-point scalar path, which is always correct.  A false negative therefore
only costs a little discretization time, not correctness.
"""
function _equations_match(eq_template, eq_scalar; atol::Real = 1e-8)
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
    # Use deterministic test points (irrational-ish values to avoid accidental zeros)
    test_offsets = (0.7182818, 1.4142135, 2.2360679)
    for offset in test_offsets
        subs = Dict(v => offset + i * 0.31415926 for (i, v) in enumerate(all_vars))
        val = Symbolics.value(substitute(diff_expr, subs))
        if !(val isa Number) || abs(val) > atol
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

"""
    _local_sample_indices(n_region) -> Vector{NTuple{N,Int}}

Pick up to 3 representative local index tuples inside an ArrayOp region of
size `n_region` (one per dimension): the first point `(1,…,1)`, the midpoint
`(cld.(n_region,2)...)`, and the last point `(n_region...)`.  Duplicates
(common in 1-wide regions) are removed.
"""
function _local_sample_indices(n_region)
    N = length(n_region)
    first_idx = ntuple(_ -> 1, N)
    mid_idx = ntuple(d -> cld(n_region[d], 2), N)
    last_idx = ntuple(d -> n_region[d], N)
    samples = NTuple{N, Int}[]
    for p in (first_idx, mid_idx, last_idx)
        p in samples || push!(samples, p)
    end
    return samples
end

"""
    _validate_arrayop_or_fallback(candidate, sample_at, n_region, lo, hi, ndim,
                                   is_periodic, s, depvars, pde, derivweights,
                                   bcmap, eqvar, indexmap, boundaryvalfuncs;
                                   debug_label="ArrayOp", validate=false)

Validate the ArrayOp `candidate` by comparing the template instantiated at
several local points against the scalar path at the corresponding absolute
grid points.  If any comparison fails, fall back to per-point scalar
discretization over `lo[d]:hi[d]`.

Sampled local indices come from [`_local_sample_indices`](@ref): the first
point, the midpoint, and the last point of the ArrayOp region (up to 3
distinct points).  This catches bugs that manifest at specific grid positions
without paying the cost of checking every point.

Skips validation entirely for periodic dimensions — the two paths use
structurally different wrapping that prevents `_equations_match` from
succeeding even when numerics agree, and the periodic non-uniform path
already falls back to the standard path before reaching here.
"""
function _validate_arrayop_or_fallback(candidate, sample_at,
                                        n_region::AbstractVector{Int},
                                        lo::AbstractVector{Int},
                                        hi::AbstractVector{Int},
                                        ndim::Int,
                                        is_periodic::AbstractVector{Bool},
                                        s, depvars, pde, derivweights,
                                        bcmap, eqvar, indexmap, boundaryvalfuncs;
                                        debug_label="ArrayOp", validate::Bool=false)
    !validate && return candidate
    any(is_periodic) && return candidate   # periodic path cannot be symbolically compared
    for local_idx in _local_sample_indices(n_region)
        # `Val(ndim)` forces `ntuple` to specialize on the statically-known
        # dimension count — without this the return type widens to `Tuple`
        # and the downstream `CartesianIndex(...)` / `discretize_equation_at_point`
        # calls hit runtime dispatch (JET-flagged).
        II_check = CartesianIndex(ntuple(d -> lo[d] + local_idx[d] - 1, Val(ndim)))
        eq_scalar = discretize_equation_at_point(
            II_check, s, depvars, pde, derivweights, bcmap,
            eqvar, indexmap, boundaryvalfuncs
        )
        eq_template = sample_at(local_idx)
        if !_equations_match(eq_template, eq_scalar)
            @debug "$debug_label validation failed" local_idx eq_template eq_scalar
            fallback_rect = CartesianIndices(Tuple(lo[d]:hi[d] for d in 1:ndim))
            return collect(vec(map(fallback_rect) do II
                discretize_equation_at_point(
                    II, s, depvars, pde, derivweights, bcmap,
                    eqvar, indexmap, boundaryvalfuncs
                )
            end))
        end
    end
    return candidate
end

