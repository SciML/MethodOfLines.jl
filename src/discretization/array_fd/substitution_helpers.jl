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

Periodic grids in MethodOfLines store `N` grid points where point 1 and
point N represent the *same* physical location on opposite sides of the
periodic seam — they are aliases, not distinct physical points.  The
canonical set of unique physical points is therefore `2:N` (or equivalently
`1:(N-1)`).  We use `2:N` as the canonical range, which yields the mapping:

- index ≤ 1 → index + (N-1)   (e.g., 1 → N, 0 → N-1, -1 → N-2)
- index > N → index - (N-1)   (e.g., N+1 → 2, N+2 → 3)

Downstream consumers that assume index 1 is a distinct physical point
(plotting, post-processing, result export) should be aware of the alias —
the aliased value matches what the scalar path writes at that index, so
numerical values at index 1 and index N are always equal in periodic runs.

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

# --- Derivative detection ---------------------------------------------------
#
# These thin wrappers exist so call sites read naturally; the heavy lifting is
# done by `PDEBase.differential_order`, which shares recursion logic with the
# rest of the PDEBase / MOL pipeline.

_contains_time_diff(expr_raw, time) = !isempty(differential_order(expr_raw, time))

_contains_spatial_diff(expr_raw, spatial_vars) =
    any(x -> !isempty(differential_order(expr_raw, x)), spatial_vars)

