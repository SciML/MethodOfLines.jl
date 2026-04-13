# --- Spherical Laplacian ArrayOp rules --------------------------------------

"""
    _spherical_template(ctx::ArrayOpContext, info, nsi, var_rules;
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
function _spherical_template(ctx::ArrayOpContext, info, nsi, var_rules;
                              full_nonlinlap_cache=nothing,
                              full_interior_centered_cache=nothing)
    s            = ctx.s
    depvars      = ctx.depvars
    derivweights = ctx.derivweights
    indexmap     = ctx.indexmap
    _idxs        = ctx.idxs
    bases        = ctx.bases
    is_periodic  = ctx.is_periodic
    gl_vec       = ctx.gl_vec

    u = info.u
    r = info.r
    innerexpr = info.innerexpr

    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = _ConstSR(u_raw)
    u_spatial = ivs(u, s)

    # Compute the grid coordinate at the current ArrayOp index.
    # Use Const-wrapped grid lookup (works for both uniform and non-uniform).
    # Safe because the result goes into termlevel_dict which bypasses pde_substitute.
    grid_r = collect(s.grid[r])
    dim = indexmap[r]
    grid_r_c = _ConstSR(grid_r)
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
        wm_c = _ConstSR(fi_d1.weight_matrix)
        om_c = _ConstSR(fi_d1.offset_matrix)
        D1_template = sum(1:fi_d1.padded_len) do j
            w = Symbolics.wrap(wm_c[j, _idxs[dim]])
            off_val = Symbolics.wrap(om_c[j, _idxs[dim]])
            w * _tap_expr(ctx, u_c, u_spatial, r, off_val)
        end
    else
        # Standard centered D1 template
        D1_op = derivweights.map[Differential(r)]
        d1_is_uniform = D1_op.dx isa Number
        d1_offsets = collect(half_range(D1_op.stencil_length))
        d1_taps = [_tap_expr(ctx, u_c, u_spatial, r, off) for off in d1_offsets]
        if d1_is_uniform
            D1_template = sym_dot(D1_op.stencil_coefs, d1_taps)
        else
            d1_wmat_c = _ConstSR(
                _stencil_coefs_to_matrix(D1_op)
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
        nlap_template = _nonlinlap_full_template(ctx, innerexpr, u, r, nsi, fi_nlap)
    else
        nlap_template = _nonlinlap_template(ctx, innerexpr, u, r, nsi)
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
function _build_spherical_rules(ctx::ArrayOpContext, caches::StencilCaches,
                                 pde, var_rules)
    s                             = ctx.s
    depvars                       = ctx.depvars
    derivweights                  = ctx.derivweights
    indexmap                      = ctx.indexmap
    _idxs                         = ctx.idxs
    bases                         = ctx.bases
    is_periodic                   = ctx.is_periodic
    gl_vec                        = ctx.gl_vec
    nonlinlap_cache               = caches.nonlinlap
    spherical_terms_info          = caches.spherical_terms
    full_nonlinlap_cache          = caches.full_nonlinlap
    full_interior_centered_cache  = caches.full_centered

    vr_dict = Dict(var_rules)
    spherical_rules = Pair[]

    for (term, info) in spherical_terms_info
        haskey(nonlinlap_cache, (info.u, info.r)) || continue
        nsi = nonlinlap_cache[(info.u, info.r)]

        sph_expr = _spherical_template(
            ctx, info, nsi, var_rules;
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

