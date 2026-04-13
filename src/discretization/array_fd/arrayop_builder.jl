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

Returns `(eqs, sample_at)` where `eqs` is a single-element vector containing
the ArrayOp equation, and `sample_at(local_idx::NTuple{N,Int}) -> Equation`
is a closure that instantiates the template at a given *local* index tuple
within the ArrayOp region (local index 1 maps to absolute grid index
`lo_centered[d]`).  `sample_at` is consumed by
[`_validate_arrayop_or_fallback`](@ref), which samples multiple points to
catch position-dependent bugs.
"""
function _build_interior_arrayop(ctx::ArrayOpContext, caches::StencilCaches,
                                  n_centered, pde, bcmap, eqvar)
    # Destructure for brevity below.  Names retain their historical meaning.
    s            = ctx.s
    depvars      = ctx.depvars
    derivweights = ctx.derivweights
    indexmap     = ctx.indexmap
    _idxs        = ctx.idxs
    bases        = ctx.bases
    is_periodic  = ctx.is_periodic
    gl_vec       = ctx.gl_vec

    stencil_cache                 = caches.centered
    upwind_cache                  = caches.upwind
    nonlinlap_cache               = caches.nonlinlap
    weno_cache                    = caches.weno
    staggered_cache               = caches.staggered
    full_interior_centered_cache  = caches.full_centered
    full_interior_upwind_cache    = caches.full_upwind
    full_nonlinlap_cache          = caches.full_nonlinlap
    spherical_terms_info          = caches.spherical_terms

    ndim = length(n_centered)

    # -- FD rules for centred (even-order) and staggered (odd-order) derivatives
    fd_rules = Pair[]
    for u in depvars
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = _ConstSR(u_raw)
        u_spatial = ivs(u, s)
        for x in u_spatial
            for d in derivweights.orders[x]
                # Staggered odd-order derivatives: fixed offset per alignment
                if !iseven(d) && staggered_cache !== nothing && haskey(staggered_cache, (u, x, d))
                    ssi = staggered_cache[(u, x, d)]
                    taps = [_tap_expr(ctx, u_c, u_spatial, x, off) for off in ssi.interior_offsets]
                    expr = sym_dot(ssi.D_wind.stencil_coefs, taps)
                    push!(fd_rules, (Differential(x)^d)(u) => expr)
                    continue
                end
                si = stencil_cache[(u, x, d)]

                if full_interior_centered_cache !== nothing && haskey(full_interior_centered_cache, (u, x, d))
                    # Full-interior mode: position-dependent weight+offset matrices
                    fisi = full_interior_centered_cache[(u, x, d)]
                    wm_c = _ConstSR(fisi.weight_matrix)
                    om_c = _ConstSR(fisi.offset_matrix)
                    dim = indexmap[x]
                    expr = sum(1:fisi.padded_len) do j
                        w = Symbolics.wrap(wm_c[j, _idxs[dim]])
                        off_val = Symbolics.wrap(om_c[j, _idxs[dim]])
                        w * _tap_expr(ctx, u_c, u_spatial, x, off_val)
                    end
                else
                    # Standard centered-only mode
                    taps = [_tap_expr(ctx, u_c, u_spatial, x, off) for off in si.offsets]
                    if si.is_uniform
                        expr = sym_dot(si.D_op.stencil_coefs, taps)
                    else
                        # Non-uniform: index into weight matrix with symbolic point index.
                        dim = indexmap[x]
                        bpc = si.D_op.boundary_point_count
                        if is_periodic[dim]
                            # Periodic non-uniform: extended N-column weight matrix
                            ext_wmat = _build_periodic_wmat(si.D_op, collect(s.grid[x]))
                            wmat_c = _ConstSR(ext_wmat)
                            point_idx = _idxs[dim] + bases[dim]
                        else
                            wmat_c = _ConstSR(si.weight_matrix)
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
        u_c = _ConstSR(u_raw)
        u_spatial = ivs(u, s)
        push!(var_rules, u => _tap_expr(ctx, u_c, u_spatial))
    end
    eqvar_ivs = ivs(eqvar, s)
    for x in eqvar_ivs
        grid_c = _ConstSR(collect(s.grid[x]))
        dim = indexmap[x]
        # grid vectors are 1-D so we build the index by hand rather than
        # calling `_tap_expr` (which assumes a depvar's full spatial tuple).
        push!(var_rules, x => Symbolics.wrap(grid_c[_idxs[dim] + bases[dim]]))
    end

    # -- Upwind (odd-order) rules with IfElse wind switching ------------------
    # Staggered grids use alignment-based offsets instead of upwind switching,
    # matching the scalar path which sets advection_rules = [] for StaggeredGrid.
    upwind_rules = Pair[]
    if !isempty(upwind_cache) && (staggered_cache === nothing || isempty(staggered_cache))
        upwind_term_rules, upwind_fallback_rules =
            _build_upwind_rules(ctx, caches, pde, bcmap, var_rules)
        upwind_rules = upwind_term_rules
        # Upwind fallback rules are expression-level (they replace bare
        # derivatives like Dx(u), not entire PDE terms).  Put them in
        # fd_rules so they are applied via pde_substitute, matching the
        # scalar path's generate_winding_rules fallback behaviour.
        # They overwrite the centred FD rules for the same derivatives
        # added above (Dict keeps the last entry for duplicate keys).
        append!(fd_rules, upwind_fallback_rules)
    end

    # -- Mixed derivative rules -----------------------------------------------
    mixed_rules = _build_mixed_derivative_rules(ctx)

    # -- Nonlinear Laplacian rules --------------------------------------------
    nl_rules = Pair[]
    if !isempty(nonlinlap_cache)
        nl_rules = _build_nonlinlap_rules(ctx, caches, pde, var_rules)
    end

    # -- Spherical Laplacian rules -------------------------------------------
    sph_rules = Pair[]
    if !isempty(spherical_terms_info) && !isempty(nonlinlap_cache)
        sph_rules = _build_spherical_rules(ctx, caches, pde, var_rules)
    end

    # -- WENO rules ----------------------------------------------------------
    weno_rules = Pair[]
    if !isempty(weno_cache)
        weno_rules = _build_weno_rules(ctx, caches, pde, var_rules)
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
    # Check both sides — the time derivative could be on either side of the
    # rearranged equation (though typically lhs after rearrangement).
    is_algebraic = !_contains_time_diff(Symbolics.unwrap(pde.lhs), s.time) &&
                   !_contains_time_diff(Symbolics.unwrap(pde.rhs), s.time)

    # -- Separate time derivative from spatial terms --------------------------
    eqvar_raw = Symbolics.unwrap(s.discvars[eqvar])
    eqvar_c = _ConstSR(eqvar_raw)
    eqvar_idx_exprs = [_idxs[d] + bases[d] for d in 1:ndim]

    # Closure that instantiates the template at any local index tuple —
    # used by `_validate_arrayop_or_fallback` to sample multiple points.
    sample_at = let t_lhs = template_lhs, t_rhs = template_rhs, idxs = _idxs, nd = ndim
        function (local_idx::NTuple)
            sub = Dict{Any, Any}(idxs[d] => local_idx[d] for d in 1:nd)
            lhs_k = pde_substitute(t_lhs, sub)
            rhs_k = pde_substitute(t_rhs, sub)
            return lhs_k ~ rhs_k
        end
    end

    if is_algebraic
        # Algebraic equation: no Dt term.  Wrap both sides directly as ArrayOps.
        ao_ranges = Dict(_idxs[d] => (1:1:n_centered[d]) for d in 1:ndim)

        lhs_raw = Symbolics.unwrap(template_lhs)
        rhs_raw = Symbolics.unwrap(template_rhs)
        # Handle numeric RHS (e.g., literal 0 after rearrangement)
        if !(lhs_raw isa SymbolicUtils.BasicSymbolic)
            lhs_raw = _ConstSR(lhs_raw)
        end
        if !(rhs_raw isa SymbolicUtils.BasicSymbolic)
            rhs_raw = _ConstSR(rhs_raw)
        end

        lhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
            _idxs, lhs_raw, +, nothing, ao_ranges
        )
        rhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
            _idxs, rhs_raw, +, nothing, ao_ranges
        )
        arrayop_eq = Symbolics.wrap(lhs_ao) ~ Symbolics.wrap(rhs_ao)

        return [arrayop_eq], sample_at
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

    return [arrayop_eq], sample_at
end

