function generate_array_interior_eqs(
        s, depvars, pde, derivweights, bcmap, eqvar,
        indexmap, boundaryvalfuncs, interior_ranges;
        validate=false
    )

    # -- Fast path for equations with no spatial derivatives --------------------
    eq_has_spatial_derivs = _contains_spatial_diff(Symbolics.unwrap(pde.lhs), s.x̄) ||
                           _contains_spatial_diff(Symbolics.unwrap(pde.rhs), s.x̄)

    ndim = length(interior_ranges)
    lo_vec = [r[1] for r in interior_ranges]
    hi_vec = [r[2] for r in interior_ranges]

    if !eq_has_spatial_derivs
        # Fast path for equations with no spatial derivatives (e.g., algebraic
        # equations like R(t,x) ~ f(params)).  These need no boundary frame,
        # no WENO/upwind handling, and no validation against the scalar path.
        # Build a simple ArrayOp using variable substitution rules only.
        # Note: stencil_cache is still needed because _build_interior_arrayop
        # iterates over derivweights.orders[x] (global) for all depvars.
        n_region = [hi_vec[d] - lo_vec[d] + 1 for d in 1:ndim]
        if all(n_region .> 0)
            ctx = ArrayOpContext(n_region, lo_vec, s, depvars, derivweights, indexmap)
            caches = StencilCaches(
                centered = precompute_stencils(s, depvars, derivweights),
            )
            candidate, _ = _build_interior_arrayop(ctx, caches, n_region, pde, bcmap, eqvar)
            return candidate
        else
            return Equation[]
        end
    end

    # -- determine whether the ArrayOp path can handle this PDE ---------------
    # Use lightweight checks that avoid full stencil precomputation.
    has_odd_orders = any(
        any(isodd(d) for d in derivweights.orders[x])
        for u in depvars for x in ivs(u, s)
    )
    can_upwind = has_odd_orders && derivweights.advection_scheme isa UpwindScheme
    is_staggered = s.staggeredvars !== nothing
    _is_weno = derivweights.advection_scheme isa FunctionalScheme &&
               derivweights.advection_scheme.name == "WENO"
    _has_halfoffset = any(
        haskey(derivweights.halfoffsetmap[1], Differential(x)) &&
        haskey(derivweights.halfoffsetmap[2], Differential(x)) &&
        haskey(derivweights.interpmap, x)
        for u in depvars for x in ivs(u, s)
    )

    # Detect spherical Laplacian terms first (they take priority over nonlinlap).
    spherical_terms_info = _detect_spherical_terms(pde, s, depvars)
    has_spherical = !isempty(spherical_terms_info) && _has_halfoffset

    # Detect nonlinear Laplacian terms.
    nonlinlap_terms = _detect_nonlinlap_terms(pde, s, depvars, spherical_terms_info)
    has_nonlinlap = !isempty(nonlinlap_terms) && _has_halfoffset

    can_centered = !(derivweights.advection_scheme isa FunctionalScheme) || _is_weno
    can_template = !has_odd_orders || can_upwind || can_centered || has_nonlinlap || has_spherical || _is_weno || is_staggered

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

    # -- Precompute stencil caches (only reached for equations that use ArrayOps)
    upwind_cache = precompute_upwind_stencils(s, depvars, derivweights)
    nonlinlap_cache = precompute_nonlinlap_stencils(s, depvars, derivweights)
    weno_cache = precompute_weno_stencils(s, depvars, derivweights)
    staggered_cache = precompute_staggered_stencils(s, depvars, derivweights)

    # Refine lightweight checks with actual cache results
    has_weno = !isempty(weno_cache)
    has_nonlinlap = !isempty(nonlinlap_terms) && !isempty(nonlinlap_cache)
    has_spherical = !isempty(spherical_terms_info) && !isempty(nonlinlap_cache)
    is_staggered = !isempty(staggered_cache)

    sph_vars = if has_spherical
        unique([(info.u, info.r) for info in values(spherical_terms_info)])
    else
        nothing
    end
    stencil_cache = precompute_stencils(s, depvars, derivweights; spherical_vars=sph_vars)

    # -- N-D ArrayOp path ------------------------------------------------------
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
        for x in ivs(u, s)
            eq_dim = indexmap[x]  # dimension index in eqvar's ordering
            for d in derivweights.orders[x]
                if iseven(d)
                    bpc = stencil_cache[(u, x, d)].D_op.boundary_point_count
                    max_lower_bpc[eq_dim] = max(max_lower_bpc[eq_dim], bpc)
                    max_upper_bpc[eq_dim] = max(max_upper_bpc[eq_dim], bpc)
                elseif isodd(d) && haskey(upwind_cache, (u, x, d))
                    usi = upwind_cache[(u, x, d)]
                    lower_bpc = max(usi.neg.D_op.offside, usi.pos.D_op.offside)
                    upper_bpc = max(usi.neg.D_op.boundary_point_count, usi.pos.D_op.boundary_point_count)
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

        ctx = ArrayOpContext(n_region, lo_vec, s, depvars, derivweights, indexmap;
                              is_periodic=is_periodic, gl_vec=gl_vec)
        caches = StencilCaches(
            centered = stencil_cache, upwind = upwind_cache,
            nonlinlap = nonlinlap_cache, weno = weno_cache,
            staggered = staggered_cache,
            full_centered = fi_centered, full_upwind = fi_upwind,
            full_nonlinlap = fi_nonlinlap,
            spherical_terms = spherical_terms_info,
        )

        eqs_centered = if all(n_region .> 0)
            candidate, sample_at = _build_interior_arrayop(ctx, caches, n_region,
                                                            pde, bcmap, eqvar)
            _validate_arrayop_or_fallback(
                candidate, sample_at, n_region, lo_vec, hi_vec, ndim,
                is_periodic, s, depvars, pde, derivweights,
                bcmap, eqvar, indexmap, boundaryvalfuncs;
                debug_label="Full-interior ArrayOp", validate=validate
            )
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
            ctx = ArrayOpContext(n_centered, lo_centered, s, depvars, derivweights,
                                  indexmap; is_periodic=is_periodic, gl_vec=gl_vec)
            caches = StencilCaches(
                centered = stencil_cache, upwind = upwind_cache,
                nonlinlap = nonlinlap_cache, weno = weno_cache,
                staggered = staggered_cache,
                spherical_terms = spherical_terms_info,
            )
            candidate, sample_at = _build_interior_arrayop(ctx, caches, n_centered,
                                                            pde, bcmap, eqvar)
            _validate_arrayop_or_fallback(
                candidate, sample_at, n_centered, lo_centered, hi_centered, ndim,
                is_periodic, s, depvars, pde, derivweights,
                bcmap, eqvar, indexmap, boundaryvalfuncs;
                validate=validate
            )
        else
            Equation[]
        end
    end

    return collect(vcat(eqs_boundary, eqs_centered))
end

