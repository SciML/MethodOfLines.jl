# --- Nonlinear Laplacian ArrayOp rules --------------------------------------

"""
    _nonlinlap_template(ctx::ArrayOpContext, expr_sym, u, x, nsi)

Build the ArrayOp-indexed discretization of `Dx(expr_sym * Dx(u))` using the
precomputed `NonlinlapStencilInfo`.

At each outer stencil half-point:
1. Interpolate all depvars and grid coordinates to the half-point
2. Compute the inner derivative of `u` at the half-point
3. Substitute rules into `expr_sym * Dx(u)` to get the inner expression

Then take the outer finite difference across the inner expressions.
"""
function _nonlinlap_template(ctx::ArrayOpContext, expr_sym, u, x, nsi)
    s           = ctx.s
    depvars     = ctx.depvars
    indexmap    = ctx.indexmap
    _idxs       = ctx.idxs
    bases       = ctx.bases
    is_periodic = ctx.is_periodic
    gl_vec      = ctx.gl_vec

    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = _ConstSR(u_raw)
    u_spatial = ivs(u, s)
    dim = indexmap[x]

    # Pre-wrap Const weight matrices for non-uniform case (outside the loop)
    if !nsi.is_uniform
        interp_wmat_c = _ConstSR(nsi.interp_weight_matrix)
        inner_wmat_c = _ConstSR(nsi.inner_weight_matrix)
        outer_wmat_c = _ConstSR(nsi.outer_weight_matrix)
    end

    # Pre-collect and Const-wrap grids outside the outer_off loop
    grid_x_c = _ConstSR(collect(s.grid[x]))
    grid_iv_cs = Dict(xv => _ConstSR(collect(s.grid[xv])) for xv in s.x̄ if haskey(indexmap, xv))

    inner_exprs = map(nsi.outer_offsets) do outer_off
        # --- Interpolation rules for variables at this half-point ---
        interp_var_rules = Pair[]
        for v in depvars
            v_raw = Symbolics.unwrap(s.discvars[v])
            v_c = _ConstSR(v_raw)
            v_spatial = ivs(v, s)
            # Offset: -1 for clipped grid, + outer_off, + interp offset
            taps = [_tap_expr(ctx, v_c, v_spatial, x, outer_off + ioff - 1)
                    for ioff in nsi.interp_offsets]
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
                taps = map(nsi.interp_offsets) do ioff
                    raw_idx = _idxs[dim] + bases[dim] + outer_off + ioff - 1
                    Symbolics.wrap(grid_x_c[_maybe_wrap(raw_idx, dim, is_periodic, gl_vec)])
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
                xv_dim = indexmap[xv]
                push!(interp_iv_rules, xv => Symbolics.wrap(grid_iv_cs[xv][_idxs[xv_dim] + bases[xv_dim]]))
            end
        end

        # --- Inner derivative of u at this half-point: Dx(u) ---
        inner_deriv_taps = [_tap_expr(ctx, u_c, u_spatial, x, outer_off + ioff - 1)
                            for ioff in nsi.inner_offsets]
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
    _nonlinlap_full_template(ctx::ArrayOpContext, expr_sym, u, x, nsi, fi_nlap)

Full-interior version of `_nonlinlap_template`.  Uses pre-expanded 3D
weight+tap matrices from `fi_nlap` (a `FullNonlinlapInfo`) so that a single
ArrayOp covers ALL interior points including boundary-proximity ones.

Tap positions are absolute (from `interp_tap_3d` and `inner_tap_3d`).
For periodic dimensions, tap indices are pre-wrapped at precompute time
(see `precompute_full_nonlinlap`), so no symbolic `_maybe_wrap` is needed.
"""
function _nonlinlap_full_template(ctx::ArrayOpContext, expr_sym, u, x, nsi, fi_nlap)
    s          = ctx.s
    depvars    = ctx.depvars
    indexmap   = ctx.indexmap
    _idxs      = ctx.idxs
    bases_full = ctx.bases

    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = _ConstSR(u_raw)
    u_spatial = ivs(u, s)
    dim = indexmap[x]

    wrap = Symbolics.wrap

    # Const-wrap the precomputed matrices
    outer_wm_c   = _ConstSR(fi_nlap.outer_weight_matrix)
    interp_w3d_c = _ConstSR(fi_nlap.interp_weight_3d)
    interp_t3d_c = _ConstSR(fi_nlap.interp_tap_3d)
    inner_w3d_c  = _ConstSR(fi_nlap.inner_weight_3d)
    inner_t3d_c  = _ConstSR(fi_nlap.inner_tap_3d)

    # NOTE: All Const indexing must use raw SymReal indices (not Num/wrapped).
    # Only wrap() the final product expressions.

    # Pre-collect and Const-wrap grids outside the j_outer loop
    grid_x_c = _ConstSR(collect(s.grid[x]))
    grid_iv_cs = Dict(xv => _ConstSR(collect(s.grid[xv])) for xv in s.x̄ if haskey(indexmap, xv))

    inner_exprs = map(1:fi_nlap.padded_outer) do j_outer
        # --- Interpolation rules for variables at this half-point ---
        interp_var_rules = Pair[]
        for v in depvars
            v_raw = Symbolics.unwrap(s.discvars[v])
            v_c = _ConstSR(v_raw)
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
                taps = map(1:fi_nlap.padded_interp) do j_interp
                    iw = interp_w3d_c[j_outer, j_interp, _idxs[dim]]
                    tap_idx = interp_t3d_c[j_outer, j_interp, _idxs[dim]]
                    # grid_x_c[tap_idx]: tap_idx is raw SymReal (Int-typed), so this works
                    wrap(iw) * wrap(grid_x_c[tap_idx])
                end
                push!(interp_iv_rules, x => sum(taps))
            else
                haskey(indexmap, xv) || continue
                xv_dim = indexmap[xv]
                push!(interp_iv_rules, xv => wrap(grid_iv_cs[xv][_idxs[xv_dim] + bases_full[xv_dim]]))
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
function _build_nonlinlap_rules(ctx::ArrayOpContext, caches::StencilCaches,
                                 pde, var_rules)
    s                    = ctx.s
    depvars              = ctx.depvars
    derivweights         = ctx.derivweights
    indexmap             = ctx.indexmap
    _idxs                = ctx.idxs
    bases                = ctx.bases
    is_periodic          = ctx.is_periodic
    gl_vec               = ctx.gl_vec
    nonlinlap_cache      = caches.nonlinlap
    full_nonlinlap_cache = caches.full_nonlinlap

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
                _nonlinlap_full_template(ctx, expr_sym, u, x, nsi, fi_nlap)
            else
                _nonlinlap_template(ctx, expr_sym, u, x, nsi)
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

