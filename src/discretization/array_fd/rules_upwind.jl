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
function _build_upwind_rules(ctx::ArrayOpContext, caches::StencilCaches,
                              pde, bcmap, var_rules)
    s            = ctx.s
    depvars      = ctx.depvars
    derivweights = ctx.derivweights
    indexmap     = ctx.indexmap
    _idxs        = ctx.idxs
    bases        = ctx.bases
    is_periodic  = ctx.is_periodic
    gl_vec       = ctx.gl_vec
    upwind_cache = caches.upwind
    full_interior_upwind_cache = caches.full_upwind
    # Helper: build stencil expression for a given variable, dimension, offsets, weights.
    # For non-uniform grids, weight_matrix is a stencil_length × num_interior Matrix
    # and bpc is the offside (= low_boundary_point_count) used to align weight matrix
    # column indexing: stencil_coefs[j] corresponds to grid index (j + bpc).
    function _upwind_stencil_expr(u, x, offsets, weights;
                                   weight_matrix=nothing, bpc=0, D_op=nothing)
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = _ConstSR(u_raw)
        u_spatial = ivs(u, s)
        taps = [_tap_expr(ctx, u_c, u_spatial, x, off) for off in offsets]
        if weight_matrix === nothing
            # Uniform: constant weights
            return sym_dot(weights, taps)
        else
            # Non-uniform: index into weight matrix by interior point index
            dim = indexmap[x]
            if is_periodic[dim]
                # Periodic non-uniform: extended N-column weight matrix
                ext_wmat = _build_periodic_wmat(D_op, collect(s.grid[x]), offsets)
                wmat_c = _ConstSR(ext_wmat)
                point_idx = _idxs[dim] + bases[dim]
            else
                wmat_c = _ConstSR(weight_matrix)
                point_idx = _idxs[dim] + bases[dim] - bpc
            end
            return sum(1:length(offsets)) do k
                Symbolics.wrap(wmat_c[k, point_idx]) * taps[k]
            end
        end
    end

    # Helper: build full-interior stencil expression using weight+offset matrices.
    function _upwind_full_interior_expr(u, x, wmat, omat, padded_len)
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = _ConstSR(u_raw)
        u_spatial = ivs(u, s)
        dim = indexmap[x]
        wm_c = _ConstSR(wmat)
        om_c = _ConstSR(omat)
        return sum(1:padded_len) do j
            w = Symbolics.wrap(wm_c[j, _idxs[dim]])
            off_val = Symbolics.wrap(om_c[j, _idxs[dim]])
            w * _tap_expr(ctx, u_c, u_spatial, x, off_val)
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
                        fiusi.padded_neg
                    )
                    pos_expr = _upwind_full_interior_expr(
                        u, x, fiusi.pos_weight_matrix, fiusi.pos_offset_matrix,
                        fiusi.padded_pos
                    )
                else
                    neg_expr = _upwind_stencil_expr(
                        u, x, usi.neg.offsets, usi.neg.D_op.stencil_coefs;
                        weight_matrix=usi.neg.weight_matrix,
                        bpc=usi.neg.D_op.offside,
                        D_op=usi.neg.D_op
                    )
                    pos_expr = _upwind_stencil_expr(
                        u, x, usi.pos.offsets, usi.pos.D_op.stencil_coefs;
                        weight_matrix=usi.pos.weight_matrix,
                        bpc=usi.pos.D_op.offside,
                        D_op=usi.pos.D_op
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
                        fiusi.padded_pos
                    )
                else
                    pos_expr = _upwind_stencil_expr(
                        u, x, usi.pos.offsets, usi.pos.D_op.stencil_coefs;
                        weight_matrix=usi.pos.weight_matrix,
                        bpc=usi.pos.D_op.offside,
                        D_op=usi.pos.D_op
                    )
                end
                push!(fallback_rules, (Differential(x)^d)(u) => pos_expr)
            end
        end
    end

    return wind_rules, fallback_rules
end

