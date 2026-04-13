# --- WENO ArrayOp rules ----------------------------------------------------

"""
    _weno_template(ctx::ArrayOpContext, u, x, wsi)

Build the WENO5 (Jiang-Shu) formula as a symbolic ArrayOp expression.

Transcribes the `weno_f` function from `WENO.jl` using Const-wrapped array
taps instead of runtime values.  All coefficients are Float64 literals to
match the scalar path exactly (for `_equations_match` validation).
"""
function _weno_template(ctx::ArrayOpContext, u, x, wsi)
    s = ctx.s

    u_raw = Symbolics.unwrap(s.discvars[u])
    u_c = _ConstSR(u_raw)
    u_spatial = ivs(u, s)

    # Build the 5 shifted taps: u[i-2], u[i-1], u[i], u[i+1], u[i+2]
    taps = [_tap_expr(ctx, u_c, u_spatial, x, off) for off in wsi.offsets]
    # Map to weno_f naming: u_m2, u_m1, u_0, u_p1, u_p2
    u_m2, u_m1, u_0, u_p1, u_p2 = taps

    ε = wsi.epsilon
    dx = wsi.dx_val

    # --- Smoothness indicators (β values) --- same for both L and R sides
    β1 = 13 * (u_0 - 2 * u_p1 + u_p2)^2 / 12 + (3 * u_0 - 4 * u_p1 + u_p2)^2 / 4
    β2 = 13 * (u_m1 - 2 * u_0 + u_p1)^2 / 12 + (u_m1 - u_p1)^2 / 4
    β3 = 13 * (u_m2 - 2 * u_m1 + u_0)^2 / 12 + (u_m2 - 4 * u_m1 + 3 * u_0)^2 / 4

    # --- Left-biased (minus) weights and reconstructions ---
    γm1 = 1 / 10
    γm2 = 3 / 5
    γm3 = 3 / 10

    ωm1 = γm1 / (ε + β1)^2
    ωm2 = γm2 / (ε + β2)^2
    ωm3 = γm3 / (ε + β3)^2
    wm_denom = ωm1 + ωm2 + ωm3
    wm1 = ωm1 / wm_denom
    wm2 = ωm2 / wm_denom
    wm3 = ωm3 / wm_denom

    hm1 = (11 * u_0 - 7 * u_p1 + 2 * u_p2) / 6
    hm2 = (5 * u_0 - u_p1 + 2 * u_m1) / 6
    hm3 = (2 * u_0 + 5 * u_m1 - u_m2) / 6
    hm = wm1 * hm1 + wm2 * hm2 + wm3 * hm3

    # --- Right-biased (plus) weights and reconstructions ---
    γp1 = 3 / 10
    γp2 = 3 / 5
    γp3 = 1 / 10

    ωp1 = γp1 / (ε + β1)^2
    ωp2 = γp2 / (ε + β2)^2
    ωp3 = γp3 / (ε + β3)^2
    wp_denom = ωp1 + ωp2 + ωp3
    wp1 = ωp1 / wp_denom
    wp2 = ωp2 / wp_denom
    wp3 = ωp3 / wp_denom

    hp1 = (2 * u_0 + 5 * u_p1 - u_p2) / 6
    hp2 = (5 * u_0 + 2 * u_p1 - u_m1) / 6
    hp3 = (11 * u_0 - 7 * u_m1 + 2 * u_m2) / 6
    hp = wp1 * hp1 + wp2 * hp2 + wp3 * hp3

    return (hp - hm) / dx
end

"""
    _build_weno_rules(pde, s, depvars, weno_cache, indexmap, _idxs, bases, var_rules)

Build term-level substitution rules for WENO 1st-order derivatives.

Unlike upwind schemes, WENO internally handles both flux directions (left-
and right-biased reconstructions), so no IfElse wind switching is needed.
The result is the numerical derivative itself; coefficients simply scale it.

Returns a vector of `Pair{term => discretized_expr}`.
"""
function _build_weno_rules(ctx::ArrayOpContext, caches::StencilCaches,
                            pde, var_rules)
    s           = ctx.s
    depvars     = ctx.depvars
    indexmap    = ctx.indexmap
    _idxs       = ctx.idxs
    bases       = ctx.bases
    is_periodic = ctx.is_periodic
    gl_vec      = ctx.gl_vec
    weno_cache  = caches.weno

    terms = split_terms(pde, s.x̄)
    vr_dict = Dict(var_rules)
    weno_rules = Pair[]
    # Cache WENO template expressions to avoid recomputing in fallback loop
    weno_expr_cache = Dict{Tuple, Any}()

    for u in depvars
        for x in ivs(u, s)
            haskey(weno_cache, (u, x)) || continue
            wsi = weno_cache[(u, x)]

            weno_expr = _weno_template(ctx, u, x, wsi)
            weno_expr_cache[(u, x)] = weno_expr

            # Pattern 1: *(~~a, Dx(u), ~~b) — coefficient-multiplied 1st-order
            mul_rule = @rule *(
                ~~a,
                $(Differential(x))(u),
                ~~b
            ) => begin
                coeff = *(~a..., ~b...)
                coeff_subst = pde_substitute(coeff, vr_dict)
                coeff_subst * weno_expr
            end

            # Pattern 2: /(*(~~a, Dx(u), ~~b), ~c) — divided coefficient
            div_rule = @rule /(
                *(~~a, $(Differential(x))(u), ~~b),
                ~c
            ) => begin
                coeff = *(~a..., ~b...) / ~c
                coeff_subst = pde_substitute(coeff, vr_dict)
                coeff_subst * weno_expr
            end

            for t in terms
                matched = mul_rule(t)
                if matched !== nothing
                    push!(weno_rules, t => matched)
                    continue
                end
                matched = div_rule(t)
                if matched !== nothing
                    push!(weno_rules, t => matched)
                end
            end
        end
    end

    # Fallback: bare Dx(u) with no coefficient (reuse cached expressions)
    fallback_rules = Pair[]
    for u in depvars
        for x in ivs(u, s)
            haskey(weno_cache, (u, x)) || continue
            push!(fallback_rules, Differential(x)(u) => weno_expr_cache[(u, x)])
        end
    end

    return vcat(weno_rules, fallback_rules)
end
