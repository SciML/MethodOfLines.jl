"""
Non-uniform WENO-5 mathematical engine.
Zero-allocation ruleset compatible with Float32/64, ForwardDiff.Dual, and Symbolics.Num.
"""

"""
    compute_beta_nonuniform(u1, u2, u3, x1, x2, x3, xL, xR)

Analytic Jiang-Shu smoothness indicator on a non-uniform 3-point sub-stencil, obtained
via Simpson quadrature over `[xL, xR]` (CSE-applied; zero allocation).
"""
@inline function compute_beta_nonuniform(
        u1::U, u2::U, u3::U, x1::X, x2::X, x3::X, xL::X, xR::X,
    ) where {U <: Real, X <: Real}
    dx_face = xR - xL
    V = (x1 - x2) * (x1 - x3) * (x2 - x3)
    inv_V2 = one(X) / (V * V)

    D2 = X(2) * (u1 * (x3 - x2) + u2 * (x1 - x3) + u3 * (x2 - x1))
    dx_face_sq = dx_face * dx_face
    dx_face_qu = dx_face_sq * dx_face_sq
    D2_sq = D2 * D2
    term_d2 = dx_face_qu * D2_sq

    x1_sq = x1 * x1
    x2_sq = x2 * x2
    x3_sq = x3 * x3

    A = -u1 * x2_sq + X(2) * u1 * x2 * xR + u1 * x3_sq - X(2) * u1 * x3 * xR +
        u2 * x1_sq - X(2) * u2 * x1 * xR - u2 * x3_sq + X(2) * u2 * x3 * xR -
        u3 * x1_sq + X(2) * u3 * x1 * xR + u3 * x2_sq - X(2) * u3 * x2 * xR
    B = u1 * x2_sq - u1 * x2 * xL - u1 * x2 * xR - u1 * x3_sq + u1 * x3 * xL + u1 * x3 * xR -
        u2 * x1_sq + u2 * x1 * xL + u2 * x1 * xR + u2 * x3_sq - u2 * x3 * xL - u2 * x3 * xR +
        u3 * x1_sq - u3 * x1 * xL - u3 * x1 * xR - u3 * x2_sq + u3 * x2 * xL + u3 * x2 * xR
    C = u1 * x2_sq - X(2) * u1 * x2 * xL - u1 * x3_sq + X(2) * u1 * x3 * xL -
        u2 * x1_sq + X(2) * u2 * x1 * xL + u2 * x3_sq - X(2) * u2 * x3 * xL +
        u3 * x1_sq - X(2) * u3 * x1 * xL - u3 * x2_sq + X(2) * u3 * x2 * xL

    A_sq = A * A
    B_sq = B * B
    C_sq = C * C
    term_d1 = (dx_face_sq / X(6)) * (A_sq + X(4) * B_sq + C_sq)
    return (term_d2 + term_d1) * inv_V2
end

@inline function _lagrange_weight(x1::T, x2::T, x3::T, k::Int, x_eval::T) where {T <: Real}
    if k == 1
        return ((x_eval - x2) * (x_eval - x3)) / ((x1 - x2) * (x1 - x3))
    elseif k == 2
        return ((x_eval - x1) * (x_eval - x3)) / ((x2 - x1) * (x2 - x3))
    else
        return ((x_eval - x1) * (x_eval - x2)) / ((x3 - x1) * (x3 - x2))
    end
end

"""
    ideal_weights_lagrange(x1, x2, x3, x_eval)

Intra-stencil Lagrange weights `(w1, w2, w3)` at `x_eval`; partition of unity.
"""
@inline function ideal_weights_lagrange(
        x1::T, x2::T, x3::T, x_eval::T,
    ) where {T <: Real}
    w1 = _lagrange_weight(x1, x2, x3, 1, x_eval)
    w2 = _lagrange_weight(x1, x2, x3, 2, x_eval)
    w3 = _lagrange_weight(x1, x2, x3, 3, x_eval)
    return w1, w2, w3
end

@inline function _stencil_taylor_moment3(x1::T, x2::T, x3::T, x_eval::T) where {T <: Real}
    w1, w2, w3 = ideal_weights_lagrange(x1, x2, x3, x_eval)
    inv6 = one(T) / T(6)
    d1 = x1 - x_eval
    d2 = x2 - x_eval
    d3 = x3 - x_eval
    d1_sq = d1 * d1
    d2_sq = d2 * d2
    d3_sq = d3 * d3
    return inv6 * (w1 * d1_sq * d1 + w2 * d2_sq * d2 + w3 * d3_sq * d3)
end

@inline function _stencil_taylor_moment4(x1::T, x2::T, x3::T, x_eval::T) where {T <: Real}
    w1, w2, w3 = ideal_weights_lagrange(x1, x2, x3, x_eval)
    inv24 = one(T) / T(24)
    d1 = x1 - x_eval
    d2 = x2 - x_eval
    d3 = x3 - x_eval
    d1_sq = d1 * d1
    d2_sq = d2 * d2
    d3_sq = d3 * d3
    return inv24 * (w1 * d1_sq * d1_sq + w2 * d2_sq * d2_sq + w3 * d3_sq * d3_sq)
end

"""
    ideal_weno_linear_weights(x1_S1, x2_S1, x3_S1, x1_S2, x2_S2, x3_S2, x1_S3, x2_S3, x3_S3, x_f)

Linear ideal sub-stencil weights `(d1, d2, d3)` for Lagrange-based quadratic reconstructions
at face `x_f`.

Note: These ideal weights are derived using the Lagrange basis. On a uniform grid, they
reduce to `(1/16, 5/8, 5/16)` rather than the Jiang-Shu ENO constants `(0.1, 0.6, 0.3)`,
but yield the exact same 5th-order flux as `weno_f_uniform`.

Enforced constraints (Taylor moment matching on Lagrange sub-stencil operators):

    d1 + d2 + d3 = 1,
    Σ d_k c_{k,3} = 0,
    Σ d_k c_{k,4} = 0,

where `c_{k,n}` is `_stencil_taylor_moment` on sub-stencil k.
"""
@inline function ideal_weno_linear_weights(
        x1_S1::T, x2_S1::T, x3_S1::T,
        x1_S2::T, x2_S2::T, x3_S2::T,
        x1_S3::T, x2_S3::T, x3_S3::T,
        x_f::T,
    ) where {T <: Real}
    c13 = _stencil_taylor_moment3(x1_S1, x2_S1, x3_S1, x_f)
    c23 = _stencil_taylor_moment3(x1_S2, x2_S2, x3_S2, x_f)
    c33 = _stencil_taylor_moment3(x1_S3, x2_S3, x3_S3, x_f)
    c14 = _stencil_taylor_moment4(x1_S1, x2_S1, x3_S1, x_f)
    c24 = _stencil_taylor_moment4(x1_S2, x2_S2, x3_S2, x_f)
    c34 = _stencil_taylor_moment4(x1_S3, x2_S3, x3_S3, x_f)

    detM = c23 * c34 - c33 * c24 - c13 * c34 + c33 * c14 + c13 * c24 - c23 * c14
    inv_detM = one(T) / detM

    d1 = (c23 * c34 - c33 * c24) * inv_detM
    d2 = (c33 * c14 - c13 * c34) * inv_detM
    d3 = (c13 * c24 - c23 * c14) * inv_detM
    return d1, d2, d3
end

@inline function _positive_part(d::T) where {T <: Real}
    return IfElse.ifelse(d >= zero(T), d, -d)
end

"""
    shi_hu_shu_weights(d1, d2, d3, b1, b2, b3, epsilon)

Shi-Hu-Shu (2002) splitting for nonlinear WENO weights; partition of unity enforced.
"""
@inline function shi_hu_shu_weights(
        d1::D, d2::D, d3::D,
        b1::B, b2::B, b3::B,
        epsilon::Real,
    ) where {D <: Real, B <: Real}
    T = promote_type(D, B)
    θ = T(3)
    ε = T(epsilon)
    half = T(1) / T(2)

    dp1 = half * (d1 + θ * _positive_part(d1))
    dp2 = half * (d2 + θ * _positive_part(d2))
    dp3 = half * (d3 + θ * _positive_part(d3))

    dm1 = dp1 - d1
    dm2 = dp2 - d2
    dm3 = dp3 - d3

    σp = dp1 + dp2 + dp3
    σm = dm1 + dm2 + dm3

    inv1 = inv(ε + b1)
    inv2 = inv(ε + b2)
    inv3 = inv(ε + b3)
    αp1 = dp1 * inv1 * inv1
    αp2 = dp2 * inv2 * inv2
    αp3 = dp3 * inv3 * inv3
    αp_sum = αp1 + αp2 + αp3
    ωp1 = αp1 / αp_sum
    ωp2 = αp2 / αp_sum
    ωp3 = αp3 / αp_sum

    αm1 = dm1 * inv1 * inv1
    αm2 = dm2 * inv2 * inv2
    αm3 = dm3 * inv3 * inv3
    αm_sum = αm1 + αm2 + αm3
    ωm1 = αm1 / αm_sum
    ωm2 = αm2 / αm_sum
    ωm3 = αm3 / αm_sum

    ω1 = σp * ωp1 - σm * ωm1
    ω2 = σp * ωp2 - σm * ωm2
    ω3 = σp * ωp3 - σm * ωm3
    return ω1, ω2, ω3
end

"""
    weno_f_nonuniform(u, p, t, x, dx)

Jiang-Shu WENO-5 semidiscrete flux derivative at the central node on a non-uniform grid.
Returns `(hp - hm) / dx_face` for direct insertion into MOL finite-difference rules.
"""
Base.@propagate_inbounds @inline function weno_f_nonuniform(
        u, p, t, x, _dx::AbstractVector,
    )
    # Note: Grid geometry is explicitly computed from coordinates (x) to prevent floating-point mismatch.
    # _dx is required by the FunctionalScheme signature but intentionally ignored.
    ε = p[1]

    u_m2 = u[1]
    u_m1 = u[2]
    u_0 = u[3]
    u_p1 = u[4]
    u_p2 = u[5]

    x_face_l = (x[2] + x[3]) / 2
    x_face_r = (x[3] + x[4]) / 2
    dx_face = (x[4] - x[2]) / 2

    x1_S1, x2_S1, x3_S1 = x[3], x[4], x[5]
    x1_S2, x2_S2, x3_S2 = x[2], x[3], x[4]
    x1_S3, x2_S3, x3_S3 = x[1], x[2], x[3]

    w1_S1_l, w2_S1_l, w3_S1_l = ideal_weights_lagrange(x1_S1, x2_S1, x3_S1, x_face_l)
    w1_S2_l, w2_S2_l, w3_S2_l = ideal_weights_lagrange(x1_S2, x2_S2, x3_S2, x_face_l)
    w1_S3_l, w2_S3_l, w3_S3_l = ideal_weights_lagrange(x1_S3, x2_S3, x3_S3, x_face_l)
    dm1, dm2, dm3 = ideal_weno_linear_weights(
        x1_S1, x2_S1, x3_S1, x1_S2, x2_S2, x3_S2, x1_S3, x2_S3, x3_S3, x_face_l,
    )

    w1_S1_r, w2_S1_r, w3_S1_r = ideal_weights_lagrange(x1_S1, x2_S1, x3_S1, x_face_r)
    w1_S2_r, w2_S2_r, w3_S2_r = ideal_weights_lagrange(x1_S2, x2_S2, x3_S2, x_face_r)
    w1_S3_r, w2_S3_r, w3_S3_r = ideal_weights_lagrange(x1_S3, x2_S3, x3_S3, x_face_r)
    dp1, dp2, dp3 = ideal_weno_linear_weights(
        x1_S1, x2_S1, x3_S1, x1_S2, x2_S2, x3_S2, x1_S3, x2_S3, x3_S3, x_face_r,
    )

    β1 = compute_beta_nonuniform(
        u_0, u_p1, u_p2, x1_S1, x2_S1, x3_S1, x_face_l, x_face_r,
    )
    β2 = compute_beta_nonuniform(
        u_m1, u_0, u_p1, x1_S2, x2_S2, x3_S2, x_face_l, x_face_r,
    )
    β3 = compute_beta_nonuniform(
        u_m2, u_m1, u_0, x1_S3, x2_S3, x3_S3, x_face_l, x_face_r,
    )

    wm1, wm2, wm3 = shi_hu_shu_weights(dm1, dm2, dm3, β1, β2, β3, ε)
    wp1, wp2, wp3 = shi_hu_shu_weights(dp1, dp2, dp3, β1, β2, β3, ε)

    hm1 = w1_S1_l * u_0 + w2_S1_l * u_p1 + w3_S1_l * u_p2
    hm2 = w1_S2_l * u_m1 + w2_S2_l * u_0 + w3_S2_l * u_p1
    hm3 = w1_S3_l * u_m2 + w2_S3_l * u_m1 + w3_S3_l * u_0

    hp1 = w1_S1_r * u_0 + w2_S1_r * u_p1 + w3_S1_r * u_p2
    hp2 = w1_S2_r * u_m1 + w2_S2_r * u_0 + w3_S2_r * u_p1
    hp3 = w1_S3_r * u_m2 + w2_S3_r * u_m1 + w3_S3_r * u_0

    hm = wm1 * hm1 + wm2 * hm2 + wm3 * hm3
    hp = wp1 * hp1 + wp2 * hp2 + wp3 * hp3

    return (hp - hm) / dx_face
end
