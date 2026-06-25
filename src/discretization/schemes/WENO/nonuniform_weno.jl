# Node-centered Lagrange WENO-5 on non-uniform grids. The three 3-point sub-stencil derivative
# estimates p_k'(x_i) are combined with closed-form derivative-ideal weights d_k under a
# Shi-Hu-Shu positive/negative split. References: Fornberg (1988); Jiang & Shu (1996);
# Shi, Hu & Shu (2002).

# Fornberg (1988) finite-difference weights for a 3-point stencil, returning the m = 0, 1, 2
# derivative operators. Unrolled over SVectors for stack allocation; index map ν = 1,2,3 -> 0,1,2.
@inline function _fornberg3_weights(α::NTuple{3, T}, xt::T) where {T}
    α0, α1, α2 = α

    # n = 0 base case.
    n1m0 = one(T); n1m1 = zero(T); n1m2 = zero(T)
    c1 = one(T)

    # n = 1: introduce α1.
    c2 = α1 - α0
    r1 = c1 / c2
    tA = α1 - xt
    a1m0 = (tA * n1m0) / c2
    a1m1 = (tA * n1m1 - n1m0) / c2
    a1m2 = (tA * n1m2 - 2 * n1m1) / c2
    sB = α0 - xt
    a2m0 = r1 * (-(sB) * n1m0)
    a2m1 = r1 * (n1m0 - sB * n1m1)
    a2m2 = r1 * (2 * n1m1 - sB * n1m2)
    c1 = c2

    n1m0 = a1m0; n1m1 = a1m1; n1m2 = a1m2
    n2m0 = a2m0; n2m1 = a2m1; n2m2 = a2m2

    # n = 2: introduce α2.
    c2 = (α2 - α0) * (α2 - α1)
    r2 = c1 / c2
    c3a = α2 - α0
    c3b = α2 - α1
    tA2 = α2 - xt
    b1m0 = (tA2 * n1m0) / c3a
    b1m1 = (tA2 * n1m1 - n1m0) / c3a
    b1m2 = (tA2 * n1m2 - 2 * n1m1) / c3a
    b2m0 = (tA2 * n2m0) / c3b
    b2m1 = (tA2 * n2m1 - n2m0) / c3b
    b2m2 = (tA2 * n2m2 - 2 * n2m1) / c3b
    sB2 = α1 - xt
    b3m0 = r2 * (-(sB2) * n2m0)
    b3m1 = r2 * (n2m0 - sB2 * n2m1)
    b3m2 = r2 * (2 * n2m1 - sB2 * n2m2)

    m0 = SVector{3, T}(b1m0, b2m0, b3m0)   # values
    m1 = SVector{3, T}(b1m1, b2m1, b3m1)   # first derivative
    m2 = SVector{3, T}(b1m2, b2m2, b3m2)   # second derivative
    return m0, m1, m2
end

@inline _dot3(w::SVector{3}, a, b, c) = w[1] * a + w[2] * b + w[3] * c

# Smoothness indicator β_k (Jiang & Shu 1996) via Simpson's rule over cell i, exact for the
# degree-2 sub-stencil integrand, together with the sub-stencil derivative estimate r_k = p_k'(x_i).
@inline function _substencil_beta_r(
        α::NTuple{3, T}, ua, ub, uc, xi, xL, xM, xph, Δx
    ) where {T}
    _, m1i, _ = _fornberg3_weights(α, xi)
    _, m1L, _ = _fornberg3_weights(α, xL)
    _, m1M, m2M = _fornberg3_weights(α, xM)
    _, m1R, _ = _fornberg3_weights(α, xph)

    r = _dot3(m1i, ua, ub, uc)            # p_k'(x_i)
    pL = _dot3(m1L, ua, ub, uc)           # p_k'(x_{i-1/2})
    pM = _dot3(m1M, ua, ub, uc)           # p_k'(x_M)
    pR = _dot3(m1R, ua, ub, uc)           # p_k'(x_{i+1/2})
    pp = _dot3(m2M, ua, ub, uc)           # p_k''  (constant on a 3-point stencil)

    I1 = (Δx / 6) * (pL^2 + 4 * pM^2 + pR^2)   # ∫ (p')^2 dx  (Simpson, exact)
    I2 = Δx * pp^2                              # ∫ (p'')^2 dx
    val = Δx * I1 + Δx^3 * I2
    β = IfElse.ifelse(val < zero(val), zero(val), val)
    return β, r
end

# Zero-allocation core. Geometry and weights are formed in Tx = eltype(x); promotion against
# eltype(u) (Symbolics.Num, ForwardDiff.Dual, Float32) occurs at the dot products.
@inline function _weno_f_nonuniform_core(u, p, x)
    Tx = eltype(x)
    ε = p[1]
    θ = Tx(3)
    half = Tx(1) / 2

    @inbounds begin
        x1 = Tx(x[1]); x2 = Tx(x[2]); x3 = Tx(x[3]); x4 = Tx(x[4]); x5 = Tx(x[5])
        u1 = u[1]; u2 = u[2]; u3 = u[3]; u4 = u[4]; u5 = u[5]
    end

    # Reconstruction target is the center node x_i = x3. The cell [x_{i-1/2}, x_{i+1/2}] and its
    # Simpson nodes are used only by the smoothness indicators.
    xi = x3
    xph = (x3 + x4) / 2          # x_{i+1/2}
    xmh = (x2 + x3) / 2          # x_{i-1/2}
    Δx = xph - xmh               # cell width Δx_i
    xL = xmh
    xM = (xL + xph) / 2

    αS0 = (x1, x2, x3)
    αS1 = (x2, x3, x4)
    αS2 = (x3, x4, x5)

    β0, r0 = _substencil_beta_r(αS0, u1, u2, u3, xi, xL, xM, xph, Δx)
    β1, r1 = _substencil_beta_r(αS1, u2, u3, u4, xi, xL, xM, xph, Δx)
    β2, r2 = _substencil_beta_r(αS2, u3, u4, u5, xi, xL, xM, xph, Δx)

    # Closed-form derivative-ideal weights: the unique d_k with Σ_k d_k p_k'(x_i) = P'_5(x_i),
    # reducing to (1/6, 2/3, 1/6) on a uniform grid. Σ_k d_k = 1.
    d0 = ((x3 - x4) * (x3 - x5)) / ((x1 - x4) * (x1 - x5))
    d2 = ((x3 - x1) * (x3 - x2)) / ((x5 - x1) * (x5 - x2))
    d1 = one(Tx) - d0 - d2

    # Shi-Hu-Shu (2002) positive/negative split (θ = 3). In this node-centered direct-derivative
    # topology the d_k remain non-negative, so the split is presently a defensive mechanism and a
    # placeholder for future Taylor-series flux expansions that may introduce negative weights. The
    # positive and negative branches are kept separate and recombined as σp·Rp − σm·Rm.
    dp0 = half * (d0 + θ * abs(d0)); dp1 = half * (d1 + θ * abs(d1)); dp2 = half * (d2 + θ * abs(d2))
    dm0 = dp0 - d0; dm1 = dp1 - d1; dm2 = dp2 - d2
    σp = dp0 + dp1 + dp2
    σm = dm0 + dm1 + dm2

    ap0 = (dp0 / σp) / (ε + β0)^2; ap1 = (dp1 / σp) / (ε + β1)^2; ap2 = (dp2 / σp) / (ε + β2)^2
    sp = ap0 + ap1 + ap2
    ωp0 = ap0 / sp; ωp1 = ap1 / sp; ωp2 = ap2 / sp

    am0 = (dm0 / σm) / (ε + β0)^2; am1 = (dm1 / σm) / (ε + β1)^2; am2 = (dm2 / σm) / (ε + β2)^2
    sm = am0 + am1 + am2
    ωm0 = am0 / sm; ωm1 = am1 / sm; ωm2 = am2 / sm

    Rp = ωp0 * r0 + ωp1 * r1 + ωp2 * r2
    Rm = ωm0 * r0 + ωm1 * r1 + ωm2 * r2

    return σp * Rp - σm * Rm
end

"""
    weno_f_nonuniform(u, p, t, x, dx)

Node-centered Lagrange WENO-5 reconstruction of a first spatial derivative on a non-uniform grid.

Given the length-5 interior stencil values `u` and the corresponding absolute node coordinates
`x`, this implementation reconstructs the direct spatial derivative at the center node `x[3]` to
achieve 4th-order accuracy on non-uniform grids. It is a non-conservative formulation. The
smoothness parameter `ε = p[1]` regularizes the nonlinear weights; `t` and `dx` are accepted to
satisfy the `FunctionalScheme{5,0}` contract but are unused, as all geometry is taken from `x`.

The coordinates `x` must be strictly monotonically increasing and distinct (`Δx_i > 0`): the
formulation divides by node differences and is undefined on degenerate grids.

References: Fornberg (1988); Jiang & Shu (1996); Shi, Hu & Shu (2002).
"""
@inline function weno_f_nonuniform(u, p, t, x, dx::AbstractVector)
    @boundscheck (length(u) == 5 && length(x) == 5) ||
        throw(ArgumentError("weno_f_nonuniform requires a length-5 interior stencil (got length(u)=$(length(u)), length(x)=$(length(x)))."))
    return _weno_f_nonuniform_core(u, p, x)
end

# Scalar dx method: uniform-stepsize fallback required by the FunctionalScheme contract.
@inline function weno_f_nonuniform(u, p, t, x, dx::Number)
    @boundscheck (length(u) == 5 && length(x) == 5) ||
        throw(ArgumentError("weno_f_nonuniform requires a length-5 interior stencil (got length(u)=$(length(u)), length(x)=$(length(x)))."))
    return _weno_f_nonuniform_core(u, p, x)
end
