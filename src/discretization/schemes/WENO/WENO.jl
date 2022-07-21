
"""
Implements the WENO scheme of Jiang and Shu.

Specified in https://repository.library.brown.edu/studio/item/bdr:297524/PDF/ (Page 8-9)

Implementation inspired by https://github.com/ranocha/HyperbolicDiffEq.jl/blob/84c2d882e0c8956457c7d662bf7f18e3c27cfa3d/src/finite_volumes/weno_jiang_shu.jl
"""
@inline function weno(II::CartesianIndex, s::DiscreteSpace, b, jx, u, dx::Number)
    j, x = jx
    ε = 1e-6

    I1 = unitindex(ndims(u, s), j)

    udisc = s.discvars[u]

    u_m2 = udisc[wrapperiodic(II - 2I1, s, b, u, jx)]
    u_m1 = udisc[wrapperiodic(II - I1, s, b, u, jx)]
    u_0 = udisc[II]
    u_p1 = udisc[wrapperiodic(II + I1, s, b, u, jx)]
    u_p2 = udisc[wrapperiodic(II + 2I1, s, b, u, jx)]

    γ1 = 1 / 10
    γ2 = 3 / 5
    γ3 = 3 / 10

    β1 = 13 * (u_m2 - 2 * u_m1 + u_0)^2 / 12 + (u_m2 - 4 * u_m1 + 3 * u_0)^2 / 4
    β2 = 13 * (u_m1 - 2 * u_0 + u_p1)^2 / 12 + (u_m1 - u_p1)^2 / 4
    β3 = 13 * (u_0 - 2 * u_p1 + u_p2)^2 / 12 + (3 * u_0 - 4 * u_p1 + u_p2)^2 / 4

    ω1 = γ1 / (ε + β1)^2
    ω2 = γ2 / (ε + β2)^2
    ω3 = γ3 / (ε + β3)^2

    w_denom = ω1 + ω2 + ω3
    wp1 = ω1 / w_denom
    wp2 = ω2 / w_denom
    wp3 = ω3 / w_denom

    wm1 = wp3
    wm2 = wp2
    wm3 = wp1

    # * Note: H. Ranchoa has these reversed, check here first for sign error
    hp1 = (2u_m2 - 7u_m1 + 11u_0) / 6
    hp2 = -(u_m1 + 5u_0 + 2u_p1) / 6
    hp3 = (2u_0 + 5u_p1 / 6 - u_p2) / 6

    hm1 = (2u_0 + 5u_m1 / 6 - u_m2) / 6
    hm2 = -(u_p1 + 5u_0 + 2u_m1) / 6
    hm3 = (2u_p2 - 7u_p1 + 11u_0) / 6

    hp = wp1 * hp1 + wp2 * hp2 + wp3 * hp3
    hm = wm1 * hm1 + wm2 * hm2 + wm3 * hm3

    return (hp - hm) / dx
end

"""
This is a catch all ruleset, as such it does not use @rule. Any first order derivative may be adequately approximated by a WENO scheme.
"""
@inline function generate_WENO_rules(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, pmap, indexmap, terms)
    return reduce(vcat, [[(Differential(x))(u) => weno(Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], (x2i(s, u, x), x), u, s.dxs[x]) for x in params(u, s)] for u in depvars])
end
