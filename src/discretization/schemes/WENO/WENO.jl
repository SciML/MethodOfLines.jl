
"""
Implements the WENO scheme of Jiang and Shu.

Specified in https://repository.library.brown.edu/studio/item/bdr:297524/PDF/ (Page 8-9)

Implementation *heavily* inspired by https://github.com/ranocha/HyperbolicDiffEq.jl/blob/84c2d882e0c8956457c7d662bf7f18e3c27cfa3d/src/finite_volumes/weno_jiang_shu.jl by H. Ranocha.
"""
function weno(II::CartesianIndex, s::DiscreteSpace, wenoscheme::WENOScheme, bs, jx, u, dx::Number)
    j, x = jx
    ε = wenoscheme.epsilon

    I1 = unitindex(ndims(u, s), j)

    udisc = s.discvars[u]

    Im2 = bwrap(II - 2I1, bs, s, jx)
    Im1 = bwrap(II - I1, bs, s, jx)
    Ip1 = bwrap(II + I1, bs, s, jx)
    Ip2 = bwrap(II + 2I1, bs, s, jx)
    is = map(I -> I[j], [Im2, Im1, Ip1, Ip2])
    for i in is
        if i < 1
            return nothing
        elseif i > length(s, x)
            return nothing
        end
    end

    u_m2 = udisc[Im2]
    u_m1 = udisc[Im1]
    u_0 = udisc[II]
    u_p1 = udisc[Ip1]
    u_p2 = udisc[Ip2]

    γm1 = 1 / 10
    γm2 = 3 / 5
    γm3 = 3 / 10

    β1 = 13 * (u_0 - 2 * u_p1 + u_p2)^2 / 12 + (3 * u_0 - 4 * u_p1 + u_p2)^2 / 4
    β2 = 13 * (u_m1 - 2 * u_0 + u_p1)^2 / 12 + (u_m1 - u_p1)^2 / 4
    β3 = 13 * (u_m2 - 2 * u_m1 + u_0)^2 / 12 + (u_m2 - 4 * u_m1 + 3 * u_0)^2 / 4

    ωm1 = γm1 / (ε + β1)^2
    ωm2 = γm2 / (ε + β2)^2
    ωm3 = γm3 / (ε + β3)^2

    wm_denom = ωm1 + ωm2 + ωm3
    wm1 = ωm1 / wm_denom
    wm2 = ωm2 / wm_denom
    wm3 = ωm3 / wm_denom

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

    hm1 = (11 * u_0 - 7 * u_p1 + 2 * u_p2) / 6
    hm2 = (5 * u_0 - u_p1 + 2 * u_m1) / 6
    hm3 = (2 * u_0 + 5 * u_m1 - u_m2) / 6

    hp1 = (2 * u_0 + 5 * u_p1 - u_p2) / 6
    hp2 = (5 * u_0 + 2 * u_p1 - u_m1) / 6
    hp3 = (11 * u_0 - 7 * u_m1 + 2 * u_m2) / 6

    hp = wp1 * hp1 + wp2 * hp2 + wp3 * hp3
    hm = wm1 * hm1 + wm2 * hm2 + wm3 * hm3

    return recursive_unwrap((hp - hm) / dx)
end

function weno(II::CartesianIndex, s::DiscreteSpace, b, jx, u, dx::AbstractVector)
    @assert false "WENO scheme not implemented for nonuniform grids."
end

"""
This is a catch all ruleset, as such it does not use @rule.
"""
@inline function generate_WENO_rules(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    return reduce(safe_vcat, [[(Differential(x))(u) => weno(Idx(II, s, u, indexmap), s, derivweights.advection_scheme, filter_interfaces(bcmap[operation(u)][x]), (x2i(s, u, x), x), u, s.dxs[x]) for x in params(u, s)] for u in depvars], init = [])
end
