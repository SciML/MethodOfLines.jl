"""
Performs a mixed centered difference in `x` centered at index `II` of `u`
ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes
"""
function mixed_central_difference((Dx, Dy), II, s, (xbs, ybs), (jx, ky), u, ufunc)
    j, x = jx
    k, y = ky
    xweights, xItap = central_difference_weights_and_stencil(Dx, II, s, xbs, jx, u)
    yweights, yItap = central_difference_weights_and_stencil(Dy, II, s, ybs, ky, u)
    # TODO: Fix interface bcs

    out = sum(zip(xweights, xItap)) do (wx, xI)
        sum(zip(yweights, yItap)) do (wy, yI)
            xoffset = xI - II
            yoffset = yI - II
            I = II + xoffset + yoffset
            wx * wy * ufunc(u, I, x)
        end
    end

    return out
end

@inline function generate_mixed_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms
    )
    central_ufunc(u, I, x) = s.discvars[u][I]
    return reduce(
        safe_vcat,
        [
            reduce(
                    safe_vcat,
                    [
                        [
                            (
                                Differential(x) *
                                Differential(y)
                            )(u) => unit_correct(unit_correct(
                                mixed_central_difference(
                                    (derivweights.map[Differential(x)], derivweights.map[Differential(y)]),
                                    Idx(II, s, u, indexmap),
                                    s,
                                    (
                                        filter_interfaces(bcmap[operation(u)][x]),
                                        filter_interfaces(bcmap[operation(u)][y]),
                                    ),
                                    ((x2i(s, u, x), x), (x2i(s, u, y), y)),
                                    u,
                                    central_ufunc
                                ),
                                x, 1, derivweights.unit_map
                            ), y, 1, derivweights.unit_map) for y in remove(ivs(u, s), unwrap(x))
                        ] for x in ivs(u, s)
                    ],
                    init = []
                ) for u in depvars
        ],
        init = []
    )
end
