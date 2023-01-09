########################################################################################
# Stencil interface
########################################################################################

function central_difference(D::DerivativeOperator, interior, s, bs, jx, u, udisc)
    args = params(u, s)
    interior = get_interior(u, s, interior)
    is = get_is(u, s)

    j, x = jx
    lenx = length(s, x)
    haslower, hasupper = haslowerupper(bs, x)

    lowerops = []
    upperops = []

    if !haslower
        lowerops = map(interior[j][1]:D.boundary_point_count) do iboundary
            lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
        end
    end
    if !hasupper
        upperops = map((lenx-D.boundary_point_count+1):interior[j][end]) do iboundary
            upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
        end
    end
    boundaryoppairs = vcat(lowerops, upperops)

    interiorop = interior_deriv(D, udisc, half_range(D.stencil_length), j, is, interior, bs)

    return Construct_ArrayMaker(interior, vcat(Tuple(interior) => interiorop, boundaryoppairs))
end

@inline function generate_cartesian_rules(interior, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, pmap, indexmap, terms)
    return reduce(vcat,
                  [reduce(vcat,
                          [[(Differential(x)^d)(u) =>
                              central_difference(derivweights.map[Differential(x)^d],
                                                 interior, s, pmap.map[operation(u)][x],
                                                 (x2i(s, u, x), x), u, s.discvars[u])
                             for d in (let orders = derivweights.orders[x]
                                           orders[iseven.(orders)]
                                       end)]
                           for x in params(u, s)])
                    for u in depvars])
end
