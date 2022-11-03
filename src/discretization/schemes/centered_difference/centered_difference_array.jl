########################################################################################
# Stencil interface
########################################################################################

function central_difference(D::DerivativeOperator, interior, s, b, jx, u, udisc)
    args = params(u, s)
    ranges = map(x -> axes(s.grid[x])[1], args)
    interior = map(x -> interior[x], args)
    is = map(x -> s.index_syms[x], args)

    j, x = jx
    lenx = length(s, x)

    if b isa Val{false}
        lowerops = map(interior[x][1]:D.boundary_point_count) do iboundary
            lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
        end

        upperops = map((lenx-D.boundary_point_count+1):interior[x][end]) do iboundary
            upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
        end
    else
        lowerops = []
        upperops = []
    end
    boundaryoppairs = vcat(lowerops, upperops)

    interiorop = interior_deriv(D, udisc, half_range(D.stencil_length), jx, is, interior, b)

    return NullBG_Arraymaker(ranges, vcat((interior...) => interiorop, boundaryoppairs))
end

@inline function generate_cartesian_rules(interior, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, pmap, indexmap, terms)
    return reduce(vcat,
                  [reduce(vcat,
                          [[(Differential(x)^d)(u) =>
                              central_difference(derivweights.map[Differential(x)^d],
                                                 interior, s, pmap.map[operation(u)][x],
                                                 (x2i(s, u, x), x), u, central_ufunc)
                             for d in (let orders = derivweights.orders[x]
                                           orders[iseven.(orders)]
                                       end)]
                           for x in params(u, s)])
                    for u in depvars])
end
