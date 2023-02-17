########################################################################################
# Stencil interface
########################################################################################

function _upwind_difference(D, ranges, interior, is, s,
                            bs, jx, u, udisc, ispositive)
    args = params(u, s)

    j, x = jx
    lenx = length(s, x)
    haslower, hasupper = haslowerupper(bs, x)

    upperops = []
    lowerops = []
    if ispositive
        if !haslower
            lowerops = map(interior[j][1]:D.offside) do iboundary
                lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
            end
        end
        interiorop = interior_deriv(D, udisc, s, -D.stencil_length+1:0, j, is, interior, b)
    else
        if !hasupper
            upperops = map((lenx-D.boundary_point_count+1):interior[j][end]) do iboundary
                upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
            end
        end
        interiorop = interior_deriv(D, udisc, s, 0:D.stencil_length-1, j, is, interior, b)
    end
    boundaryoppairs = safe_vcat(lowerops, upperops)

    NullBG_ArrayMaker(ranges, safe_vcat([Tuple(interior) => interiorop], boundaryoppairs))[interior...]
end

"""
# upwind_difference
Generate a finite difference expression in `u` using the upwind difference at point `II::CartesianIndex`
in the direction of `x`
"""
function upwind_difference(d::Int, ranges, interior, is, s::DiscreteSpace, b, derivweights,
                           jx, u, udisc, ispositive)
    j, x = jx
    # return if this is an ODE
    ndims(u, s) == 0 && return Fill(Num(0), ())
    D = if !ispositive
        derivweights.windmap[1][Differential(x)^d]
    else
        derivweights.windmap[2][Differential(x)^d]
    end
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    return _upwind_difference(D, ranges, interior, is, s, b, jx, u, udisc, ispositive)
end

function upwind_difference(expr, d::Int, interior, s::DiscreteSpace, b,
                           depvars, derivweights, (j, x), u, udisc, indexmap)
    # TODO: Allow derivatives in expr

    valrules = arrayvalmaps(s, u, depvars, interior)
    exprarr = broadcast_substitute(expr, valrules)
    ranges = get_ranges(u, s)

    IfElse.ifelse.(exprarr .> 0,
        exprarr .* upwind_difference(d, ranges, get_interior(u, s, interior), get_is(u, s), s, b, derivweights, (j, x), u, udisc, true),
        exprarr .* upwind_difference(d, ranges, get_interior(u, s, interior), get_is(u, s), s, b, derivweights, (j, x), u, udisc, false))
end

@inline function generate_winding_rules(interior, s::DiscreteSpace, depvars,
                                        derivweights::DifferentialDiscretizer, pmap,
                                        indexmap, terms, skip = [])
    # for all independent variables and dependant variables
    rules = safe_vcat(#Catch multiplication
        reduce(safe_vcat,
               [reduce(safe_vcat,
                       [[@rule *(~~a, $(Differential(x)^d)(u), ~~b) =>
                                 upwind_difference(*(~a..., ~b...), d, interior, s,
                                                   pmap.map[operation(u)][x], depvars,
                                                   derivweights, (x2i(s, u, x), x), u,
                                                   s.discvars[u], indexmap)
                          for d in (let orders = derivweights.orders[x]
                                       setdiff(orders[isodd.(orders)], skip)
                                   end)]
                         for x in params(u, s)], init = [])
                for u in depvars], init = []),

        #Catch division and multiplication, see issue #1
        reduce(safe_vcat,
               [reduce(safe_vcat,
                       [[@rule /(*(~~a, $(Differential(x)^d)(u), ~~b), ~c) =>
                                 upwind_difference(*(~a..., ~b...) / ~c, d, interior, s,
                                                   pmap.map[operation(u)][x], depvars,
                                                   derivweights, (x2i(s, u, x), x), u,
                                                   s.discvars[u], indexmap)
                          for d in (let orders = derivweights.orders[x]
                                        setdiff(orders[isodd.(orders)], skip)

                                   end)]
                         for x in params(u, s)], init = [])
                for u in depvars], init = [])
    )

    wind_rules = []

    # wind_exprs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(wind_rules, t => r(t))
            end
        end
    end

    return safe_vcat(wind_rules, vec(mapreduce(safe_vcat, depvars) do u
        mapreduce(safe_vcat, params(u, s), init = []) do x
            j = x2i(s, u, x)
            is = get_is(u, s)
            uinterior = get_interior(u, s, interior)
            uranges = get_ranges(u, s)
            let orders = derivweights.orders[x]
                oddorders = setdiff(orders[isodd.(orders)], skip)

                # for all odd orders
                if length(oddorders) > 0
                    map(oddorders) do d
                        (Differential(x)^d)(u) =>
                          upwind_difference(d, uranges, uinterior, is, s, pmap.map[operation(u)][x],
                                            derivweights, (j, x), u, s.discvars[u], true)
                    end
                else
                    []
                end
            end
        end
    end))
end
