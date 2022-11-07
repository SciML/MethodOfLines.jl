########################################################################################
# Stencil interface
########################################################################################

function _upwind_difference(D, interior, is, s,
                            b, jx, u, udisc, ispositive)
    args = params(u, s)
    is = map(x -> s.index_syms[x], args)

    j, x = jx
    lenx = length(s, x)
    if ispositive
        upperops = []
        if b isa Val{false}
            lowerops = map(interior[x][1]:D.offside) do iboundary
                lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
            end
        else
            upperops = []
        end
        interiorop = interior_deriv(D, -D.stencil_length+1:0, jx, args, interior, b)
    else
        if b isa Val{false}
            upperops = map((lenx-D.boundary_point_count+1):interior[x][end]) do iboundary
                upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
            end
        else
            lowerops = []
        end
        interiorop = interior_deriv(D, udisc, 0:D.stencil_length-1, jx, is, args, interior, b)
    end
    boundaryoppairs = vcat(lowerops, upperops)

    Construct_ArrayMaker(interior, vcat((interior...) => interiorop, boundaryoppairs))
end

"""
# upwind_difference
Generate a finite difference expression in `u` using the upwind difference at point `II::CartesianIndex`
in the direction of `x`
"""
function upwind_difference(d::Int, interior, is, s::DiscreteSpace, b, derivweights,
                           jx, u, udisc, ispositive)
    j, x = jx
    # return if this is an ODE
    ndims(u, s) == 0 && return Num(0)
    D = if !ispositive
        derivweights.windmap[1][Differential(x)^d]
    else
        derivweights.windmap[2][Differential(x)^d]
    end
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    return _upwind_difference(D, interior, is, s, b, jx, u, udisc, ispositive)
end

function upwind_difference(expr, d::Int, interior, s::DiscreteSpace, b,
                           depvars, derivweights, (j, x), u, udisc, indexmap)
    # TODO: Allow derivatives in expr
    interior = map(x -> interior[x], params(u, s))
    is = map(x -> s.index_syms[x], params(u, s))
    expr = substitute(expr, valmaps(s, u, depvars, Idx(CartesianIndex(is...),
                      s, depvar(u, s), indexmap), indexmap))

    exprarr = FillArrayMaker(expr, is, interior, interior)
    IfElse.ifelse.(exprarr .> 0,
        exprarr .* upwind_difference(d, interior, is, s, b, derivweights, (j, x), u, udisc, true),
        exprarr .* upwind_difference(d, interior, is, s, b, derivweights, (j, x), u, udisc, false))
end

@inline function generate_winding_rules(interior, s::DiscreteSpace, depvars,
                                        derivweights::DifferentialDiscretizer, pmap,
                                        indexmap, terms)
    # for all independent variables and dependant variables
    rules = vcat(#Catch multiplication
        reduce(vcat,
               [reduce(vcat,
                       [[@rule *(~~a, $(Differential(x)^d)(u), ~~b) =>
                                 upwind_difference(*(~a..., ~b...), d, interior, s,
                                                   pmap.map[operation(u)][x], depvars,
                                                   derivweights, (x2i(s, u, x), x), u,
                                                   s.discvars[u], indexmap)
                          for d in (let orders = derivweights.orders[x]
                                       orders[isodd.(orders)]
                                   end)]
                         for x in params(u, s)])
                for u in depvars]),

        #Catch division and multiplication, see issue #1
        reduce(vcat,
               [reduce(vcat,
                       [[@rule /(*(~~a, $(Differential(x)^d)(u), ~~b), ~c) =>
                                 upwind_difference(*(~a..., ~b...) / ~c, d, interior, s,
                                                   pmap.map[operation(u)][x], depvars,
                                                   derivweights, (x2i(s, u, x), x), u,
                                                   s.discvars[u], indexmap)
                          for d in (let orders = derivweights.orders[x]
                                       orders[isodd.(orders)]
                                   end)]
                         for x in params(u, s)])
                for u in depvars])
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

    return vcat(wind_rules, vec(mapreduce(vcat, depvars) do u
        mapreduce(vcat, params(u, s)) do x
            j = x2i(s, u, x)
            is = get_is(u, s)
            interior = get_interior(u, s, interior)
            let orders = derivweights.orders[x]
                oddorders = orders[isodd.(orders)]
                # for all odd orders
                if length(oddorders) > 0
                    map(oddorders) do d
                        (Differential(x)^d)(u) =>
                          upwind_difference(d, interior, is, s, pmap.map[operation(u)][x],
                                            derivweights, (j, x), u, s.discvars[u], true)
                    end
                else
                    []
                end
            end
        end
    end))
end
