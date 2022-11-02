########################################################################################
# Stencil interface
########################################################################################

function central_difference(D::DerivativeOperator{T,N,Wind,DX}, interior, s, b, jx, u, udisc) where {T,N,Wind,DX<:Number}
    args = params(u, s)
    ranges = map(x -> axes(s.grid[x])[1], params(u, s))
    interior = map(x -> interior[x], params(u, s))
    j, x = jx
    lenx = length(s, x)

    if b isa Val{false}
        lowerops = map(interior[x][1]:D.boundary_point_count) do iboundary
            (lower_boundary_deriv(D, iboundary, j, args, is, interior), iboundary)
        end

        upperops = map((lenx-D.boundary_point_count+1):interior[x][end]) do iboundary
            (upper_boundary_deriv(D, iboundary, j, args, is, interior, lenx), iboundary)
        end
    else
        lowerops = []
        upperops = []
    end

    interiorop = central_interior_deriv(D, jx, args, interior, udisc, b)
    boundaryoppairs = prepare_boundary_ops(vcat(lowerops, upperops))

    ArrayMaker{Real}((last.(ranges)...), vcat((ranges...) => 0,
        (interior...) => interiorop,
        boundaryoppairs))
end

function central_interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, jx, args, is, interior, b) where {T,N,Wind,DX<:Number}
    j, x = jx
    weights = D.stencil_coefs
    taps = half_range(D.stencil_length) .+ is[j]
    InteriorDerivArrayOp(weights, taps, jx, args, is, interior)
end

function central_interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, jx, args, is, interior, b) where {T,N,Wind,DX<:AbstractVector}
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    j, x = jx
    weights = D.stencil_coefs[is[j]-D.boundary_point_count]
    taps = half_range(D.stencil_length) .+ is[j]
    InteriorDerivArrayOp(weights, taps, jx, args, is, interior)
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
