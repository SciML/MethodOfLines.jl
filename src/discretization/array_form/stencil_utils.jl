function lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
    weights = D.low_boundary_coefs[iboundary]
    taps = 1:D.boundary_stencil_length
    BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
    weights = D.low_boundary_coefs[lenx-iboundary+1]
    taps = (lenx+D.boundary_stencil_length+1):lenx
    BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function prepare_boundary_ops(boundaryops, interior, j)
    function maketuple(i)
        out = map(1:length(interior)) do k
            k == j ? i : interior[k]
        end
        return (out...)
    end
    return map(boundaryops) do (op, iboundary)
        maketuple(iboundary) => op
    end
end

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, u, udisc, offsets, jx, is, interior, b) where {T,N,Wind,DX<:Number}
    j, x = jx
    weights = D.stencil_coefs
    taps = offsets .+ is[j]
    InteriorDerivArrayOp(weights, taps, u, udisc, jx, is, interior)
end

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, u, udisc, offsets, jx, is, interior, b) where {T,N,Wind,DX<:AbstractVector}
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    j, x = jx
    weights = D.stencil_coefs[is[j]-D.boundary_point_count]
    taps = offsets .+ is[j]
    InteriorDerivArrayOp(weights, taps, u, udisc, jx, is, interior)
end

function BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
    # * I Possibly needs updating
    I = map(1:ndims(udisc)) do i
        if i == j
            taps
        else
            is[i]
        end
    end
    expr = dot(weights, udisc[I...])
    symindices = setdiff(1:length(args), j)
    output_idx = (is[symindices]...)
    ranges = Dict(output_idx .=> interior[symindices])
    return ArrayOp(Array{symtype(expr),length(output_idx)}, output_idx, expr, +, nothing, ranges)
end

function InteriorDerivArrayOp(weights, taps, u, udisc, jx, output_idx, interior)
    # * I Possibly needs updating
    j, x = jx
    I = map(1:ndims(udisc)) do i
        if i == j
            wrapperiodic.(taps, [s], [b], [u], [jx])
        else
            output_idx[i]
        end
    end
    expr = dot(weights, udisc[I])
    ranges = Dict(output_idx .=> interior) # hope this doesn't check bounds eagerly
    return ArrayOp(Array{symtype(expr),length(output_idx)}, output_idx, expr, +, nothing, ranges)
end

function FillArrayOp(expr, output_idx, interior)
    ranges = Dict(output_idx .=> interior) # hope this doesn't check bounds eagerly
    return ArrayOp(Array{symtype(expr),length(output_idx)}, output_idx, expr, +, nothing, ranges)
end

NullBG_ArrayMaker(ranges, ops) = ArrayMaker{Real}(last.(ranges...), vcat((ranges...) => 0, ops) )

FillArrayMaker(expr, is, ranges, interior) = NullBG_ArrayMaker(ranges, [FillArrayOp(expr, is, interior)])

ArrayMakerWrap(udisc, ranges) = Arraymaker{Real}(last.(ranges), [(ranges...) => udisc])
