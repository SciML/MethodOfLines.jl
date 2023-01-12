function lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
    weights = D.low_boundary_coefs[iboundary]
    taps = 1:D.boundary_stencil_length
    prepare_boundary_op((BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior),
            iboundary), interior, j)
end

function upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
    weights = D.high_boundary_coefs[lenx - iboundary + 1]
    taps = (lenx-D.boundary_stencil_length+1):lenx
    prepare_boundary_op((BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior),
            iboundary), interior, j)
end

function integral_op_pair(weights, taps, udisc, j, is, interior, i)
    prepare_boundary_ops((IntegralArrayOp(weights, taps, i, udisc, j, is, interior),
            i), interior, j)
end

function prepare_boundary_op(boundaryop, interior, j)
    function maketuple(i)
        out = map(1:length(interior)) do k
            k == j ? i : interior[k]
        end
        return Tuple(out)
    end
    (op, iboundary) = boundaryop
    return maketuple(iboundary) => op
end

function prepare_boundary_ops(boundaryops, interior, j)
    function maketuple(i)
        out = map(1:length(interior)) do k
            k == j ? i : interior[k]
        end
        return Tuple(out)
    end
    return map(boundaryops) do (op, iboundary)
        maketuple(iboundary) => op
    end
end

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, udisc, s, offsets, j, is, interior, bs) where {T,N,Wind,DX<:Number}
    weights = D.stencil_coefs
    taps = offsets .+ is[j]
    InteriorDerivArrayOp(weights, taps, udisc, s, j, is, interior, bs)
end

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, udisc, s, offsets, j, is, interior, bs) where {T,N,Wind,DX<:AbstractVector}
    @assert length(bs) == 0 "Interface boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    weights = D.stencil_coefs[is[j]-D.boundary_point_count]
    taps = offsets .+ is[j]
    InteriorDerivArrayOp(weights, taps, udisc, s, j, is, interior, bs)
end

function BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
    # * I Possibly needs updating
    Is = map(taps) do tap
        map(1:ndims(udisc)) do i
            if i == j
                tap
            else
                is[i]
            end
        end
    end
    expr = dot(weights, map(I -> udisc[I...], Is))

    symindices = setdiff(1:ndims(udisc), [j])
    output_idx = Tuple(is[symindices])
    return FillArrayOp(expr, output_idx, interior[symindices])
end

function IntegralArrayOp(weights, taps, k, udisc, j, is, interior, iswd = false)
    # * I Possibly needs updating
    if k == 1
        expr = Num(0)
    else
        Is = map(taps) do tap
            map(1:ndims(udisc)) do i
                if i == j
                    tap
                else
                    is[i]
                end
            end
        end
        expr = dot(weights, map(I -> udisc[I...], Is))
    end
    # Don't reduce dimension of interior if its a whole domain integral
    symindices = setdiff(1:ndims(udisc), [j])
    output_idx = Tuple(is[symindices])
    if iswd
        symindices = 1:ndims(udisc)
    end
    return FillArrayOp(expr, output_idx, interior[symindices]) .+ IntegralArrayOp(weights, taps, k-1, udisc, j, is, interior)
end

function InteriorDerivArrayOp(weights, taps, udisc, s, j, output_idx, interior, bs)
    # * I Possibly needs updating
    Is = map(taps) do tap
        _is = map(1:ndims(udisc)) do i
            if i == j
                tap
            else
                output_idx[i]
            end
        end
        CartesianIndex(wrap.(_is)...)
    end
    # Wrap interfaces
    Is = map(I -> bwrap(I, bs, s, j), Is)
    expr = dot(weights, map(I -> udisc[I], Is))
    return FillArrayOp(expr, output_idx, interior)
end

function FillArrayOp(expr, output_idx, interior)
    ranges = Dict(output_idx .=> interior) # hope this doesn't check bounds eagerly
    return ArrayOp(Array{symtype(expr),length(output_idx)},
                   output_idx, expr, +, nothing, ranges)
end

NullBG_ArrayMaker(ranges, ops) = ArrayMaker{Real}(Tuple(map(r -> r[end] - r[1] + 1, ranges)),
                                                  vcat(Tuple(ranges) => 0, ops))
Construct_ArrayMaker(ranges,
                     ops) = ArrayMaker{Real}(Tuple(map(r -> r[end] - r[1] + 1, ranges)), ops)

FillArrayMaker(expr, is,
               ranges, interior) = NullBG_ArrayMaker(ranges,
                                                     [FillArrayOp(expr, is, interior)])

ArrayMakerWrap(udisc, ranges) = Arraymaker{Real}(Tuple(map(r -> r[end] - r[1] + 1, ranges)),
                                                 [Tuple(ranges) => udisc])

#####

get_interior(u, s, interior) = map(x -> interior[x], params(u, s))
get_ranges(u, s) = map(x -> first(axes(s.grid[x])), params(u, s))
get_is(u, s) = map(x -> s.index_syms[x], params(u, s))
