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

function integral_op_pair(dx, udisc, j, is, interior, i)
    prepare_boundary_op((IntegralArrayOp(dx, i, udisc, j, is, interior),
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

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, udisc, s, offsets, j, is, interior, bs, isx = false) where {T,N,Wind,DX<:Number}
    weights = D.stencil_coefs
    taps = offsets .+ is[j]
    InteriorDerivArrayOp(weights, taps, udisc, s, j, is, interior, bs, isx)
end

function interior_deriv(D::DerivativeOperator{T,N,Wind,DX}, udisc, s, offsets, j, is, interior, bs, isx = false) where {T,N,Wind,DX<:AbstractVector}
    @assert length(bs) == 0 "Interface boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    weights = D.stencil_coefs[is[j]-D.boundary_point_count]
    taps = offsets .+ is[j]
    InteriorDerivArrayOp(weights, taps, udisc, s, j, is, interior, bs, isx)
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

reduce_interior(interior, j) = first(interior[j]) == 1 ? [interior[1:j-1]..., 2:last(interior[j]), interior[j+1:end]...] : interior

function trapezium_sum(ranges, udisc, is, j)
    N = ndims(udisc)
    I1 = UnitIndex(N, j)
    I = CartesianIndex(wrap.(is)...)
    Im1 = I - I1

    interior = reduce_interior(interior, j)
    expr = (dx[Im1[j]]*udisc[Im1] + dx[I[j]]*udisc[I]) / 2

    return FillArrayMaker(expr, is, ranges, interior)
end


function IntegralArrayOp(k, udisc, j, is, interior, iswd = false)
    if iswd
        interior = [interior[1:j-1]..., 1:size(udisc, j), interior[j:end]...]
    end
    trapop = trapezium_sum(interior, udisc, is, j)
    op = sum(trapop[1:(k-first(interior[j])+1)], dims = j)

    return op
end

function InteriorDerivArrayOp(weights, taps, udisc, s, j, output_idx, interior, bs, isx = false)
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
    Is = map(I -> bwrap(I, bs, s, j, isx), Is)

    expr = dot(weights, map(I -> udisc[I], Is))
    return FillArrayOp(expr, output_idx, interior)
end

function FillArrayOp(expr, output_idx, interior)
    ranges = Dict(output_idx .=> interior) # hope this doesn't check bounds eagerly
    @show ranges
    return ArrayOp(Array{symtype(expr),length(output_idx)},
                   output_idx, expr, +, nothing, ranges)
end

NullBG_ArrayMaker(ranges, ops) = ArrayMaker{Real}(Tuple(map(r -> r[end] - r[1] + 1, ranges)),
                                                  vcat(Tuple(ranges) => 0, ops))
Construct_ArrayMaker(ranges,
                     ops) = ArrayMaker{Real}(Tuple(map(r -> r[end] - r[1] + 1, ranges)), ops)

FillArrayMaker(expr, is,
               ranges, interior) = NullBG_ArrayMaker(ranges,
                                                     [Tuple(interior) => FillArrayOp(expr, is, interior)])

ArrayMakerWrap(udisc, ranges) = Arraymaker{Real}(Tuple(map(r -> r[end] - r[1] + 1, ranges)),
                                                 [Tuple(ranges) => udisc])

#####

function get_interior(u, s, interior)
    map(params(u, s)) do x
        if haskey(interior, x)
            interior[x]
        else
            (1, length(s, x))
        end
    end
end

get_ranges(u, s) = map(x -> first(axes(s.grid[x])), params(u, s))
get_is(u, s) = map(x -> s.index_syms[x], params(u, s))
