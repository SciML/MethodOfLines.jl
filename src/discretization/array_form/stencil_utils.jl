function lower_boundary_deriv(D, iboundary, j, args, is, interior)
    weights = D.low_boundary_coefs[iboundary]
    taps = 1:D.boundary_stencil_length
    BoundaryDerivArrayOp(weights, taps, j, args, is, interior)
end

function upper_boundary_deriv(D, iboundary, j, args, is, interior, lenx)
    weights = D.low_boundary_coefs[lenx-iboundary+1]
    taps = (lenx+D.boundary_stencil_length+1):lenx
    BoundaryDerivArrayOp(weights, taps, j, args, is, interior)
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

function BoundaryDerivArrayOp(weights, taps, j, args, is, interior)
    # * I Possibly needs updating
    I = map(1:length(args)) do i
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
    return ArrayOp(Array{symtype(expr), length(output_idx)}, output_idx, expr, +, nothing, ranges)
end

function InteriorDerivArrayOp(weights, taps, jx, args, output_idx, interior)
    # * I Possibly needs updating
    j, x = jx
    I = map(1:length(args)) do i
        if i == j
            wrapperiodic.(taps, [s], [b], [u], [jx])
        else
            is[i]
        end
    end
    expr = dot(weights, udisc[I])
    symindices = setdiff(1:length(args), j)
    ranges = Dict(output_idx .=> interior) # hope this doesn't check bounds eagerly
    return ArrayOp(Array{symtype(expr), length(output_idx)}, output_idx, expr, +, nothing, ranges)
end
