########################################################################################
# Stencil interface
########################################################################################

function half_offset_centered_difference(D::DerivativeOperator, interior, s, b, jx, u, udisc, len)
    args = params(u, s)
    ranges = map(x -> axes(s.grid[x])[1], args)
    interior = map(x -> interior[x], args)
    is = map(x -> s.index_syms[x], args)

    j, x = jx
    offset = len == 0 ? 0 : -1
    lenx = len == 0 ? length(s, x) : len

    if b isa Val{false}
        lowerops = map(interior[x][1]:D.boundary_point_count) do iboundary
            lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
        end

        upperops = map((lenx-D.boundary_point_count+1):(interior[x][end]+offset)) do iboundary
            upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
        end
    else
        lowerops = []
        upperops = []
    end

    interiorop = interior_deriv(D, udisc,
                                (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2)),
                                jx, is, interior, b)
    boundaryoppairs = vcat(lowerops, upperops)

    return Construct_ArrayMaker(interior, vcat((interior...) => interiorop, boundaryoppairs))
end
