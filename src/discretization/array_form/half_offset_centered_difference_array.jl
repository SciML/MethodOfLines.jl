########################################################################################
# Stencil interface
########################################################################################

function half_offset_centered_difference(D::DerivativeOperator, interior, s, b, jx, u, udisc, len)
    args = params(u, s)
    interior = get_interior(u, s, interior)
    is = get_is(u, s)

    j, x = jx
    offset = len == 0 ? 0 : -1
    lenx = len == 0 ? length(s, x) : len

    if b isa Val{false}
        lowerops = map(interior[j][1]:D.boundary_point_count) do iboundary
            lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
        end

        upperops = map((lenx-D.boundary_point_count+1):(interior[j][end]+offset)) do iboundary
            upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
        end
    else
        lowerops = []
        upperops = []
    end

    interiorop = interior_deriv(D, udisc,
                                (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2)),
                                j, is, interior, b)
    boundaryoppairs = vcat(lowerops, upperops)

    return Construct_ArrayMaker(interior, vcat(Tuple(interior) => interiorop, boundaryoppairs))
end
