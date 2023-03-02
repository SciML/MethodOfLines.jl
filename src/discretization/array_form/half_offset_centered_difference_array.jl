########################################################################################
# Stencil interface
########################################################################################

function half_offset_centered_difference(D::DerivativeOperator, interior, s, bs, jx, u, udisc, len, isx = false)
    interior = get_interior(u, s, interior)
    ranges = get_ranges(u, s)
    is = get_is(u, s)

    j, x = jx
    offset = len == 0 ? 0 : -1
    lenx = len == 0 ? length(s, x) : len

    haslower, hasupper = haslowerupper(bs, x)

    lowerops = []
    upperops = []

    if !haslower
        lowerops = map(interior[j][1]:D.boundary_point_count) do iboundary
            lower_boundary_deriv(D, udisc, iboundary, j, is, interior)
        end
    end
    if !hasupper
        upperops = map((lenx-D.boundary_point_count+1):interior[j][end]+offset) do iboundary
            upper_boundary_deriv(D, udisc, iboundary, j, is, interior, lenx)
        end
    end
    if !isx
        interiorop = interior_deriv(D, udisc, s,
                                    (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2)),
                                    j, is, interior, bs)
    else
        interiorop = interior_deriv(D, OrderedIndexArray(udisc, j, ndims(u, s)), s,
                                    (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2)),
                                    j, is, interior, bs, isx)
    end
    boundaryoppairs = vcat(lowerops, upperops)

    return NullBG_ArrayMaker(ranges, vcat(Tuple(interior) => interiorop, boundaryoppairs))[interior...]
end
