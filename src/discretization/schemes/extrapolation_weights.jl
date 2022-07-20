function BoundaryInterpolatorExtrapolator(approximation_order::Int, dx::T) where {T <: Real, N}
    @assert approximation_order>1 "approximation_order must be greater than 1."
    stencil_length = approximation_order - 1 + (approximation_order) % 2
    boundary_stencil_length = approximation_order + 1 # Add 1 since there is an extra 0 weight for the current index
    left_boundary_x = collect(0:(boundary_stencil_length - 1))
    right_boundary_x = collect(reverse((-boundary_stencil_length + 1):0))

    boundary_point_count = div(stencil_length, 2)
    L_boundary_deriv_spots = left_boundary_x[1:div(stencil_length, 2)]

    stencil_coefs = [] # Not defined on the interior
    _low_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T},
                                                                     insert(calculate_weights(derivative_order,
                                                                                        oneunit(T) *
                                                                                        x0,
                                                                                        remove(left_boundary_x, x0)), i, zero(T))
                                                              for (i,x0) in enumerate(L_boundary_deriv_spots)]
    low_boundary_coefs = convert(SVector{boundary_point_count}, _low_boundary_coefs)

    # _high_boundary_coefs    = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, reverse(right_boundary_x))) for x0 in R_boundary_deriv_spots]
    high_boundary_coefs = convert(SVector{boundary_point_count},
                                  [reverse(v)
                                   for v in _low_boundary_coefs])

    offside = 0

    coefficients = nothing

    DerivativeOperator{T, Nothing, false, T, typeof(stencil_coefs),
                       typeof(low_boundary_coefs), typeof(high_boundary_coefs),
                       typeof(coefficients),
                       Nothing}(derivative_order, approximation_order, dx, 1,
                                stencil_length,
                                stencil_coefs,
                                boundary_stencil_length,
                                boundary_point_count,
                                low_boundary_coefs,
                                high_boundary_coefs, offside, coefficients, nothing)
end
