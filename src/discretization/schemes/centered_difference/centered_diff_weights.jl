"""
A helper function to compute the coefficients of a derivative operator including the boundary coefficients in the centered scheme.
"""
function CompleteCenteredDifference(derivative_order::Int,
                                    approximation_order::Int, dx::T) where {T <: Real, N}
    @assert approximation_order>1 "approximation_order must be greater than 1."
    stencil_length = derivative_order + approximation_order - 1 +
                     (derivative_order + approximation_order) % 2
    boundary_stencil_length = derivative_order + approximation_order
    dummy_x = (-div(stencil_length, 2)):div(stencil_length, 2)
    left_boundary_x = 0:(boundary_stencil_length - 1)
    right_boundary_x = reverse((-boundary_stencil_length + 1):0)

    boundary_point_count = div(stencil_length, 2) # -1 due to the ghost point
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    #deriv_spots             = (-div(stencil_length,2)+1) : -1  # unused
    L_boundary_deriv_spots = left_boundary_x[1:div(stencil_length, 2)]

    stencil_coefs = convert(SVector{stencil_length, T},
                            (1 / dx^derivative_order) *
                            calculate_weights(derivative_order, zero(T), dummy_x))
    _low_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{
                                                                              boundary_stencil_length,
                                                                              T},
                                                                      (1 /
                                                                       dx^derivative_order) *
                                                                      calculate_weights(derivative_order,
                                                                                        oneunit(T) *
                                                                                        x0,
                                                                                        left_boundary_x))
                                                              for x0 in L_boundary_deriv_spots]
    low_boundary_coefs = convert(SVector{boundary_point_count}, _low_boundary_coefs)

    # _high_boundary_coefs    = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, reverse(right_boundary_x))) for x0 in R_boundary_deriv_spots]
    high_boundary_coefs = convert(SVector{boundary_point_count},
                                  [reverse(v) * (-1)^derivative_order
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

function CompleteCenteredDifference(derivative_order::Int,
                                    approximation_order::Int,
                                    x::AbstractVector{T}) where {T <: Real, N}
    stencil_length = derivative_order + approximation_order - 1 +
                     (derivative_order + approximation_order) % 2
    boundary_stencil_length = derivative_order + approximation_order
    boundary_point_count = endpoint = div(stencil_length, 2)
    len = length(x)
    dx = [x[i + 1] - x[i] for i in 1:(length(x) - 1)]
    interior_x = (boundary_point_count + 1):(len - boundary_point_count)
    low_boundary_x = [zero(T); cumsum(dx[1:(boundary_stencil_length - 1)])]
    high_boundary_x = cumsum(dx[(end - boundary_stencil_length + 1):end])
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    deriv_spots = (-div(stencil_length, 2) + 1):-1

    stencil_coefs = [convert(SVector{stencil_length, T},
                             calculate_weights(derivative_order, x[i],
                                               x[(i - endpoint):(i + endpoint)]))
                     for i in interior_x]
    low_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{
                                                                             boundary_stencil_length,
                                                                             T},
                                                                     calculate_weights(derivative_order,
                                                                                       low_boundary_x[i + 1],
                                                                                       low_boundary_x))
                                                             for i in 0:(boundary_point_count - 1)]

    high_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{
                                                                              boundary_stencil_length,
                                                                              T},
                                                                      calculate_weights(derivative_order,
                                                                                        high_boundary_x[end - i],
                                                                                        high_boundary_x))
                                                              for i in 0:(boundary_point_count - 1)]

    offside = 0
    coefficients = nothing

    DerivativeOperator{eltype(x), Nothing, false, typeof(dx), typeof(stencil_coefs),
                       typeof(low_boundary_coefs), typeof(high_boundary_coefs),
                       typeof(coefficients),
                       Nothing}(derivative_order, approximation_order, dx,
                                len, stencil_length,
                                stencil_coefs,
                                boundary_stencil_length,
                                boundary_point_count,
                                low_boundary_coefs,
                                high_boundary_coefs, offside, coefficients, nothing)
end
