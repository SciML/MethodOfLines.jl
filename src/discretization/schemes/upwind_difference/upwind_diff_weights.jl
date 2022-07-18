
"""
A helper function to compute the coefficients of a derivative operator including the boundary coefficients in the upwind scheme.
"""
function CompleteUpwindDifference(derivative_order::Int,
                                  approximation_order::Int, dx::T,
                                  offside::Int = 0) where {T <: Real, N}
    @assert offside>-1 "Number of offside points should be non-negative"

    stencil_length = derivative_order + approximation_order
    boundary_stencil_length = derivative_order + approximation_order
    low_boundary_point_count = offside
    high_boundary_point_count = stencil_length - 1 - offside

    # TODO: Clean up the implementation here so that it is more readable and easier to extend in the future
    dummy_x = (0.0 - offside):(stencil_length - 1.0 - offside)
    stencil_coefs = convert(SVector{stencil_length, T},
                            (1 / dx^derivative_order) *
                            calculate_weights(derivative_order, 0.0, dummy_x))
    low_boundary_x = 0.0:(boundary_stencil_length - 1)
    L_boundary_deriv_spots = 0.0:(low_boundary_point_count - 1)
    _low_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{
                                                                              boundary_stencil_length,
                                                                              T},
                                                                      (1 /
                                                                       dx^derivative_order) *
                                                                      calculate_weights(derivative_order,
                                                                                        oneunit(T) *
                                                                                        x0,
                                                                                        low_boundary_x))
                                                              for x0 in L_boundary_deriv_spots]
    low_boundary_coefs = convert(SVector{low_boundary_point_count}, _low_boundary_coefs)

    high_boundary_x = 0.0:-1.0:(-(boundary_stencil_length - 1.0))
    R_boundary_deriv_spots = 0.0:-1.0:(-(high_boundary_point_count - 1.0))
    _high_boundary_coefs = SVector{boundary_stencil_length, T}[convert(SVector{
                                                                               boundary_stencil_length,
                                                                               T},
                                                                       ((-1 / dx)^derivative_order) *
                                                                       calculate_weights(derivative_order,
                                                                                         oneunit(T) *
                                                                                         x0,
                                                                                         high_boundary_x))
                                                               for x0 in R_boundary_deriv_spots]
    high_boundary_coefs = convert(SVector{high_boundary_point_count},
                                  reverse(_high_boundary_coefs))

    coefficients = nothing

    DerivativeOperator{T, Nothing, true, T, typeof(stencil_coefs),
                       typeof(low_boundary_coefs), typeof(high_boundary_coefs), Nothing,
                       Nothing}(derivative_order, approximation_order, dx, 1,
                                stencil_length,
                                stencil_coefs,
                                boundary_stencil_length,
                                high_boundary_point_count,
                                low_boundary_coefs,
                                high_boundary_coefs, offside, coefficients, nothing)
end

function CompleteUpwindDifference(derivative_order::Int,
                                  approximation_order::Int, x::AbstractVector{T},
                                  offside::Int = 0) where {T <: Real, N}
    @assert offside>-1 "Number of offside points should be non-negative"

    stencil_length = derivative_order + approximation_order
    @assert offside<=stencil_length - 1 "Number of offside points should be less than or equal to the stencil length"
    boundary_stencil_length = derivative_order + approximation_order
    low_boundary_point_count = offside
    high_boundary_point_count = stencil_length - 1 - offside

    dx = [x[i + 1] - x[i] for i in 1:(length(x) - 1)]

    low_boundary_x = @view(x[1:boundary_stencil_length])
    high_boundary_x = @view(x[(end - boundary_stencil_length + 1):end])

    L_boundary_deriv_spots = x[1:low_boundary_point_count]
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    #deriv_spots             = (-div(stencil_length,2)+1) : -1  # unused

    R_boundary_deriv_spots = x[(end - high_boundary_point_count + 1):end]

    stencil_coefs = [convert(SVector{stencil_length, eltype(x)},
                             calculate_weights(derivative_order, x[i],
                                               @view(x[(i - offside):(i + stencil_length - 1 - offside)])))
                     for i in (low_boundary_point_count + 1):(length(x) - high_boundary_point_count)]
    # For each boundary point, for each tappoint in the half offset central difference stencil, we need to calculate the coefficients for the stencil.

    low_boundary_coefs = [convert(SVector{boundary_stencil_length, eltype(x)},
                                  calculate_weights(derivative_order, offset,
                                                    low_boundary_x))
                          for offset in L_boundary_deriv_spots]

    high_boundary_coefs = [convert(SVector{boundary_stencil_length, eltype(x)},
                                   calculate_weights(derivative_order, offset,
                                                     high_boundary_x))
                           for offset in R_boundary_deriv_spots]

    # _high_boundary_coefs    = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, reverse(right_boundary_x))) for x0 in R_boundary_deriv_spots]

    offside = 0
    coefficients = nothing

    DerivativeOperator{eltype(x), Nothing, false, typeof(dx), typeof(stencil_coefs),
                       typeof(low_boundary_coefs), typeof(high_boundary_coefs),
                       typeof(coefficients), Nothing}(derivative_order, approximation_order,
                                                      dx, 1, stencil_length,
                                                      stencil_coefs,
                                                      boundary_stencil_length,
                                                      high_boundary_point_count,
                                                      low_boundary_coefs,
                                                      high_boundary_coefs, offside,
                                                      coefficients, nothing)
end
