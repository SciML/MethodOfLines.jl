"""
A helper function to compute the coefficients of a derivative operator including the boundary coefficients in the half offset centered scheme. See table 2 in https://web.njit.edu/~jiang/math712/fornberg.pdf
"""
function CompleteHalfCenteredDifference(derivative_order::Int,
                                        approximation_order::Int,
                                        dx::T) where {T <: Real}
    @assert approximation_order>1 "approximation_order must be greater than 1."
    centered_stencil_length = approximation_order + 2 * Int(floor(derivative_order / 2)) +
                              (approximation_order % 2)
    boundary_stencil_length = derivative_order + approximation_order
    endpoint = div(centered_stencil_length, 2)
    dummy_x = (1 - endpoint):endpoint
    left_boundary_x = 1:(boundary_stencil_length)
    right_boundary_x = reverse((-boundary_stencil_length):-1)

    boundary_point_count = div(centered_stencil_length, 2) # -1 due to the ghost point
    # ? Is fornberg valid when taking an x0 outside of the stencil i.e at the boundary?
    xoffset = range(1.5, length = boundary_point_count, step = 1.0)

    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    #deriv_spots             = (-div(stencil_length,2)+1) : -1  # unused
    L_boundary_deriv_spots = xoffset[1:div(centered_stencil_length, 2)]

    stencil_coefs = convert(SVector{centered_stencil_length, T},
                            (1 / dx^derivative_order) *
                            calculate_weights(derivative_order, convert(T, 0.5), dummy_x))
    # For each boundary point, for each tappoint in the half offset central difference stencil, we need to calculate the coefficients for the stencil.

    _low_boundary_coefs = [convert(SVector{boundary_stencil_length, T},
                                   (1 / dx^derivative_order) *
                                   calculate_weights(derivative_order, offset,
                                                     left_boundary_x))
                           for offset in L_boundary_deriv_spots]
    low_boundary_coefs = convert(SVector{boundary_point_count}, _low_boundary_coefs)

    _high_boundary_coefs = [reverse(stencil) * (-1)^derivative_order
                            for stencil in low_boundary_coefs]
    high_boundary_coefs = convert(SVector{boundary_point_count}, _high_boundary_coefs)
    # _high_boundary_coefs    = SVector{boundary_stencil_length, T}[convert(SVector{boundary_stencil_length, T}, (1/dx^derivative_order) * calculate_weights(derivative_order, oneunit(T)*x0, reverse(right_boundary_x))) for x0 in R_boundary_deriv_spots]

    offside = 0
    coefficients = nothing

    DerivativeOperator{T, Nothing, false, T, typeof(stencil_coefs),
                       typeof(low_boundary_coefs), typeof(high_boundary_coefs),
                       typeof(coefficients), Nothing}(derivative_order, approximation_order,
                                                      dx, 1, centered_stencil_length,
                                                      stencil_coefs,
                                                      boundary_stencil_length,
                                                      boundary_point_count,
                                                      low_boundary_coefs,
                                                      high_boundary_coefs, offside,
                                                      coefficients, nothing)
end

function CompleteHalfCenteredDifference(derivative_order::Int,
                                        approximation_order::Int,
                                        x::T) where {T <: AbstractVector{<:Real}}
    @assert approximation_order>1 "approximation_order must be greater than 1."
    centered_stencil_length = approximation_order + 2 * Int(floor(derivative_order / 2)) +
                              (approximation_order % 2)
    boundary_stencil_length = derivative_order + approximation_order
    endpoint = div(centered_stencil_length, 2)
    hx = [(x[i] + x[i + 1]) / 2 for i in 1:(length(x) - 1)]
    dx = [x[i + 1] - x[i] for i in 1:(length(x) - 1)]

    low_boundary_x = @view(x[1:boundary_stencil_length])
    high_boundary_x = @view(x[(end - boundary_stencil_length + 1):end])

    boundary_point_count = div(centered_stencil_length, 2) # -1 due to the ghost point
    # ? Is fornberg valid when taking an x0 outside of the stencil i.e at the boundary?

    L_boundary_deriv_spots = hx[1:div(centered_stencil_length, 2)]
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    #deriv_spots             = (-div(stencil_length,2)+1) : -1  # unused

    R_boundary_deriv_spots = hx[length(hx):-1:(length(hx) - div(centered_stencil_length, 2) + 1)]

    stencil_coefs = [convert(SVector{centered_stencil_length, eltype(x)},
                             calculate_weights(derivative_order, hx[i],
                                               x[(i - endpoint + 1):(i + endpoint)]))
                     for i in (endpoint + 1):(length(x) - endpoint)]
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
                                                      dx, 1, centered_stencil_length,
                                                      stencil_coefs,
                                                      boundary_stencil_length,
                                                      boundary_point_count,
                                                      low_boundary_coefs,
                                                      high_boundary_coefs, offside,
                                                      coefficients, nothing)
end
