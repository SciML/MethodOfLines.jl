function BoundaryInterpolatorExtrapolator(approximation_order::Int, dx::T) where {T <: Real}
    @assert approximation_order>1 "approximation_order must be greater than 1."
    stencil_length = approximation_order - 1 + (approximation_order) % 2
    boundary_stencil_length = approximation_order   # Add 1 since there is an extra 0 weight for the current index
    dummy_x = (-div(stencil_length, 2)):div(stencil_length, 2)

    left_boundary_x = collect(0:(boundary_stencil_length - 1))
    right_boundary_x = collect(reverse((-boundary_stencil_length + 1):0))

    boundary_point_count = div(stencil_length, 2) # This is the number of boundary points to use in the stencil: not optimal.
    L_boundary_deriv_spots = left_boundary_x[1:boundary_point_count]

    stencil_coefs = convert(SVector{stencil_length, T},
        insert(calculate_weights(0, zero(T), remove(dummy_x, 0)),
            findfirst(x -> isequal(0, x), dummy_x), zero(T))) # Not defined on the interior
    _low_boundary_coefs = SVector{boundary_stencil_length, T}[convert(
                                                                  SVector{
                                                                      boundary_stencil_length,
                                                                      T},
                                                                  insert(
                                                                      calculate_weights(0,
                                                                          oneunit(T) *
                                                                          x0,
                                                                          remove(
                                                                              left_boundary_x,
                                                                              x0)),
                                                                      i,
                                                                      zero(T)))
                                                              for (i, x0) in enumerate(L_boundary_deriv_spots)]
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
        Nothing}(0, approximation_order, dx, 1,
        stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs, offside, coefficients, nothing)
end

function BoundaryInterpolatorExtrapolator(approximation_order::Int,
        x::AbstractVector{T}) where {T <: Real}
    stencil_length = approximation_order - 1 +
                     (approximation_order) % 2

    midpoint = div(stencil_length, 2) + (stencil_length % 2)
    boundary_stencil_length = approximation_order
    boundary_point_count = endpoint = div(stencil_length, 2)
    len = length(x)
    dx = [x[i + 1] - x[i] for i in 1:(length(x) - 1)]
    interior_x = (endpoint + 1):(len - endpoint)
    low_boundary_x = [zero(T); cumsum(dx[1:(boundary_stencil_length - 1)])]
    high_boundary_x = cumsum(dx[(end - boundary_stencil_length + 1):end])
    # Because it's a N x (N+2) operator, the last stencil on the sides are the [b,0,x,x,x,x] stencils, not the [0,x,x,x,x,x] stencils, since we're never solving for the derivative at the boundary point.
    stencil_coefs = stencil_coefs = [convert(SVector{stencil_length, T},
                                         insert(
                                             calculate_weights(0, x[i],
                                                 remove(x[(i - endpoint):(i + endpoint)],
                                                     x[i])),
                                             midpoint,
                                             zero(T)))
                                     for i in interior_x]

    low_boundary_coefs = SVector{boundary_stencil_length, T}[convert(
                                                                 SVector{
                                                                     boundary_stencil_length,
                                                                     T},
                                                                 insert(
                                                                     calculate_weights(0,
                                                                         low_boundary_x[i + 1],
                                                                         remove(
                                                                             low_boundary_x,
                                                                             low_boundary_x[i + 1])),
                                                                     i + 1,
                                                                     zero(T)))
                                                             for i in 0:(boundary_point_count - 1)]

    high_boundary_coefs = SVector{boundary_stencil_length, T}[convert(
                                                                  SVector{
                                                                      boundary_stencil_length,
                                                                      T},
                                                                  insert(
                                                                      calculate_weights(0,
                                                                          high_boundary_x[end - i],
                                                                          remove(
                                                                              high_boundary_x,
                                                                              high_boundary_x[end - i])),
                                                                      length(high_boundary_x) -
                                                                      i,
                                                                      zero(T)))
                                                              for i in 0:(boundary_point_count - 1)]

    offside = 0
    coefficients = nothing

    DerivativeOperator{eltype(x), Nothing, false, typeof(dx), typeof(stencil_coefs),
        typeof(low_boundary_coefs), typeof(high_boundary_coefs),
        typeof(coefficients),
        Nothing}(0, approximation_order, dx,
        len, stencil_length,
        stencil_coefs,
        boundary_stencil_length,
        boundary_point_count,
        low_boundary_coefs,
        high_boundary_coefs, offside, coefficients, nothing)
end
