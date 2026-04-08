# Finite difference weights for non-uniform grids at boundaries

"""
    _compensated_sum_neg(c1, c2, c3)

Implements the Kahan summation algorithm to compute the negative sum of three coefficients.
This ensures that the final set of coefficients (c0, c1, c2, c3) sums to exactly zero
at the floating-point level, maintaining numerical stability and mass conservation.
"""
@inline function _compensated_sum_neg(c1::T, c2::T, c3::T) where T
    s = c1
    err = zero(T) # Floating-point error compensation term

    # Process c2
    y2 = c2 - err
    t2 = s + y2
    err = (t2 - s) - y2
    s = t2

    # Process c3
    y3 = c3 - err
    t3 = s + y3
    err = (t3 - s) - y3
    s = t3

    return -s
end

# --- Left Boundary Stencils (Forward) ---

"""
    get_nonuniform_weights_1st_deriv_left_4pt(h1_in, h2_in, h3_in)

Computes the 4-point non-uniform finite difference weights for the first derivative 
at the left boundary. Returns a tuple containing the coefficients (c0, c1, c2, c3).
"""
@inline function get_nonuniform_weights_1st_deriv_left_4pt(h1_in::Real, h2_in::Real, h3_in::Real)
    @assert h1_in > 0 && h2_in > 0 && h3_in > 0 "Grid spacings must be positive."

    h1, h2, h3 = promote(h1_in, h2_in, h3_in)
    x1, x2, x3 = h1, h1 + h2, h1 + h2 + h3

    # Factored denominators to minimize floating-point reconstruction noise
    den1 = h1 * h2 * (h2 + h3)
    den2 = -h2 * h3 * (h1 + h2)
    den3 = h3 * (h2 + h3) * (h1 + h2 + h3)

    c1 = (x2 * x3) / den1
    c2 = (x1 * x3) / den2
    c3 = (x1 * x2) / den3

    # Enforce zero-sum property using compensated summation
    c0 = _compensated_sum_neg(c1, c2, c3)

    return (c0, c1, c2, c3)
end

"""
    get_nonuniform_weights_2nd_deriv_left_4pt(h1_in, h2_in, h3_in)

Computes the 4-point non-uniform finite difference weights for the second derivative 
at the left boundary. Returns a tuple containing the coefficients (c0, c1, c2, c3).
"""
@inline function get_nonuniform_weights_2nd_deriv_left_4pt(h1_in::Real, h2_in::Real, h3_in::Real)
    @assert h1_in > 0 && h2_in > 0 && h3_in > 0 "Grid spacings must be positive."

    h1, h2, h3 = promote(h1_in, h2_in, h3_in)
    x1, x2, x3 = h1, h1 + h2, h1 + h2 + h3

    den1 = h1 * h2 * (h2 + h3)
    den2 = -h2 * h3 * (h1 + h2)
    den3 = h3 * (h2 + h3) * (h1 + h2 + h3)

    c1 = -2 * (x2 + x3) / den1
    c2 = -2 * (x1 + x3) / den2
    c3 = -2 * (x1 + x2) / den3

    # Enforce zero-sum property using compensated summation
    c0 = _compensated_sum_neg(c1, c2, c3)

    return (c0, c1, c2, c3)
end

# --- Right Boundary Stencils (Backward) ---

"""
    get_nonuniform_weights_1st_deriv_right_4pt(h1_in, h2_in, h3_in)

Calculates weights for the first derivative at the right boundary using symmetry.
"""
@inline function get_nonuniform_weights_1st_deriv_right_4pt(h1_in::Real, h2_in::Real, h3_in::Real)
    c = get_nonuniform_weights_1st_deriv_left_4pt(h1_in, h2_in, h3_in)
    return (-c[1], -c[2], -c[3], -c[4])
end

"""
    get_nonuniform_weights_2nd_deriv_right_4pt(h1_in, h2_in, h3_in)

Calculates weights for the second derivative at the right boundary using symmetry.
"""
@inline function get_nonuniform_weights_2nd_deriv_right_4pt(h1_in::Real, h2_in::Real, h3_in::Real)
    return get_nonuniform_weights_2nd_deriv_left_4pt(h1_in, h2_in, h3_in)
end