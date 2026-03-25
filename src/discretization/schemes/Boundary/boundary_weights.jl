# boundary_weights.jl

# 🛡️ LEFT BOUNDARY (Forward Difference)

function get_nonuniform_weights_1st_deriv_left(h1_in::Real, h2_in::Real)
    h1, h2 = promote(h1_in, h2_in) 
    T = typeof(h1)
    
    @assert h1 > zero(T) && h2 > zero(T) "Grid spacings (h1, h2) must be strictly positive!"
    den1 = h1 * (h1 + h2)
    den2 = h1 * h2
    den3 = h2 * (h1 + h2)

    c0 = -(2*one(T)*h1 + h2) / den1
    c1 = (h1 + h2) / den2
    c2 = -h1 / den3
    return (c0, c1, c2)
end

function get_nonuniform_weights_2nd_deriv_left(h1_in::Real, h2_in::Real)
    h1, h2 = promote(h1_in, h2_in)
    T = typeof(h1)
    
    @assert h1 > zero(T) && h2 > zero(T) "Grid spacings (h1, h2) must be strictly positive!"
    den1 = h1 * (h1 + h2)
    den2 = h1 * h2
    den3 = h2 * (h1 + h2)

    c0 = 2*one(T) / den1
    c1 = -2*one(T) / den2
    c2 = 2*one(T) / den3
    return (c0, c1, c2)
end

# RIGHT BOUNDARY (Backward Difference)

function get_nonuniform_weights_1st_deriv_right(h1_in::Real, h2_in::Real)
    c0, c1, c2 = get_nonuniform_weights_1st_deriv_left(h1_in, h2_in)
    return (-c0, -c1, -c2)
end

function get_nonuniform_weights_2nd_deriv_right(h1_in::Real, h2_in::Real)
    return get_nonuniform_weights_2nd_deriv_left(h1_in, h2_in)
end