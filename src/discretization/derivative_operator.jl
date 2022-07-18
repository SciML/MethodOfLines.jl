struct DerivativeOperator{T <: Real, N, Wind, T2, S1, S2, S3, T3, F}
    derivative_order::Int
    approximation_order::Int
    dx::T2
    len::Int
    stencil_length::Int
    stencil_coefs::S1
    boundary_stencil_length::Int
    boundary_point_count::Int
    low_boundary_coefs::S2
    high_boundary_coefs::S3
    offside::Int
    coefficients::T3
    coeff_func::F
end
