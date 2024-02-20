"""
Get the weights and stencil for the inner half offset centered difference for the nonlinear laplacian for a given index and differentiating variable.

Does not discretize so that the weights can be used in a replacement rule
TODO: consider refactoring this to harmonize with centered difference

Each index corresponds to the weights and index for the derivative at index i+1/2
"""
function get_half_offset_weights_and_stencil(
        D::DerivativeOperator{T, N, Wind, DX}, II::Base.AbstractCartesianIndex,
        s::DiscreteSpace, bs, u, jx, len = 0) where {T, N, Wind, DX <: Number}
    j, x = jx
    len = len == 0 ? length(s, x) : len
    haslower, hasupper = haslowerupper(bs, x)
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count) & !haslower
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
    elseif (II[j] > (len - D.boundary_point_count)) & !hasupper
        weights = D.high_boundary_coefs[len - II[j]]
        offset = len - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):0]
    else
        weights = D.stencil_coefs
        Itap = [II + i * I1
                for i in (1 - div(D.stencil_length, 2)):(div(D.stencil_length, 2))]
    end

    return (weights, Itap)
end

function get_half_offset_weights_and_stencil(
        D::DerivativeOperator{T, N, Wind, DX}, II::Base.AbstractCartesianIndex,
        s::DiscreteSpace, bs, u, jx, len = 0) where {T, N, Wind, DX <: AbstractVector}
    j, x = jx
    @assert length(bs)==0 "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    len = len == 0 ? length(s, x) : len
    @assert II[j] != length(s, x)

    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count)
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
    elseif (II[j] > (len - D.boundary_point_count))
        weights = D.high_boundary_coefs[len - II[j]]
        offset = len - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):0]
    else
        weights = D.stencil_coefs[II[j] - D.boundary_point_count]
        Itap = [II + i * I1
                for i in (1 - div(D.stencil_length, 2)):(div(D.stencil_length, 2))]
    end

    return (weights, Itap)
end

# i is the index of the offset, assuming that there is one precalculated set of weights for each offset required for a first order finite difference
function half_offset_centered_difference(D, II, s, bs, jx, u, ufunc)
    ndims(u, s) == 0 && return Num(0)
    j, x = jx
    weights, Itap = get_half_offset_weights_and_stencil(D, II, s, bs, u, jx)
    return sym_dot(weights, ufunc(u, Itap, x))
end
