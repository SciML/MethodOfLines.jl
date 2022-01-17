# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T, D1}
    approx_order::Int
    map::D1
    halfoffsetmap::Dict{Num, DiffEqOperators.DerivativeOperator}
    interpmap::Dict{Num, DiffEqOperators.DerivativeOperator}
    orders::Dict{Num, Vector{Int}}
end

function DifferentialDiscretizer(pde, s, discretization)
    approx_order = discretization.approx_order
    d_orders(x) = reverse(sort(collect(union(differential_order(pde.rhs, x), differential_order(pde.lhs, x)))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]
    differentialmap = Array{Pair{Num,DiffEqOperators.DerivativeOperator},1}()
    nonlinlap = []
    interp = []
    orders = []
    # Hardcoded to centered difference, generate weights for each differential
    # TODO: Add handling for upwinding
    for x in s.nottime
        push!(orders, x => d_orders(x))
        # TODO: Only generate weights for derivatives that are actually used and avoid redundant calculations
        rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, s.dxs[x] ) for d in last(orders).second]

        differentialmap = vcat(differentialmap, rs)
        push!(nonlinlap, x => CompleteHalfCenteredDifference(0, approx_order, s.dxs[x])
        push!(interp, x => CompleteHalfCenteredDifference(1, approx_order, s.dxs[x]))
    end

    return DifferentialDiscretizer{eltype(orders), typeof(Dict(differentialmap))}(approx_order, Dict(differentialmap), Dict(nonlinlap), Dict(interp), Dict(orders))
end


# ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes 
function central_difference(D, II, s, jx, u, ufunc)
    j, x = jx
    # unit index in direction of the derivative
    I1 = unitindices(nparams(s))[j] 
    # offset is important due to boundary proximity

    if II[j] <= D.boundary_point_count
        weights = D.low_boundary_coefs[II[j]]    
        offset = 1 - II[j]
        Itap = [II + (i+offset)*I1 for i in 0:(D.boundary_stencil_length-1)]
    elseif II[j] > (length(s, x) - D.boundary_point_count)
        weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
        offset = length(s, x) - II[j]
        Itap = [II + (i+offset)*I1 for i in (-D.boundary_stencil_length+1):1:0]
    else
        weights = D.stencil_coefs
        Itap = [II + i*I1 for i in half_range(D.stencil_length)]
    end   
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.

    return dot(weights, ufunc(u, Itap, x))
end

"""
Inner function for `get_half_offset_weights_and_stencil` (see below).
"""
function _get_weights_and_stencil(D, II, I1, s, k, j, x)
    # k => i of itap - 
    # offset is important due to boundary proximity
    # The low boundary coeffs has a heirarchy of coefficients following: number of indices from boundary -> which half offset point does it correspond to -> weights
    if II[j] <= (D.boundary_point_count-1)
        weights = D.low_boundary_coefs[II[j]][k]    
        offset = 1 - II[j]
        Itap = [II + (i+offset)*I1 for i in 0:(D.boundary_stencil_length-1)]
    elseif II[j] > (length(s, x) - D.boundary_point_count - 1)
        weights = D.high_boundary_coefs[length(s,x)-II[j]+1][k]
        offset = length(s, x) - II[j]
        Itap = [II + (i+offset)*I1 for i in (-D.boundary_stencil_length+1):1:0]
    else
        weights = D.stencil_coefs
        Itap = [II + (i+1)*I1 for i in half_range(D.stencil_length)]
    end    

    return (weights, Itap)
end

"""
Get the weights and stencil for the inner half offset centered difference for the nonlinear laplacian for a given index and differentiating variable.

Does not discretize so that the weights can be used in a replacement rule
TODO: consider refactoring this to harmonize with centered difference
"""
function get_half_offset_weights_and_stencil(D, II, s, offset, jx)
    j, x = jx
    I1 = unitindices(nparams(s))[j]

    # Shift the current index to the correct offset
    II_prime = II + offset*I1
    
    return _get_weights_and_stencil(D, II_prime, I1, s, offset, j, x)
end

# i is the index of the offset, assuming that there is one precalculated set of weights for each offset required for a first order finite difference
function half_offset_centered_difference(D, II, s, offset, i, jx, u, ufunc)
    j, x = jx
    I1 = unitindices(nparams(s))[j]
    # Shift the current index to the correct offset
    II_prime = II + offset*I1
    # Get the weights and stencil
    (weights, Itap) = _get_weights_and_stencil(D, II_prime, I1, s, i, j, x)
    return dot(weights, ufunc(u, Itap, x))
end