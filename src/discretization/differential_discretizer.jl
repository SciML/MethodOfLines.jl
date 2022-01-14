# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T}
    approx_order
    map::Dict{Num, DiffEqOperators.DerivativeOperator}
    nonlinlapmap::Dict{Num, NTuple{2, DiffEqOperators.DerivativeOperator}}
    orders::Vector{T}
end

function DifferentialDiscretizer(pde, s, discretization)
    approx_order = discretization.approx_order
    d_orders(x) = reverse(sort(collect(union(differential_order(pde.rhs, x), differential_order(pde.lhs, x)))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]
    differentialmap = Array{Pair{Num,DiffEqOperators.DerivativeOperator},1}()
    nonlinlap = Array{Pair{Num, NTuple{2,DiffEqOperators.DerivativeOperator}}}(undef, nparams(s))
    orders = []
    # Hardcoded to centered difference, generate weights for each differential
    # TODO: Add handling for upwinding
    for x in s.nottime
        push!(orders, x => d_orders(x))
        # TODO: Only generate weights for derivatives that are actually used and avoid redundant calculations
        rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, s.dxs[x],length(s.grid[x])) for d in last(orders).second]

        differentialmap = vcat(differentialmap, rs)
        nonlinlap[j] = (x => (CompleteHalfCenteredDifference(0, approx_order, s.dxs[x]), CompleteHalfCenteredDifference(1, approx_order, s.dxs[x])))
    end

    return DifferentialDiscretizer{eltype(orders)}(Dict(differentialmap), Dict(nonlinlap), Dict(orders))
end

# ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes 
function central_difference(D, II, s, jx, u, ufunc)
    j, x = jx
    # unit index in direction of the derivative
    I1 = unitindices(nparams(s))[j] 
    # offset is important due to boundary proximity
    if II[j] <= D.boundary_point_count
        weights = D.low_boundary_coefs[II[j]]    
        offset = D.boundary_point_count - II[j] + 1
        Itap = [II + (i+offset)*I1 for i in half_range(D.boundary_stencil_length)]
    elseif II[j] > (length(x) - D.boundary_point_count)
        weights = D.high_boundary_coefs[length(s.grid[x])-II[j]+1]
        offset = length(x) - II[j] - D.boundary_point_count
        Itap = [II + (i+offset)*I1 for i in half_range(D.boundary_stencil_length)]
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
        offset = D.boundary_point_count - II[j] + 1
        # ? Is this offset correct?
        Itap = [II + (i+offset)*I1 for i in half_range(D.boundary_stencil_length)]
    elseif II[j] > (length(x) - D.boundary_point_count - 1)
        weights = D.high_boundary_coefs[length(s.grid[x])-II[j]+1][k]
        offset = length(x) - II[j] - D.boundary_point_count
        Itap = [II + (i+offset)*I1 for i in half_range(D.boundary_stencil_length)]
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