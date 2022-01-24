# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T, D1}
    approx_order::Int
    map::D1
    halfoffsetmap::Dict
    interpmap::Dict{Num, DiffEqOperators.DerivativeOperator}
    orders::Dict{Num, Vector{Int}}
end

function DifferentialDiscretizer(pde, bcs, s, discretization)
    approx_order = discretization.approx_order
    # TODO: Include bcs in this calculation
    d_orders(x) = reverse(sort(collect(union(differential_order(pde.rhs, x), differential_order(pde.lhs, x), (differential_order(bc.rhs, x) for bc in bcs)..., (differential_order(bc.lhs, x) for bc in bcs)...))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.x̄), (k,u) in enumerate(s.ū)]
    differentialmap = Array{Pair{Num,DiffEqOperators.DerivativeOperator},1}()
    nonlinlap = []
    interp = []
    orders = []
    # Hardcoded to centered difference, generate weights for each differential
    # TODO: Add handling for upwinding

    for x in s.x̄
        orders_ = d_orders(x)
        push!(orders, x => orders_)
        _orders = Set(vcat(orders_, [1,2]))
        # TODO: Only generate weights for derivatives that are actually used and avoid redundant calculations
        rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, s.dxs[x] ) for d in _orders]
        

        nonlinlap = vcat(nonlinlap, [Differential(x)^d => CompleteHalfCenteredDifference(d, approx_order, s.dxs[x]) for d in _orders])
        differentialmap = vcat(differentialmap, rs)
        # A 0th order derivative off the grid is an interpolation
        push!(interp, x => CompleteHalfCenteredDifference(0, approx_order, s.dxs[x]))
    end

    return DifferentialDiscretizer{eltype(orders), typeof(Dict(differentialmap))}(approx_order, Dict(differentialmap), Dict(nonlinlap), Dict(interp), Dict(orders))
end

"""
Performs a centered difference in `x` centered at index `II` of `u`
ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes 
"""
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
Get the weights and stencil for the inner half offset centered difference for the nonlinear laplacian for a given index and differentiating variable.

Does not discretize so that the weights can be used in a replacement rule
TODO: consider refactoring this to harmonize with centered difference
"""
function get_half_offset_weights_and_stencil(D, II, s, jx)
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
        Itap = [II + i*I1 for i in (1-div(D.stencil_length,2)):(div(D.stencil_length,2))]
    end    

    return (weights, Itap)
end


# i is the index of the offset, assuming that there is one precalculated set of weights for each offset required for a first order finite difference
function half_offset_centered_difference(D, II, s, jx, u, ufunc)
    j, x = jx
    weights, Itap = get_half_offset_weights_and_stencil(D, II, s, jx)
    return dot(weights, ufunc(u, Itap, x))
end