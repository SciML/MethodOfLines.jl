# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T}
    map::Dict{Num, DerivativeOperator}
    nonlinlapmap::Tuple{DerivativeOperator, DerivativeOperator}
    orders::Vector{T}
end

function DifferentialDiscretizer(pde, s, discretization)
    approx_order = discretization.approx_order
    d_orders(x) = reverse(sort(collect(union(differential_order(pde.rhs, x), differential_order(pde.lhs, x)))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]
    differentialmap = Array{Pair{Num,DerivativeOperator},1}()
    nonlinlap = Array{Pair{Num, NTuple{2,DerivativeOperator}}}(undef, nparams(s))
    orders = Int[]
    # Hardcoded to centered difference, generate weights for each differential
    # TODO: Add handling for upwinding
    for x in s.nottime
        push!(orders, x => d_orders(x))
        # TODO: Only generate weights for derivatives that are actually used and avoid redundant calculations
        rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, s.dxs[x],length(s.grid[x])) for d in last(orders).val]

        differentialmap = vcat(differentialmap, rs)
        nonlinlap[j] = (x => (CompleteHalfCenteredDifference(0, approx_order, s.dxs[x]), CompleteHalfCenteredDifference(1, approx_order, s.dxs[x])))
    end

    return DifferentialDiscretizer{eltype(orders)}(Dict(differentialmap), Dict(nonlinlap), Dict(orders))
end


function central_deriv_cartesian(D, II, s, jx, u)
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

    return dot(weights, s.discvars[u][Itap])
end


"""
Inner function for `get_weights_and_stencil` (see below).
"""
function _get_weights_and_stencil(D, II, I1, s, k, j, x)
    # k => i of itap
    # offset is important due to boundary proximity
    # The low boundary coeffs has a heirarchy of coefficients following: number of indices from boundary -> which half offset point does it correspond to -> weights
    if II[j] <= (D.boundary_point_count-1)
        weights = D.low_boundary_coefs[II[j]][k]    
        offset = D.boundary_point_count - II[j] + 1
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
"""
function get_weights_and_stencil(D, II, s, offset, i, jx)
    j, x = jx
    I1 = unitindices(nparams(s))[j]

    # Shift the current index to the correct offset
    II_prime = II + offset*I1
    
    return _get_weights_and_stencil(D, II_prime, I1, s, i, j, x)
end