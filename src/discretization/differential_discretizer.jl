# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T, D1}
    approx_order::Int
    map::D1
    halfoffsetmap::Dict
    windmap::Dict
    interpmap::Dict{Num, DiffEqOperators.DerivativeOperator}
    orders::Dict{Num, Vector{Int}}
end

function DifferentialDiscretizer(pdesys, s, discretization)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs isa Vector ? pdesys.bcs : [pdesys.bcs]
    approx_order = discretization.approx_order
    upwind_order = discretization.upwind_order
    d_orders(x) = reverse(sort(collect(union((differential_order(pde.rhs, x) for pde in pdeeqs)..., (differential_order(pde.lhs, x) for pde in pdeeqs)..., (differential_order(bc.rhs, x) for bc in bcs)..., (differential_order(bc.lhs, x) for bc in bcs)...))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.x̄), (k,u) in enumerate(s.ū)]
    differentialmap = Array{Pair{Num,DiffEqOperators.DerivativeOperator},1}()
    nonlinlap = []
    wind = []
    interp = []
    orders = []
    # Hardcoded to centered difference, generate weights for each differential

    for x in s.x̄
        orders_ = d_orders(x)
        push!(orders, x => orders_)
        _orders = Set(vcat(orders_, [1,2]))
        # TODO: Only generate weights for derivatives that are actually used and avoid redundant calculations
        rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, s.dxs[x] ) for d in _orders]

        wind = vcat(wind, [(Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, s.dxs[x], 0) for d in orders_[isodd.(orders_)]])
        

        nonlinlap = vcat(nonlinlap, [Differential(x)^d => CompleteHalfCenteredDifference(d, approx_order, s.dxs[x]) for d in _orders])
        differentialmap = vcat(differentialmap, rs)
        # A 0th order derivative off the grid is an interpolation
        push!(interp, x => CompleteHalfCenteredDifference(0, approx_order, s.dxs[x]))
        
    end

    return DifferentialDiscretizer{eltype(orders), typeof(Dict(differentialmap))}(approx_order, Dict(differentialmap), Dict(nonlinlap), Dict(wind), Dict(interp), Dict(orders))
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

@inline function _upwind_difference(D, I, s, jx)
    j, x = jx
    I1 = unitindices(nparams(s))[j] 
    if I > (length(s, x) - D.boundary_point_count)
        weights = D.high_boundary_coefs[length(s, x)-I+1]
        offset = length(s, x) - I
        Itap = [(i+offset)*I1 for i in (-D.boundary_stencil_length+1):1:0]
    else
        weights = D.stencil_coefs
        Itap = [i*I1 for i in 0:D.stencil_length-1]
    end   
    return weights, Itap
end

function upwind_difference(d::Int, II::CartesianIndex{N}, s::DiscreteSpace{N}, derivweights, jx, u, ufunc, ispositive) where N
    j, x = jx
    D = derivweights.windmap[Differential(x)^d]
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    if !ispositive
        weights, Itap = _upwind_difference(D, length(s, x) - II[j] +1, s, jx)
        #don't need to reverse because it's already reversed by subtracting Itap
        weights = -reverse(weights)
        Itap = (II,) .- reverse(Itap)
    else
        weights, Itap = _upwind_difference(D, II[j], s, jx)
        Itap = (II,) .+ Itap
        weights = weights
    end
    return dot(weights, ufunc(u, Itap, x))
end


"""
Get the weights and stencil for the inner half offset centered difference for the nonlinear laplacian for a given index and differentiating variable.

Does not discretize so that the weights can be used in a replacement rule
TODO: consider refactoring this to harmonize with centered difference

Each index corresponds to the weights and index for the derivative at index i+1/2
"""
function get_half_offset_weights_and_stencil(D::DerivativeOperator, II::CartesianIndex, s::DiscreteSpace, jx, len = 0)
    j, x = jx
    len = len == 0 ? length(s, x) : len
    @assert II[j] != length(s, x)

    # unit index in direction of the derivative
    I1 = unitindices(nparams(s))[j] 
    # offset is important due to boundary proximity

    if II[j] < D.boundary_point_count
        weights = D.low_boundary_coefs[II[j]]    
        offset = 1 - II[j]
        Itap = [II + (i+offset)*I1 for i in 0:(D.boundary_stencil_length-1)]
    elseif II[j] > (len - D.boundary_point_count)
        try
            weights = D.high_boundary_coefs[len-II[j]]
        catch e
            print(II, len, D.high_boundary_coefs)
            throw(e)
        end
        offset = len - II[j]
        Itap = [II + (i+offset)*I1 for i in (-D.boundary_stencil_length+1):0]
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