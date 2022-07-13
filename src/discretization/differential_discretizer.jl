# Use DiffEqOperators to generate weights and calculate derivative orders
struct DifferentialDiscretizer{T,D1}
    approx_order::Int
    map::D1
    halfoffsetmap::Tuple{Dict,Dict}
    windmap::Tuple{Dict,Dict}
    interpmap::Dict{Num,DiffEqOperators.DerivativeOperator}
    orders::Dict{Num,Vector{Int}}
end

function DifferentialDiscretizer(pdesys, s, discretization)
    pdeeqs = pdesys.eqs isa Vector ? pdesys.eqs : [pdesys.eqs]
    bcs = pdesys.bcs isa Vector ? pdesys.bcs : [pdesys.bcs]
    approx_order = discretization.approx_order
    upwind_order = discretization.upwind_order
    d_orders(x) = reverse(sort(collect(union((differential_order(pde.rhs, x) for pde in pdeeqs)..., (differential_order(pde.lhs, x) for pde in pdeeqs)..., (differential_order(bc.rhs, x) for bc in bcs)..., (differential_order(bc.lhs, x) for bc in bcs)...))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.x̄), (k,u) in enumerate(s.ū)]
    differentialmap = Array{Pair{Num,DiffEqOperators.DerivativeOperator},1}()
    nonlinlap_inner = []
    nonlinlap_outer = []
    windpos = []
    windneg = []
    interp = []
    orders = []
    # Hardcoded to centered difference, generate weights for each differential

    for x in s.x̄
        orders_ = d_orders(x)
        push!(orders, x => orders_)
        _orders = Set(vcat(orders_, [1, 2]))

        if s.grid[x] isa StepRangeLen # Uniform grid case
            # TODO: Only generate weights for derivatives that are actually used and avoid redundant calculations
            rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, s.dxs[x]) for d in _orders]

            windpos = vcat(windpos, [(Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, s.dxs[x], 0) for d in orders_[isodd.(orders_)]])
            windneg = vcat(windneg, [(Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, s.dxs[x], d + upwind_order - 1) for d in orders_[isodd.(orders_)]])


            nonlinlap_inner = vcat(nonlinlap_inner, [Differential(x)^d => CompleteHalfCenteredDifference(d, approx_order, s.dxs[x]) for d in _orders])
            nonlinlap_outer = push!(nonlinlap_outer, Differential(x) => CompleteHalfCenteredDifference(1, approx_order, s.dxs[x]))
            differentialmap = vcat(differentialmap, rs)
            # A 0th order derivative off the grid is an interpolation
            push!(interp, x => CompleteHalfCenteredDifference(0, approx_order, s.dxs[x]))

        elseif s.grid[x] isa AbstractVector # The nonuniform grid case
            rs = [(Differential(x)^d) => CompleteCenteredDifference(d, approx_order, s.grid[x]) for d in _orders]

            windpos = vcat(windpos, [(Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, s.grid[x], 0) for d in orders_[isodd.(orders_)]])
            windneg = vcat(windneg, [(Differential(x)^d) => CompleteUpwindDifference(d, upwind_order, s.grid[x], d + upwind_order - 1) for d in orders_[isodd.(orders_)]])

            discx = s.grid[x]
            nonlinlap_inner = vcat(nonlinlap_inner, [Differential(x)^d => CompleteHalfCenteredDifference(d, approx_order, s.grid[x]) for d in _orders])
            nonlinlap_outer = push!(nonlinlap_outer, Differential(x) => CompleteHalfCenteredDifference(1, approx_order, [(discx[i+1] + discx[i]) / 2 for i in 1:length(discx)-1]))
            differentialmap = vcat(differentialmap, rs)
            # A 0th order derivative off the grid is an interpolation
            push!(interp, x => CompleteHalfCenteredDifference(0, approx_order, s.grid[x]))
        else
            @assert false "s.grid contains nonvectors"
        end
    end

    return DifferentialDiscretizer{eltype(orders),typeof(Dict(differentialmap))}(approx_order, Dict(differentialmap), (Dict(nonlinlap_inner), Dict(nonlinlap_outer)), (Dict(windpos), Dict(windneg)), Dict(interp), Dict(orders))
end

"""
Performs a centered difference in `x` centered at index `II` of `u`
ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes
"""
function central_difference(D::DerivativeOperator{T,N,Wind,DX}, II, s, b, jx, u, ufunc) where {T,N,Wind,DX<:Number}
    j, x = jx
    ndims(u, s) == 0 && return Num(0)
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count) & (b isa Val{false})
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (length(s, x) - D.boundary_point_count)) & (b isa Val{false})
        weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
        offset = length(s, x) - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):1:0]
    else
        weights = D.stencil_coefs
        Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i in half_range(D.stencil_length)]
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.

    return dot(weights, ufunc(u, Itap, x))
end

function central_difference(D::DerivativeOperator{T,N,Wind,DX}, II, s, b, jx, u, ufunc) where {T,N,Wind,DX<:AbstractVector}
    j, x = jx
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    ndims(u, s) == 0 && return Num(0)
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count) & (b isa Val{false})
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (length(s, x) - D.boundary_point_count)) & (b isa Val{false})
        weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
        offset = length(s, x) - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):1:0]
    else
        weights = D.stencil_coefs[II[j]-D.boundary_point_count]
        Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i in half_range(D.stencil_length)]
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.

    return dot(weights, ufunc(u, Itap, x))
end

@inline function _upwind_difference(D::DerivativeOperator{T,N,Wind,DX}, II, s, b, ispositive, u, jx) where {T,N,Wind,DX<:Number}
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    if !ispositive
        if (II[j] > (length(s, x) - D.boundary_point_count)) & (b isa Val{false})
            weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):0]
        else
            weights = D.stencil_coefs
            Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i in 0:D.stencil_length-1]
        end
    else
        if (II[j] <= D.offside) & (b isa Val{false})
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
        else
            weights = D.stencil_coefs
            Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i in -D.stencil_length+1:0]
        end
    end
    return weights, Itap
end

@inline function _upwind_difference(D::DerivativeOperator{T,N,Wind,DX}, II, s, b, ispositive, u, jx) where {T,N,Wind,DX<:AbstractVector}
    j, x = jx
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    I1 = unitindex(ndims(u, s), j)
    if !ispositive
        @assert D.offside == 0

        if (II[j] > (length(s, x) - D.boundary_point_count))
            weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):0]
        else
            weights = D.stencil_coefs[II[j]]
            Itap = [II + i * I1 for i in 0:D.stencil_length-1]
        end
    else
        if (II[j] <= D.offside)
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
        else
            weights = D.stencil_coefs[II[j]-D.offside]
            Itap = [II + i * I1 for i in -D.stencil_length+1:0]
        end
    end
    return weights, Itap
end

function upwind_difference(d::Int, II::CartesianIndex, s::DiscreteSpace, b, derivweights, jx, u, ufunc, ispositive)
    j, x = jx
    ndims(u, s) == 0 && return Num(0)
    D = !ispositive ? derivweights.windmap[1][Differential(x)^d] : derivweights.windmap[2][Differential(x)^d]
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    weights, Itap = _upwind_difference(D, II, s, b, ispositive, u, jx)
    return dot(weights, ufunc(u, Itap, x))
end


"""
Get the weights and stencil for the inner half offset centered difference for the nonlinear laplacian for a given index and differentiating variable.

Does not discretize so that the weights can be used in a replacement rule
TODO: consider refactoring this to harmonize with centered difference

Each index corresponds to the weights and index for the derivative at index i+1/2
"""
function get_half_offset_weights_and_stencil(D::DerivativeOperator{T,N,Wind,DX}, II::CartesianIndex, s::DiscreteSpace, b, u, jx, len=0) where {T,N,Wind,DX<:Number}
    j, x = jx
    len = len == 0 ? length(s, x) : len
    @assert II[j] != length(s, x)

    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count) & (b isa Val{false})
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (len - D.boundary_point_count)) & (b isa Val{false})
        weights = D.high_boundary_coefs[len-II[j]]
        offset = len - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):0]
    else
        weights = D.stencil_coefs
        Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i in (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2))]
    end

    return (weights, Itap)
end

function get_half_offset_weights_and_stencil(D::DerivativeOperator{T,N,Wind,DX}, II::CartesianIndex, s::DiscreteSpace, b, u, jx, len=0) where {T,N,Wind,DX<:AbstractVector}
    j, x = jx
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    len = len == 0 ? length(s, x) : len
    @assert II[j] != length(s, x)

    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count)
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (len - D.boundary_point_count))
        weights = D.high_boundary_coefs[len-II[j]]
        offset = len - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length+1):0]
    else
        weights = D.stencil_coefs[II[j]-D.boundary_point_count]
        Itap = [II + i * I1 for i in (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2))]
    end

    return (weights, Itap)
end

# i is the index of the offset, assuming that there is one precalculated set of weights for each offset required for a first order finite difference
function half_offset_centered_difference(D, II, s, b, jx, u, ufunc)
    ndims(u, s) == 0 && return Num(0)
    j, x = jx
    weights, Itap = get_half_offset_weights_and_stencil(D, II, s, b, u, jx)
    return dot(weights, ufunc(u, Itap, x))
end

"""
Implements the WENO scheme of Jiang and Shu.

Specified in https://repository.library.brown.edu/studio/item/bdr:297524/PDF/ (Page 8-9)

Implementation inspired by https://github.com/ranocha/HyperbolicDiffEq.jl/blob/84c2d882e0c8956457c7d662bf7f18e3c27cfa3d/src/finite_volumes/weno_jiang_shu.jl
"""
@inline function weno(II::CartesianIndex, s::DiscreteSpace, b, jx, u, dx::Number)
    j, x = jx
    ε = 1e-6

    I1 = unitindex(ndims(u, s), j)

    udisc = s.discvars[u]

    u_m2 = udisc[wrapperiodic(II - 2I1, s, b, u, jx)]
    u_m1 = udisc[wrapperiodic(II - I1, s, b, u, jx)]
    u_0 = udisc[II]
    u_p1 = udisc[wrapperiodic(II + I1, s, b, u, jx)]
    u_p2 = udisc[wrapperiodic(II + 2I1, s, b, u, jx)]

    γ1 = 1 / 10
    γ2 = 3 / 5
    γ3 = 3 / 10

    β1 = 13 * (u_m2 - 2 * u_m1 + u_0)^2 / 12 + (u_m2 - 4 * u_m1 + 3 * u_0)^2 / 4
    β2 = 13 * (u_m1 - 2 * u_0 + u_p1)^2 / 12 + (u_m1 - u_p1)^2 / 4
    β3 = 13 * (u_0 - 2 * u_p1 + u_p2)^2 / 12 + (3 * u_0 - 4 * u_p1 + u_p2)^2 / 4

    ω1 = γ1 / (ε + β1)^2
    ω2 = γ2 / (ε + β2)^2
    ω3 = γ3 / (ε + β3)^2

    w_denom = ω1 + ω2 + ω3
    wp1 = ω1 / w_denom
    wp2 = ω2 / w_denom
    wp3 = ω3 / w_denom

    wm1 = wp3
    wm2 = wp2
    wm3 = wp1

    # * Note: H. Ranchoa has these reversed, check here first for sign error
    hp1 = (2u_m2 - 7u_m1 + 11u_0) / 6
    hp2 = -(u_m1 + 5u_0 + 2u_p1) / 6
    hp3 = (2u_0 + 5u_p1 / 6 - u_p2) / 6

    hm1 = (2u_0 + 5u_m1 / 6 - u_m2) / 6
    hm2 = -(u_p1 + 5u_0 + 2u_m1) / 6
    hm3 = (2u_p2 - 7u_p1 + 11u_0) / 6

    hp = wp1 * hp1 + wp2 * hp2 + wp3 * hp3
    hn = wm1 * hm1 + wm2 * hm2 + wm3 * hm3

    return (hp - hm) / dx
end
