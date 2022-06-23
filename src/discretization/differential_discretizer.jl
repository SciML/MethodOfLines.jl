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
    d_orders(x) = reverse(
        sort(
            collect(
                union(
                    (differential_order(pde.rhs, x) for pde in pdeeqs)...,
                    (differential_order(pde.lhs, x) for pde in pdeeqs)...,
                    (differential_order(bc.rhs, x) for bc in bcs)...,
                    (differential_order(bc.lhs, x) for bc in bcs)...,
                ),
            ),
        ),
    )

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
            rs = [
                (Differential(x)^d) =>
                    CompleteCenteredDifference(d, approx_order, s.dxs[x]) for
                d in _orders
            ]

            windpos = vcat(
                windpos,
                [
                    (Differential(x)^d) =>
                        CompleteUpwindDifference(d, upwind_order, s.dxs[x], 0) for
                    d in orders_[isodd.(orders_)]
                ],
            )
            windneg = vcat(
                windneg,
                [
                    (Differential(x)^d) => CompleteUpwindDifference(
                        d,
                        upwind_order,
                        s.dxs[x],
                        d + upwind_order - 1,
                    ) for d in orders_[isodd.(orders_)]
                ],
            )


            nonlinlap_inner = vcat(
                nonlinlap_inner,
                [
                    Differential(x)^d =>
                        CompleteHalfCenteredDifference(d, approx_order, s.dxs[x]) for
                    d in _orders
                ],
            )
            nonlinlap_outer = push!(
                nonlinlap_outer,
                Differential(x) =>
                    CompleteHalfCenteredDifference(1, approx_order, s.dxs[x]),
            )
            differentialmap = vcat(differentialmap, rs)
            # A 0th order derivative off the grid is an interpolation
            push!(interp, x => CompleteHalfCenteredDifference(0, approx_order, s.dxs[x]))

        elseif s.grid[x] isa AbstractVector # The nonuniform grid case
            rs = [
                (Differential(x)^d) =>
                    CompleteCenteredDifference(d, approx_order, s.grid[x]) for
                d in _orders
            ]

            windpos = vcat(
                windpos,
                [
                    (Differential(x)^d) =>
                        CompleteUpwindDifference(d, upwind_order, s.grid[x], 0) for
                    d in orders_[isodd.(orders_)]
                ],
            )
            windneg = vcat(
                windneg,
                [
                    (Differential(x)^d) => CompleteUpwindDifference(
                        d,
                        upwind_order,
                        s.grid[x],
                        d + upwind_order - 1,
                    ) for d in orders_[isodd.(orders_)]
                ],
            )

            discx = s.grid[x]
            nonlinlap_inner = vcat(
                nonlinlap_inner,
                [
                    Differential(x)^d =>
                        CompleteHalfCenteredDifference(d, approx_order, s.grid[x]) for
                    d in _orders
                ],
            )
            nonlinlap_outer = push!(
                nonlinlap_outer,
                Differential(x) => CompleteHalfCenteredDifference(
                    1,
                    approx_order,
                    [(discx[i+1] + discx[i]) / 2 for i = 1:length(discx)-1],
                ),
            )
            differentialmap = vcat(differentialmap, rs)
            # A 0th order derivative off the grid is an interpolation
            push!(interp, x => CompleteHalfCenteredDifference(0, approx_order, s.grid[x]))
        else
            @assert false "s.grid contains nonvectors"
        end
    end

    return DifferentialDiscretizer{eltype(orders),typeof(Dict(differentialmap))}(
        approx_order,
        Dict(differentialmap),
        (Dict(nonlinlap_inner), Dict(nonlinlap_outer)),
        (Dict(windpos), Dict(windneg)),
        Dict(interp),
        Dict(orders),
    )
end

"""
Performs a centered difference in `x` centered at index `II` of `u`
ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes
"""
function central_difference(
    D::DerivativeOperator{T,N,Wind,DX},
    II,
    s,
    b,
    jx,
    u,
    ufunc,
) where {T,N,Wind,DX<:Number}
    j, x = jx
    ndims(u, s) == 0 && return Num(0)
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count) & (b isa Val{false})
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i = 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (length(s, x) - D.boundary_point_count)) & (b isa Val{false})
        weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
        offset = length(s, x) - II[j]
        Itap = [II + (i + offset) * I1 for i = (-D.boundary_stencil_length+1):1:0]
    else
        weights = D.stencil_coefs
        Itap =
            [wrapperiodic(II + i * I1, s, b, u, jx) for i in half_range(D.stencil_length)]
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.

    return dot(weights, ufunc(u, Itap, x))
end

function central_difference(
    D::DerivativeOperator{T,N,Wind,DX},
    II,
    s,
    b,
    jx,
    u,
    ufunc,
) where {T,N,Wind,DX<:AbstractVector}
    j, x = jx
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    ndims(u, s) == 0 && return Num(0)
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count) & (b isa Val{false})
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i = 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (length(s, x) - D.boundary_point_count)) & (b isa Val{false})
        weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
        offset = length(s, x) - II[j]
        Itap = [II + (i + offset) * I1 for i = (-D.boundary_stencil_length+1):1:0]
    else
        weights = D.stencil_coefs[II[j]-D.boundary_point_count]
        Itap =
            [wrapperiodic(II + i * I1, s, b, u, jx) for i in half_range(D.stencil_length)]
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.

    return dot(weights, ufunc(u, Itap, x))
end

@inline function _upwind_difference(
    D::DerivativeOperator{T,N,Wind,DX},
    II,
    s,
    b,
    ispositive,
    u,
    jx,
) where {T,N,Wind,DX<:Number}
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    if !ispositive
        if (II[j] > (length(s, x) - D.boundary_point_count)) & (b isa Val{false})
            weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i = (-D.boundary_stencil_length+1):0]
        else
            weights = D.stencil_coefs
            Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i = 0:D.stencil_length-1]
        end
    else
        if (II[j] <= D.offside) & (b isa Val{false})
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i = 0:(D.boundary_stencil_length-1)]
        else
            weights = D.stencil_coefs
            Itap = [wrapperiodic(II + i * I1, s, b, u, jx) for i = -D.stencil_length+1:0]
        end
    end
    return weights, Itap
end

@inline function _upwind_difference(
    D::DerivativeOperator{T,N,Wind,DX},
    II,
    s,
    b,
    ispositive,
    u,
    jx,
) where {T,N,Wind,DX<:AbstractVector}
    j, x = jx
    @assert b isa Val{false} "Periodic boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    I1 = unitindex(ndims(u, s), j)
    if !ispositive
        @assert D.offside == 0

        if (II[j] > (length(s, x) - D.boundary_point_count))
            weights = D.high_boundary_coefs[length(s, x)-II[j]+1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i = (-D.boundary_stencil_length+1):0]
        else
            weights = D.stencil_coefs[II[j]]
            Itap = [II + i * I1 for i = 0:D.stencil_length-1]
        end
    else
        if (II[j] <= D.offside)
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i = 0:(D.boundary_stencil_length-1)]
        else
            weights = D.stencil_coefs[II[j]-D.offside]
            Itap = [II + i * I1 for i = -D.stencil_length+1:0]
        end
    end
    return weights, Itap
end

function upwind_difference(
    d::Int,
    II::CartesianIndex,
    s::DiscreteSpace,
    b,
    derivweights,
    jx,
    u,
    ufunc,
    ispositive,
)
    j, x = jx
    ndims(u, s) == 0 && return Num(0)
    D =
        !ispositive ? derivweights.windmap[1][Differential(x)^d] :
        derivweights.windmap[2][Differential(x)^d]
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
function get_half_offset_weights_and_stencil(
    D::DerivativeOperator{T,N,Wind,DX},
    II::CartesianIndex,
    s::DiscreteSpace,
    b,
    u,
    jx,
    len = 0,
) where {T,N,Wind,DX<:Number}
    j, x = jx
    len = len == 0 ? length(s, x) : len
    @assert II[j] != length(s, x)

    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity

    if (II[j] <= D.boundary_point_count) & (b isa Val{false})
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i = 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (len - D.boundary_point_count)) & (b isa Val{false})
        weights = D.high_boundary_coefs[len-II[j]]
        offset = len - II[j]
        Itap = [II + (i + offset) * I1 for i = (-D.boundary_stencil_length+1):0]
    else
        weights = D.stencil_coefs
        Itap = [
            wrapperiodic(II + i * I1, s, b, u, jx) for
            i = (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2))
        ]
    end

    return (weights, Itap)
end

function get_half_offset_weights_and_stencil(
    D::DerivativeOperator{T,N,Wind,DX},
    II::CartesianIndex,
    s::DiscreteSpace,
    b,
    u,
    jx,
    len = 0,
) where {T,N,Wind,DX<:AbstractVector}
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
        Itap = [II + (i + offset) * I1 for i = 0:(D.boundary_stencil_length-1)]
    elseif (II[j] > (len - D.boundary_point_count))
        weights = D.high_boundary_coefs[len-II[j]]
        offset = len - II[j]
        Itap = [II + (i + offset) * I1 for i = (-D.boundary_stencil_length+1):0]
    else
        weights = D.stencil_coefs[II[j]-D.boundary_point_count]
        Itap = [II + i * I1 for i = (1-div(D.stencil_length, 2)):(div(D.stencil_length, 2))]
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
