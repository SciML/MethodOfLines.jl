"""
Performs a centered difference in `x` centered at index `II` of `u`
ufunc is a function that returns the correct discretization indexed at Itap, it is designed this way to allow for central differences of arbitrary expressions which may be needed in some schemes
"""
function central_difference_weights_and_stencil(
        D::DerivativeOperator{T, N, Wind, DX}, II, s,
        bs, jx, u
    ) where {T, N, Wind, DX <: Number}
    j, x = jx
    ndims(u, s) == 0 && return 0
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity
    haslower, hasupper = haslowerupper(bs, x)

    if (II[j] <= D.boundary_point_count) & !haslower
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
    elseif (II[j] > (length(s, x) - D.boundary_point_count)) & !hasupper
        weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
        offset = length(s, x) - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):1:0]
    else
        weights = D.stencil_coefs
        Itap = [bwrap(II + i * I1, bs, s, jx) for i in half_range(D.stencil_length)]
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.
    return weights, Itap
end

function central_difference_weights_and_stencil(
        D::DerivativeOperator{T, N, Wind, DX}, II, s, bs,
        jx, u
    ) where {T, N, Wind, DX <: AbstractVector}
    j, x = jx
    @assert length(bs) == 0 "Interface boundary conditions are not yet supported for nonuniform dx dimensions, such as $x, please post an issue to https://github.com/SciML/MethodOfLines.jl if you need this functionality."
    ndims(u, s) == 0 && return 0
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)

    if (II[j] <= D.boundary_point_count)
        weights = D.low_boundary_coefs[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
    elseif (II[j] > (length(s, x) - D.boundary_point_count))
        weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
        offset = length(s, x) - II[j]
        Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):1:0]
    else
        weights = D.stencil_coefs[II[j] - D.boundary_point_count]
        Itap = [II + i * I1 for i in half_range(D.stencil_length)]
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.

    return weights, Itap
end

function central_difference(D, II, s, bs, jx, u, ufunc)
    j, x = jx
    weights, Itap = central_difference_weights_and_stencil(D, II, s, bs, jx, u)
    return sym_dot(weights, ufunc(u, Itap, x))
end

"""
This is a catch all ruleset, as such it does not use @rule. Any even ordered derivative may be adequately approximated by these.
"""
@inline function generate_cartesian_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms
    )
    central_ufunc(u, I, x) = s.discvars[u][I]
    return reduce(
        safe_vcat,
        [
            reduce(
                    safe_vcat,
                    [
                        [
                            (Differential(x)^d)(u) => central_difference(
                                derivweights.map[Differential(x)^d], Idx(II, s, u, indexmap),
                                s, filter_interfaces(bcmap[operation(u)][x]),
                                (x2i(s, u, x), x), u, central_ufunc
                            )
                            for d in (
                                let orders = derivweights.orders[x]
                                    orders[iseven.(orders)]
                            end
                            )
                        ] for x in ivs(u, s)
                    ],
                    init = []
                ) for u in depvars
        ],
        init = []
    )
end

function generate_cartesian_rules(
        II::CartesianIndex, s::DiscreteSpace{N, M, G},
        depvars, derivweights::DifferentialDiscretizer, bcmap,
        indexmap, terms
    ) where {N, M, G <: StaggeredGrid}
    central_ufunc(u, I, x) = s.discvars[u][I]
    ufunc = central_ufunc
    xs = unique(reduce(safe_vcat, [ivs(u, s) for u in depvars], init = []))
    odd_orders = unique(
        filter(
            isodd, reduce(safe_vcat, [derivweights.orders[x] for x in xs], init = [])
        )
    )
    placeholder = []
    for u in depvars
        for x in xs
            j = x2i(s, u, x)
            jx = (j, x)
            bs = filter_interfaces(bcmap[operation(u)][x])
            for d in odd_orders
                ndims(u, s) == 0 && return 0
                # unit index in direction of the derivative
                I1 = unitindex(ndims(u, s), j)

                # offset is important due to boundary proximity
                haslower, hasupper = haslowerupper(bs, x)
                boundary_point_count = derivweights.map[Differential(x)^d].boundary_point_count

                if (II[j] <= boundary_point_count) & !haslower
                    if (s.staggeredvars[operation(u)] == EdgeAlignedVar) # can use centered diff
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = derivweights.windmap[1][Differential(x)^d].stencil_coefs
                        Itap = [II + (i * I1) for i in 0:1]
                    else #need one-sided
                        D = derivweights.halfoffsetmap[1][Differential(x)^d]
                        weights = D.low_boundary_coefs[II[j]]
                        offset = 1 - II[j]
                        Itap = [
                            II + (i + offset) * I1
                                for i in 0:(D.boundary_stencil_length - 1)
                        ]
                    end
                elseif (II[j] > (length(s, x) - boundary_point_count)) & !hasupper
                    if (s.staggeredvars[operation(u)] == CenterAlignedVar) # can use centered diff
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = derivweights.windmap[1][Differential(x)^d].stencil_coefs
                        Itap = [II + (i * I1) for i in -1:0]
                    else #need one-sided
                        D = derivweights.halfoffsetmap[1][Differential(x)^d]
                        weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
                        offset = length(s, x) - II[j]
                        Itap = [
                            II + (i + offset) * I1
                                for i in (-D.boundary_stencil_length + 1):1:0
                        ]
                    end
                else
                    if (s.staggeredvars[operation(u)] == CenterAlignedVar)
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = D.stencil_coefs
                        Itap = [bwrap(II + i * I1, bs, s, jx) for i in 0:1]
                    else
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = D.stencil_coefs
                        Itap = [bwrap(II + i * I1, bs, s, jx) for i in -1:0]
                    end
                end
                append!(
                    placeholder,
                    [(Differential(x)^d)(u) => sym_dot(weights, ufunc(u, Itap, x))]
                )
            end
        end
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.
    return reduce(safe_vcat, placeholder, init = [])
end

function central_difference(
        derivweights::DifferentialDiscretizer, II, s::DiscreteSpace{W, M, G},
        bs, jx, u, ufunc, d
    ) where {W, M, G <: StaggeredGrid}
    placeholder = []
    ndims(u, s) == 0 && return 0
    j, x = jx
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)

    # offset is important due to boundary proximity
    haslower, hasupper = haslowerupper(bs, x)
    boundary_point_count = derivweights.map[Differential(x)^d].boundary_point_count

    if (II[j] <= boundary_point_count) & !haslower
        if (s.staggeredvars[operation(u)] == EdgeAlignedVar) # can use centered diff
            D = derivweights.windmap[1][Differential(x)^d]
            weights = derivweights.windmap[1][Differential(x)^d].stencil_coefs
            Itap = [II + (i * I1) for i in 0:1]
        else #need one-sided
            D = derivweights.halfoffsetmap[1][Differential(x)^d]
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
        end
    elseif (II[j] > (length(s, x) - boundary_point_count)) & !hasupper
        if (s.staggeredvars[operation(u)] == CenterAlignedVar) # can use centered diff
            D = derivweights.windmap[1][Differential(x)^d]
            weights = derivweights.windmap[1][Differential(x)^d].stencil_coefs
            Itap = [II + (i * I1) for i in -1:0]
        else #need one-sided
            D = derivweights.halfoffsetmap[1][Differential(x)^d]
            weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):1:0]
        end
    else
        if (s.staggeredvars[operation(u)] == CenterAlignedVar)
            D = derivweights.windmap[1][Differential(x)^d]
            weights = D.stencil_coefs
            Itap = [bwrap(II + i * I1, bs, s, jx) for i in 0:1]
        else
            D = derivweights.windmap[1][Differential(x)^d]
            weights = D.stencil_coefs
            Itap = [bwrap(II + i * I1, bs, s, jx) for i in -1:0]
        end
    end
    append!(placeholder, [sym_dot(weights, ufunc(u, Itap, x))])
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.
    return reduce(safe_vcat, placeholder, init = [])
end
