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
    # Pre-allocate result array to avoid repeated vcat allocations
    result = Pair{Num, Any}[]
    for u in depvars
        for x in ivs(u, s)
            orders = derivweights.orders[x]
            for d in orders
                iseven(d) || continue
                rule = (Differential(x)^d)(u) => central_difference(
                    derivweights.map[Differential(x)^d], Idx(II, s, u, indexmap),
                    s, filter_interfaces(bcmap[operation(u)][x]),
                    (x2i(s, u, x), x), u, central_ufunc
                )
                push!(result, rule)
            end
        end
    end
    return result
end

function generate_cartesian_rules(
        II::CartesianIndex, s::DiscreteSpace{N, M, G},
        depvars, derivweights::DifferentialDiscretizer, bcmap,
        indexmap, terms
    ) where {N, M, G <: StaggeredGrid}
    central_ufunc(u, I, x) = s.discvars[u][I]
    ufunc = central_ufunc
    # Build unique xs without nested reduce/vcat
    xs_set = Set{Any}()
    for u in depvars
        for x in ivs(u, s)
            push!(xs_set, x)
        end
    end
    xs = collect(xs_set)
    # Build unique odd_orders without nested reduce/vcat
    odd_orders_set = Set{Int}()
    for x in xs
        for ord in derivweights.orders[x]
            if isodd(ord)
                push!(odd_orders_set, ord)
            end
        end
    end
    odd_orders = collect(odd_orders_set)
    # Pre-allocate result array
    result = Pair{Num, Any}[]
    for u in depvars
        for x in xs
            j = x2i(s, u, x)
            jx = (j, x)
            bs = filter_interfaces(bcmap[operation(u)][x])
            for d in odd_orders
                ndims(u, s) == 0 && return result
                # unit index in direction of the derivative
                I1 = unitindex(ndims(u, s), j)

                # offset is important due to boundary proximity
                haslower, hasupper = haslowerupper(bs, x)
                boundary_point_count = derivweights.map[Differential(x)^d].boundary_point_count

                if (II[j] <= boundary_point_count) & !haslower
                    if (s.staggeredvars[operation(u)] == EdgeAlignedVar) # can use centered diff
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = derivweights.windmap[1][Differential(x)^d].stencil_coefs
                        Itap = (II, II + I1)
                    else #need one-sided
                        D = derivweights.halfoffsetmap[1][Differential(x)^d]
                        weights = D.low_boundary_coefs[II[j]]
                        offset = 1 - II[j]
                        Itap = ntuple(
                            i -> II + (i - 1 + offset) * I1,
                            D.boundary_stencil_length
                        )
                    end
                elseif (II[j] > (length(s, x) - boundary_point_count)) & !hasupper
                    if (s.staggeredvars[operation(u)] == CenterAlignedVar) # can use centered diff
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = derivweights.windmap[1][Differential(x)^d].stencil_coefs
                        Itap = (II - I1, II)
                    else #need one-sided
                        D = derivweights.halfoffsetmap[1][Differential(x)^d]
                        weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
                        offset = length(s, x) - II[j]
                        Itap = ntuple(
                            i -> II + (i - D.boundary_stencil_length + offset) * I1,
                            D.boundary_stencil_length
                        )
                    end
                else
                    if (s.staggeredvars[operation(u)] == CenterAlignedVar)
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = D.stencil_coefs
                        Itap = (bwrap(II, bs, s, jx), bwrap(II + I1, bs, s, jx))
                    else
                        D = derivweights.windmap[1][Differential(x)^d]
                        weights = D.stencil_coefs
                        Itap = (bwrap(II - I1, bs, s, jx), bwrap(II, bs, s, jx))
                    end
                end
                push!(
                    result,
                    (Differential(x)^d)(u) => sym_dot(weights, ufunc(u, Itap, x))
                )
            end
        end
    end
    return result
end

function central_difference(
        derivweights::DifferentialDiscretizer, II, s::DiscreteSpace{W, M, G},
        bs, jx, u, ufunc, d
    ) where {W, M, G <: StaggeredGrid}
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
            Itap = (II, II + I1)
        else #need one-sided
            D = derivweights.halfoffsetmap[1][Differential(x)^d]
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = ntuple(i -> II + (i - 1 + offset) * I1, D.boundary_stencil_length)
        end
    elseif (II[j] > (length(s, x) - boundary_point_count)) & !hasupper
        if (s.staggeredvars[operation(u)] == CenterAlignedVar) # can use centered diff
            D = derivweights.windmap[1][Differential(x)^d]
            weights = derivweights.windmap[1][Differential(x)^d].stencil_coefs
            Itap = (II - I1, II)
        else #need one-sided
            D = derivweights.halfoffsetmap[1][Differential(x)^d]
            weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
            offset = length(s, x) - II[j]
            Itap = ntuple(
                i -> II + (i - D.boundary_stencil_length + offset) * I1,
                D.boundary_stencil_length
            )
        end
    else
        if (s.staggeredvars[operation(u)] == CenterAlignedVar)
            D = derivweights.windmap[1][Differential(x)^d]
            weights = D.stencil_coefs
            Itap = (bwrap(II, bs, s, jx), bwrap(II + I1, bs, s, jx))
        else
            D = derivweights.windmap[1][Differential(x)^d]
            weights = D.stencil_coefs
            Itap = (bwrap(II - I1, bs, s, jx), bwrap(II, bs, s, jx))
        end
    end
    # Return result directly without intermediate placeholder array
    return sym_dot(weights, ufunc(u, Itap, x))
end
