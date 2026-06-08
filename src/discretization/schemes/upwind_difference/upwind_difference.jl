@inline function _upwind_difference(
        D::DerivativeOperator{T, N, Wind, DX}, II, s, bs,
        ispositive, u, jx
    ) where {T, N, Wind, DX <: Number}
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    haslower, hasupper = haslowerupper(bs, x)
    if !ispositive
        if (II[j] > (length(s, x) - D.boundary_point_count)) & !hasupper
            weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):0]
        else
            weights = D.stencil_coefs
            Itap = [bwrap(II + i * I1, bs, s, jx) for i in 0:(D.stencil_length - 1)]
        end
    else
        if (II[j] <= D.offside) & !haslower
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
        else
            weights = D.stencil_coefs
            Itap = [bwrap(II + i * I1, bs, s, jx) for i in (-D.stencil_length + 1):0]
        end
    end
    return weights, Itap
end

# Independent variable for a stencil tap on the far side of an interface;
# returns b.x2 when I.A matches the coupled variable of boundary b, otherwise x.
@inline function _upwind_stencil_grid_iv(I::RefCartesianIndex, s, bs, j, x)
    if I.A !== nothing
        for b in bs
            if I.A === s.discvars[depvar(b.u2, s)]
                return b.x2
            end
        end
    end
    return x
end

# Domain length L for same-domain periodic interfaces (x ~ x); nothing for
# cross-domain joints (x1 ~ x2).
@inline function _upwind_stencil_periodic_length(s, bs, j, x)
    for b in bs
        if b isa InterfaceBoundary && isequal(b.x, x) && isequal(b.x2, x)
            g = s.grid[x]
            return g[end] - g[firstindex(g)]
        end
    end
    return nothing
end

# Physical coordinate of stencil tap I on the nonuniform grid, resolving
# RefCartesianIndex taps to the correct domain. When `raw_j` is supplied
# (index before bwrap), shift by ±L on same-domain periodic wraps so
# Fornberg stencils stay monotone.
@inline function _upwind_stencil_coord(I, s, bs, j, x; raw_j = nothing)
    grid_iv = I isa RefCartesianIndex ? _upwind_stencil_grid_iv(I, s, bs, j, x) : x
    tap_j = I isa RefCartesianIndex ? I.I[j] : I[j]
    coord = s.grid[grid_iv][tap_j]
    if raw_j !== nothing && isequal(grid_iv, x)
        L = _upwind_stencil_periodic_length(s, bs, j, x)
        if L !== nothing
            if raw_j > length(s, x)
                coord += L
            elseif raw_j < 1
                coord -= L
            end
        end
    end
    return coord
end

# Index offsets of the one-sided upwind stencil along the derivative direction.
@inline function _nonuniform_upwind_stencil_offsets(D, ispositive)
    return if !ispositive
        0:(D.stencil_length - 1)
    else
        (-D.stencil_length + 1):0
    end
end

# True when the stencil leaves the domain or sits at a boundary point where
# the standard nonuniform upwind stencil does not apply.
@inline function _nonuniform_upwind_cross_domain_needed(
        D::DerivativeOperator, II, s, bs, ispositive, u, jx
    )
    length(bs) == 0 && return false
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    l = length(s, x)
    ij = II[j]
    offsets = _nonuniform_upwind_stencil_offsets(D, ispositive)
    is_crossing = any(i -> begin
        idx = (II + i * I1)[j]
        return idx < 1 || idx > l
    end, offsets)
    is_boundary_blindspot = ispositive ? ij == l : ij == 1
    return is_crossing || is_boundary_blindspot
end

# Build (weights, Itap) for nonuniform upwind differencing at interfaces via
# bwrap and coordinate-based calculate_weights. Weights are calculated
# dynamically here, as this is the point where the specific variable's
# boundary map (bs) and cross-domain coordinates are naturally accessible.
@inline function _nonuniform_upwind_interface_difference(
        D::DerivativeOperator{T, N, Wind, DX}, II, s, bs,
        ispositive, u, jx
    ) where {T, N, Wind, DX <: AbstractVector}
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    offsets = _nonuniform_upwind_stencil_offsets(D, ispositive)
    raw_js = [(II + offsets[k] * I1)[j] for k in eachindex(offsets)]
    Itap = [bwrap(II + offsets[k] * I1, bs, s, jx) for k in eachindex(offsets)]
    x0 = _upwind_stencil_coord(II, s, bs, j, x)
    coords = [
        _upwind_stencil_coord(Itap[k], s, bs, j, x; raw_j = raw_js[k]) for k in eachindex(Itap)
    ]
    weights = calculate_weights(D.derivative_order, x0, coords)
    return weights, Itap
end

@inline function _upwind_difference(
        D::DerivativeOperator{T, N, Wind, DX}, II, s, bs,
        ispositive, u, jx
    ) where {T, N, Wind, DX <: AbstractVector}
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    if _nonuniform_upwind_cross_domain_needed(D, II, s, bs, ispositive, u, jx)
        return _nonuniform_upwind_interface_difference(D, II, s, bs, ispositive, u, jx)
    end
    if !ispositive
        @assert D.offside == 0

        if (II[j] > (length(s, x) - D.boundary_point_count))
            weights = D.high_boundary_coefs[length(s, x) - II[j] + 1]
            offset = length(s, x) - II[j]
            Itap = [II + (i + offset) * I1 for i in (-D.boundary_stencil_length + 1):0]
        else
            weights = D.stencil_coefs[II[j]]
            Itap = [II + i * I1 for i in 0:(D.stencil_length - 1)]
        end
    else
        if (II[j] <= D.offside)
            weights = D.low_boundary_coefs[II[j]]
            offset = 1 - II[j]
            Itap = [II + (i + offset) * I1 for i in 0:(D.boundary_stencil_length - 1)]
        else
            weights = D.stencil_coefs[II[j] - D.offside]
            Itap = [II + i * I1 for i in (-D.stencil_length + 1):0]
        end
    end
    return weights, Itap
end

"""
# upwind_difference
Generate a finite difference expression in `u` using the upwind difference at point `II::CartesianIndex`
in the direction of `x`
"""
function upwind_difference(
        d::Int, II::CartesianIndex, s::DiscreteSpace,
        bs, derivweights, jx, u, ufunc, ispositive
    )
    j, x = jx
    # return if this is an ODE
    ndims(u, s) == 0 && return 0
    D = !ispositive ? derivweights.windmap[1][Differential(x)^d] :
        derivweights.windmap[2][Differential(x)^d]
    #@show D.stencil_coefs, D.stencil_length, D.boundary_stencil_length, D.boundary_point_count
    # unit index in direction of the derivative
    weights, Itap = _upwind_difference(D, II, s, bs, ispositive, u, jx)
    return sym_dot(weights, ufunc(u, Itap, x))
end

function upwind_difference(
        expr, d::Int, II::CartesianIndex, s::DiscreteSpace, bs,
        depvars, derivweights, (j, x), u, central_ufunc, indexmap
    )
    # TODO: Allow derivatives in expr
    expr = substitute(
        expr, Dict(valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap))
    )
    return IfElse.ifelse(
        expr > 0,
        expr *
            upwind_difference(d, II, s, bs, derivweights, (j, x), u, central_ufunc, true),
        expr *
            upwind_difference(d, II, s, bs, derivweights, (j, x), u, central_ufunc, false)
    )
end

@inline function generate_winding_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms; skip = []
    )
    wind_ufunc(v, I, x) = s.discvars[v][I]
    # for all independent variables and dependant variables
    rules = safe_vcat(#Catch multiplication
        reduce(
            safe_vcat,
            [
                reduce(
                        safe_vcat,
                        [
                            [
                                @rule *(
                                    ~~a,
                                    $(Differential(x)^d)(u),
                                    ~~b
                                ) => upwind_difference(
                                    *(~a..., ~b...), d, Idx(II, s, u, indexmap), s,
                                    filter_interfaces(bcmap[operation(u)][x]), depvars,
                                    derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap
                                )
                                for d in (
                                    let orders = derivweights.orders[x]
                                        setdiff(orders[isodd.(orders)], skip)
                                end
                                )
                            ] for x in ivs(u, s)
                        ],
                        init = []
                    ) for u in depvars
            ],
            init = []
        ),

        #Catch division and multiplication, see issue #1
        reduce(
            safe_vcat,
            [
                reduce(
                        safe_vcat,
                        [
                            [
                                @rule /(
                                    *(~~a, $(Differential(x)^d)(u), ~~b),
                                    ~c
                                ) => upwind_difference(
                                    *(~a..., ~b...) / ~c, d, Idx(II, s, u, indexmap), s,
                                    filter_interfaces(bcmap[operation(u)][x]), depvars,
                                    derivweights, (x2i(s, u, x), x), u, wind_ufunc, indexmap
                                )
                                for d in (
                                    let orders = derivweights.orders[x]
                                        setdiff(orders[isodd.(orders)], skip)
                                end
                                )
                            ] for x in ivs(u, s)
                        ],
                        init = []
                    ) for u in depvars
            ],
            init = []
        )
    )

    wind_rules = []

    # wind_exprs = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(wind_rules, t => r(t))
            end
        end
    end

    return safe_vcat(
        wind_rules,
        vec(
            mapreduce(safe_vcat, depvars, init = []) do u
                mapreduce(safe_vcat, ivs(u, s), init = []) do x
                    j = x2i(s, u, x)
                    let orders = setdiff(derivweights.orders[x], skip)
                        oddorders = orders[isodd.(orders)]
                        # for all odd orders
                        if length(oddorders) > 0
                            map(oddorders) do d
                                (Differential(x)^d)(u) => upwind_difference(
                                    d, Idx(II, s, u, indexmap), s,
                                    filter_interfaces(bcmap[operation(u)][x]),
                                    derivweights, (j, x), u, wind_ufunc, true
                                )
                            end
                        else
                            []
                        end
                    end
                end
            end
        )
    )
end
