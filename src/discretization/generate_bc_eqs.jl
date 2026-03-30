idx(b::LowerBoundary, s) = 1
idx(b::UpperBoundary, s) = length(s, b.x)
idx(b::HigherOrderInterfaceBoundary, s) = length(s, b.x)

@inline function edge(interiormap, s, u, j, islower)
    I = interiormap.I[interiormap.pde[depvar(u, s)]]
    # check needed on v1.6
    length(I) == 0 && return CartesianIndex{0}[]
    sd(i) = selectdim(I, j, i)
    I1 = unitindex(ndims(u, s), j)
    if islower
        edge = sd(1)
        # cast the edge of the interior to the edge of the boundary
        edge = edge .- [I1 * (edge[1][j] - 1)]
    else
        edge = sd(size(interiormap.I[interiormap.pde[depvar(u, s)]], j))
        edge = edge .+ [I1 * (size(s.discvars[depvar(u, s)], j) - edge[1][j])]
    end
    return edge
end

edge(s, b, interiormap) = edge(interiormap, s, b.u, x2i(s, b.u, b.x), !isupper(b))

@inline function generate_bc_eqs!(
        disc_state, s, boundaryvalfuncs, interiormap, boundary::AbstractTruncatingBoundary
    )
    args = ivs(depvar(boundary.u, s), s)
    indexmap = Dict([args[i] => i for i in 1:length(args)])
    return vcat!(
        disc_state.bceqs,
        generate_bc_eqs(s, boundaryvalfuncs, boundary, interiormap, indexmap)
    )
end

function generate_bc_eqs!(
        disc_state, s::DiscreteSpace, boundaryvalfuncs,
        interiormap, boundary::InterfaceBoundary
    )
    isupper(boundary) && return
    u_ = boundary.u
    x_ = boundary.x
    u__ = boundary.u2
    x__ = boundary.x2
    N = ndims(u_, s)
    j = x2i(s, depvar(u_, s), x_)
    # * Assume that the interface BC is of the simple form u(t,0) ~ u(t,1)
    Ioffset = unitindex(N, j) * (length(s, x__) - 1)
    disc1 = s.discvars[depvar(u_, s)]
    disc2 = s.discvars[depvar(u__, s)]

    return vcat!(
        disc_state.bceqs, vec(
            map(edge(s, boundary, interiormap)) do II
                disc1[II] ~ disc2[II + Ioffset]
            end
        )
    )
end

"""
    generate_bc_eqs_arrayop!(disc_state, s, boundaryvalfuncs, interiormap,
                             boundary::InterfaceBoundary)

ArrayOp version of periodic/interface BC generation. Instead of producing one
scalar equation per edge point, emits a single ArrayOp equation that tiles
`disc1[boundary, tang...] ~ disc2[boundary+offset, tang...]` along the
tangential dimensions.

Falls back to the scalar path for 1D (single-point edges) or if the edge
is not contiguous.
"""
function generate_bc_eqs_arrayop!(
        disc_state, s::DiscreteSpace, boundaryvalfuncs,
        interiormap, boundary::InterfaceBoundary
    )
    isupper(boundary) && return
    u_ = boundary.u
    x_ = boundary.x
    u__ = boundary.u2
    x__ = boundary.x2
    N = ndims(u_, s)
    j = x2i(s, depvar(u_, s), x_)

    edge_points = edge(s, boundary, interiormap)

    # For 1D or very small edges, use the scalar path
    if length(edge_points) <= 1
        return generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end

    Ioffset_val = length(s, x__) - 1
    disc1_raw = Symbolics.unwrap(s.discvars[depvar(u_, s)])
    disc2_raw = Symbolics.unwrap(s.discvars[depvar(u__, s)])
    disc1_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(disc1_raw)
    disc2_c = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}(disc2_raw)

    # Determine tangential dimensions and their ranges from the edge points
    # The edge has a fixed index in dimension j and varying indices in all others
    tang_dims = setdiff(1:N, [j])
    boundary_idx = edge_points[1][j]  # fixed index in the boundary-normal dimension

    # Compute the tangential ranges from the edge points
    tang_ranges = Vector{StepRange{Int,Int}}(undef, length(tang_dims))
    for (k, d) in enumerate(tang_dims)
        vals = [II[d] for II in edge_points]
        tang_ranges[k] = minimum(vals):1:maximum(vals)
    end

    # Create symbolic index variables for the tangential dimensions
    n_tang = length(tang_dims)
    _idxs_arr = SymbolicUtils.idxs_for_arrayop(SymbolicUtils.SymReal)
    _tang_idxs = [_idxs_arr[k] for k in 1:n_tang]
    tang_bases = [tang_ranges[k][1] - 1 for k in 1:n_tang]

    # Build full index expressions for disc1 and disc2
    # disc1 uses boundary_idx in dimension j, symbolic indices in tangential dims
    # disc2 uses boundary_idx + offset in dimension j, same tangential indices
    idx1_exprs = Vector{Any}(undef, N)
    idx2_exprs = Vector{Any}(undef, N)
    tang_k = 0
    for d in 1:N
        if d == j
            idx1_exprs[d] = boundary_idx
            idx2_exprs[d] = boundary_idx + Ioffset_val
        else
            tang_k += 1
            idx_expr = _tang_idxs[tang_k] + tang_bases[tang_k]
            idx1_exprs[d] = idx_expr
            idx2_exprs[d] = idx_expr
        end
    end

    lhs_expr = disc1_c[idx1_exprs...]
    rhs_expr = disc2_c[idx2_exprs...]

    ao_ranges = Dict(_tang_idxs[k] => (1:1:length(tang_ranges[k])) for k in 1:n_tang)

    lhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _tang_idxs, lhs_expr, +, nothing, ao_ranges
    )
    rhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _tang_idxs, rhs_expr, +, nothing, ao_ranges
    )

    return vcat!(disc_state.bceqs, [Symbolics.wrap(lhs_ao) ~ Symbolics.wrap(rhs_ao)])
end

function generate_boundary_val_funcs(s, depvars, boundarymap, indexmap, derivweights)
    return mapreduce(vcat, values(boundarymap)) do boundaries
        map(mapreduce(x -> boundaries[x], vcat, s.x̄)) do b
            # No interface values in equations
            if b isa InterfaceBoundary
                II -> []
                # Only make a map if it is actually possible to substitute in the boundary value given the indexmap
            elseif all(
                    x -> haskey(indexmap, x),
                    filter(x -> !(unwrap_const(safe_unwrap(x)) isa Number), b.indvars)
                )
                II -> boundary_value_maps(II, s, b, derivweights, indexmap)
            else
                II -> []
            end
        end
    end
end

function boundary_value_maps(
        II, s::DiscreteSpace{N, M, G}, boundary, derivweights,
        indexmap
    ) where {N, M, G <: EdgeAlignedGrid}
    u_, x_ = getvars(boundary)

    ufunc(v, I, x) = _disc_gather(s.discvars[v], I)

    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = ivs(u, s)
    j = findfirst(isequal(x_), args)
    IIold = II

    # We need to construct a new index in case the value at the boundary appears in an equation one dimension lower
    II = newindex(u_, II, s, indexmap, shift = true)

    val = unwrap_const(first(filter(z -> unwrap_const(z) isa Number, arguments(u_))))
    r = x_ => val
    othervars = map(boundary.depvars) do v
        substitute(v, r)
    end
    othervars = filter(
        v -> (length(arguments(v)) != 1) && any(isequal(x_), arguments(depvar(v, s))),
        othervars
    )

    depvarderivbcmaps = [
        (Differential(x_)^d)(u_) => half_offset_centered_difference(
                derivweights.halfoffsetmap[1][Differential(x_)^d],
                II, s, [], (j, x_), u, ufunc
            ) for d in derivweights.orders[x_]
    ]

    depvarbcmaps = [
        v_ => half_offset_centered_difference(
                derivweights.interpmap[x_], II, s, [],
                (x2i(s, depvar(v_, s), x_), x_), depvar(v_, s), ufunc
            )
            for v_ in [u_; othervars]
    ]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = filter(
        v -> !any(x -> unwrap_const(safe_unwrap(x)) isa Number, arguments(v)), boundary.depvars
    )

    integralbcmaps = generate_whole_domain_integration_rules(
        IIold, s, integralvs, indexmap, nothing, x_
    )

    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2
        otheru = depvar(u__, s)

        j = x2i(s, otheru, x__)
        is = [II[i] for i in 1:length(II)]
        is = [is[1:(j - 1)]..., 1, is[j:end]...]
        II = CartesianIndex(is...)

        depvarderivbcmaps = [
            (Differential(x__)^d)(u__) => half_offset_centered_difference(
                    derivweights.halfoffsetmap[1][Differential(x__)^d],
                    II, s, [], (j, x__), otheru, ufunc
                )
                for d in derivweights.orders[x_]
        ]

        depvarbcmaps = [
            u__ => half_offset_centered_difference(
                derivweights.interpmap[x__], II, s, [], (j, x__), otheru, ufunc
            ),
        ]

        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    return vcat(depvarderivbcmaps, integralbcmaps, depvarbcmaps)
end

function boundary_value_maps(
        II, s::DiscreteSpace{N, M, G}, boundary, derivweights,
        indexmap
    ) where {N, M, G <: StaggeredGrid}
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = _disc_gather(s.discvars[v], I)

    depvarderivbcmaps = []
    depvarbcmaps = []

    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    u = depvar(u_, s)
    args = ivs(u, s)
    j = findfirst(isequal(x_), args)
    IIold = II
    # We need to construct a new index in case the value at the boundary appears in an equation one dimension lower
    II = newindex(u_, II, s, indexmap)
    val = unwrap_const(first(filter(z -> unwrap_const(z) isa Number, arguments(u_))))
    r = x_ => val
    othervars = map(boundary.depvars) do v
        substitute(v, r)
    end
    othervars = filter(
        v -> (length(arguments(v)) != 1) && any(isequal(x_), arguments(depvar(v, s))),
        othervars
    )

    depvarderivbcmaps = [
        (Differential(x_)^d)(u_) => central_difference(
                derivweights, II, s, [], (x2i(s, u, x_), x_), u, ufunc, d
            )
            for d in derivweights.orders[x_]
    ]
    # generate_cartesian_rules(II, s, [u], derivweights, depvarbcmaps, indexmap, []);
    depvarbcmaps = [v_ => s.discvars[depvar(v_, s)][II] for v_ in [u_; othervars]]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = unwrap.(
        filter(
            v -> !any(x -> unwrap_const(safe_unwrap(x)) isa Number, arguments(v)), boundary.depvars
        )
    )

    integralbcmaps = generate_whole_domain_integration_rules(
        IIold, s, integralvs, indexmap, nothing, x_
    )

    # Deal with the other relevant variables if boundary isa InterfaceBoundary
    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2
        otheru = depvar(u__, s)

        is = [II[i] for i in setdiff(1:length(II), [j])]
        j = x2i(s, otheru, x__)

        is = vcat(is[1:(j - 1)], 1, is[j:end])
        II = CartesianIndex(is...)

        otherderivmaps = [
            (Differential(x__)^d)(u__) => central_difference(
                    derivweights.map[Differential(x__)^d], II, s,
                    [], (x2i(s, otheru, x__), x__), otheru, ufunc
                )
                for d in derivweights.orders[x__]
        ]
        otherbcmaps = [u__ => s.discvars[otheru][II]]

        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    return vcat(depvarderivbcmaps, integralbcmaps, depvarbcmaps)
end

function boundary_value_maps(
        II, s::DiscreteSpace{N, M, G}, boundary, derivweights,
        indexmap
    ) where {N, M, G <: CenterAlignedGrid}
    u_, x_ = getvars(boundary)
    ufunc(v, I, x) = _disc_gather(s.discvars[v], I)

    depvarderivbcmaps = []
    depvarbcmaps = []

    # * Assume that the BC is in terms of an explicit expression, not containing references to variables other than u_ at the boundary
    u = depvar(u_, s)
    args = ivs(u, s)
    j = findfirst(isequal(x_), args)
    IIold = II
    # We need to construct a new index in case the value at the boundary appears in an equation one dimension lower
    II = newindex(u_, II, s, indexmap)
    val = unwrap_const(first(filter(z -> unwrap_const(z) isa Number, arguments(u_))))
    r = x_ => val
    othervars = map(boundary.depvars) do v
        substitute(v, r)
    end
    othervars = filter(
        v -> (length(arguments(v)) != 1) && any(isequal(x_), arguments(depvar(v, s))),
        othervars
    )

    depvarderivbcmaps = [
        (Differential(x_)^d)(u_) => central_difference(
                derivweights.map[Differential(x_)^d], II,
                s, [], (x2i(s, u, x_), x_), u, ufunc
            )
            for d in derivweights.orders[x_]
    ]
    depvarbcmaps = [v_ => s.discvars[depvar(v_, s)][II] for v_ in [u_; othervars]]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = unwrap.(
        filter(
            v -> !any(x -> unwrap_const(safe_unwrap(x)) isa Number, arguments(v)), boundary.depvars
        )
    )

    integralbcmaps = generate_whole_domain_integration_rules(
        IIold, s, integralvs, indexmap, nothing, x_
    )

    # Deal with the other relevant variables if boundary isa InterfaceBoundary
    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2
        otheru = depvar(u__, s)

        is = [II[i] for i in setdiff(1:length(II), [j])]
        j = x2i(s, otheru, x__)

        is = vcat(is[1:(j - 1)], 1, is[j:end])
        II = CartesianIndex(is...)

        otherderivmaps = [
            (Differential(x__)^d)(u__) => central_difference(
                    derivweights.map[Differential(x__)^d], II, s,
                    [], (x2i(s, otheru, x__), x__), otheru, ufunc
                )
                for d in derivweights.orders[x__]
        ]
        otherbcmaps = [u__ => s.discvars[otheru][II]]

        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    return vcat(depvarderivbcmaps, integralbcmaps, depvarbcmaps)
end

function generate_bc_eqs(
        s::DiscreteSpace{N, M, G}, boundaryvalfuncs,
        boundary::AbstractTruncatingBoundary, interiormap, indexmap
    ) where {N, M, G}
    bc = boundary.eq
    return vec(
        map(edge(s, boundary, interiormap)) do II
            boundaryvalrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs)
            vmaps = varmaps(s, boundary.depvars, II, indexmap)
            varrules = axiesvals(s, depvar(boundary.u, s), boundary.x, II)
            rules = Dict(vcat(boundaryvalrules, vmaps, varrules))

            pde_substitute(bc.lhs, rules) ~ pde_substitute(bc.rhs, rules)
        end
    )
end

"""
`generate_extrap_eqs`

Pads the boundaries with extrapolation equations, extrapolated with 6th order lagrangian polynomials.
Reuses `central_difference` as this already dispatches the correct stencil, given a `DerivativeOperator` which contains the correct weights.
"""
function generate_extrap_eqs!(disc_state, pde, u, s, derivweights, interiormap, bcmap)
    args = ivs(u, s)
    length(args) == 0 && return

    lowerextents, upperextents = interiormap.stencil_extents[pde]
    vlower = interiormap.lower[pde]
    vupper = interiormap.upper[pde]
    ufunc(u, I, x) = _disc_gather(s.discvars[u], I)

    eqmap = [[] for _ in CartesianIndices(s.discvars[u])]
    for (j, x) in enumerate(args)
        ninterp = lowerextents[j] - vlower[j]
        I1 = unitindex(length(args), j)
        bs = bcmap[operation(u)][x]
        haslower, hasupper = haslowerupper(bs, x)
        while ninterp >= vlower[j]
            if haslower
                break
            end
            for Il in (edge(interiormap, s, u, j, true) .+ (ninterp * I1,))
                expr = central_difference(
                    derivweights.boundary[x], Il, s,
                    filter_interfaces(bcmap[operation(u)][x]), (j, x), u, ufunc
                )
                push!(eqmap[Il], expr)
            end
            ninterp = ninterp - 1
        end
        ninterp = upperextents[j] - vupper[j]
        while ninterp >= vupper[j]
            if hasupper
                break
            end
            for Iu in (edge(interiormap, s, u, j, false) .- (ninterp * I1,))
                expr = central_difference(
                    derivweights.boundary[x], Iu, s,
                    filter_interfaces(bcmap[operation(u)][x]), (j, x), u, ufunc
                )
                push!(eqmap[Iu], expr)
            end
            ninterp = ninterp - 1
        end
    end
    # Overlap handling
    for II in setdiff(collect(CartesianIndices(eqmap)), interiormap.I[pde])
        rhss = eqmap[II]
        if length(rhss) == 0
            continue
        elseif length(rhss) == 1
            push!(disc_state.bceqs, s.discvars[u][II] ~ rhss[1])
        else
            n = length(rhss)
            push!(disc_state.bceqs, s.discvars[u][II] ~ sum(rhss) / n)
        end
    end
    return
end

#TODO: Benchmark and optimize this

"""
    generate_bc_eqs_arrayop!(disc_state, s, boundaryvalfuncs, interiormap,
                             boundary::AbstractTruncatingBoundary, derivweights)

ArrayOp version of truncating BC (Dirichlet/Neumann/Robin) generation for 2D+.
Builds symbolic substitution rules using ArrayOp index variables and applies them
to the BC equation, producing a single ArrayOp instead of per-point scalar equations.

Falls back to the scalar path for:
- 1D (single-point edges)
- EdgeAlignedGrid (half-offset interpolation not yet supported)
- HigherOrderInterfaceBoundary
- BCs involving integral terms
"""
function generate_bc_eqs_arrayop!(
        disc_state, s::DiscreteSpace{N_s, M, G}, boundaryvalfuncs,
        interiormap, boundary::AbstractTruncatingBoundary, derivweights
    ) where {N_s, M, G}
    u_ = boundary.u
    x_ = boundary.x
    u = depvar(u_, s)
    args = ivs(u, s)
    N = length(args)

    # Fall back to scalar for unsupported cases
    if N <= 1 || G <: EdgeAlignedGrid || boundary isa HigherOrderInterfaceBoundary
        return generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end

    edge_points = edge(s, boundary, interiormap)
    if length(edge_points) <= 1
        return generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end

    # --- Edge geometry ---
    j = x2i(s, u, x_)
    tang_dims = setdiff(1:N, [j])
    n_tang = length(tang_dims)

    tang_ranges = Vector{StepRange{Int,Int}}(undef, n_tang)
    for (k, d) in enumerate(tang_dims)
        vals = [II[d] for II in edge_points]
        tang_ranges[k] = minimum(vals):1:maximum(vals)
    end

    # --- Symbolic index variables ---
    _idxs_arr = SymbolicUtils.idxs_for_arrayop(SymbolicUtils.SymReal)
    _tang_idxs = [_idxs_arr[k] for k in 1:n_tang]
    tang_bases = [tang_ranges[k][1] - 1 for k in 1:n_tang]

    # Boundary-normal grid index
    boundary_idx = newindex(u_, edge_points[1], s,
        Dict([args[i] => i for i in 1:N]))

    # Helper: build full index expressions for a variable v given its spatial args
    function symbolic_idx_exprs(v)
        v_args = ivs(v, s)
        tang_k = 0
        return map(v_args) do xv
            d_v = findfirst(isequal(xv), args)
            if d_v == j
                boundary_idx[d_v]
            else
                tang_k += 1
                _tang_idxs[tang_k] + tang_bases[tang_k]
            end
        end
    end

    _ConstSR = SymbolicUtils.BSImpl.Const{SymbolicUtils.SymReal}

    # --- Build substitution rules ---
    rules = Pair[]

    # 1. Variable value maps: u(t, x_boundary, y) => Const(discvar)[boundary_idx, _i + base]
    val = unwrap_const(first(filter(z -> unwrap_const(z) isa Number, arguments(u_))))
    r = x_ => val
    othervars = map(boundary.depvars) do v
        substitute(v, r)
    end
    othervars = filter(
        v -> (length(arguments(v)) != 1) && any(isequal(x_), arguments(depvar(v, s))),
        othervars
    )

    for v_ in [u_; othervars]
        v = depvar(v_, s)
        v_raw = Symbolics.unwrap(s.discvars[v])
        v_c = _ConstSR(v_raw)
        idx_e = symbolic_idx_exprs(v)
        push!(rules, v_ => Symbolics.wrap(v_c[idx_e...]))
    end

    # 2. Derivative maps: Dx^d(u_) => boundary stencil with symbolic tangential indices
    boundary_j = boundary_idx[j]
    for d in get(derivweights.orders, x_, Int[])
        D_key = Differential(x_)^d
        haskey(derivweights.map, D_key) || continue
        D_op = derivweights.map[D_key]

        # Determine boundary stencil weights and offsets
        # (same logic as central_difference_weights_and_stencil with bs=[])
        if boundary_j <= D_op.boundary_point_count
            weights = D_op.low_boundary_coefs[boundary_j]
            offset = 1 - boundary_j
            normal_offsets = collect(0:(D_op.boundary_stencil_length - 1)) .+ offset
        elseif boundary_j > (length(s, x_) - D_op.boundary_point_count)
            idx_from_end = length(s, x_) - boundary_j + 1
            weights = D_op.high_boundary_coefs[idx_from_end]
            offset = length(s, x_) - boundary_j
            normal_offsets = collect((-D_op.boundary_stencil_length + 1):0) .+ offset
        else
            weights = D_op.stencil_coefs
            normal_offsets = collect(half_range(D_op.stencil_length))
        end

        # Build symbolic stencil: Σ w_k * Const(discvar)[boundary_j + off_k, _i_tang + base]
        u_raw = Symbolics.unwrap(s.discvars[u])
        u_c = _ConstSR(u_raw)
        taps = map(normal_offsets) do off
            tang_k = 0
            idx_e = map(args) do xv
                if isequal(xv, x_)
                    boundary_j + off
                else
                    tang_k += 1
                    _tang_idxs[tang_k] + tang_bases[tang_k]
                end
            end
            Symbolics.wrap(u_c[idx_e...])
        end
        push!(rules, D_key(u_) => sym_dot(weights, taps))
    end

    # 3. Grid value maps: x => boundary_val, y => Const(grid_y)[_i + base]
    tang_k = 0
    for xv in args
        if isequal(xv, x_)
            push!(rules, xv => (boundary_idx[j] == 1 ? first(s.axies[xv]) : last(s.axies[xv])))
        else
            tang_k += 1
            grid_c = _ConstSR(collect(s.grid[xv]))
            push!(rules, xv => Symbolics.wrap(grid_c[_tang_idxs[tang_k] + tang_bases[tang_k]]))
        end
    end

    # 4. Dependent variable maps for other depvars in the BC (varmaps equivalent)
    indexmap = Dict([args[i] => i for i in 1:N])
    for v in boundary.depvars
        haskey(s.discvars, v) || continue
        v_raw = Symbolics.unwrap(s.discvars[v])
        v_c = _ConstSR(v_raw)
        # Use the edge point to determine the correct index for this variable
        II_sym = symbolic_idx_exprs(v)
        push!(rules, v => Symbolics.wrap(v_c[II_sym...]))
    end

    # --- Apply rules and build ArrayOp ---
    bc = boundary.eq
    rules_dict = Dict(rules)
    template_lhs = pde_substitute(bc.lhs, rules_dict)
    template_rhs = pde_substitute(bc.rhs, rules_dict)

    lhs_raw = Symbolics.unwrap(template_lhs)
    rhs_raw = Symbolics.unwrap(template_rhs)
    if !(lhs_raw isa SymbolicUtils.BasicSymbolic)
        lhs_raw = _ConstSR(lhs_raw)
    end
    if !(rhs_raw isa SymbolicUtils.BasicSymbolic)
        rhs_raw = _ConstSR(rhs_raw)
    end

    ao_ranges = Dict(_tang_idxs[k] => (1:1:length(tang_ranges[k])) for k in 1:n_tang)

    lhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _tang_idxs, lhs_raw, +, nothing, ao_ranges
    )
    rhs_ao = SymbolicUtils.ArrayOp{SymbolicUtils.SymReal}(
        _tang_idxs, rhs_raw, +, nothing, ao_ranges
    )

    # --- Validate against scalar path at first point ---
    indexmap_bc = Dict([args[i] => i for i in 1:N])
    eq_first_scalar = first(generate_bc_eqs(s, boundaryvalfuncs, boundary, interiormap, indexmap_bc))
    sub_first = Dict(_tang_idxs[k] => 1 for k in 1:n_tang)
    eq_first_arrayop = pde_substitute(template_lhs, sub_first) ~ pde_substitute(template_rhs, sub_first)

    if !_equations_match(eq_first_arrayop, eq_first_scalar)
        # Validation failed — fall back to scalar
        @debug "BC ArrayOp validation failed, falling back to scalar" boundary
        return generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end

    return vcat!(disc_state.bceqs, [Symbolics.wrap(lhs_ao) ~ Symbolics.wrap(rhs_ao)])
end

@inline function generate_corner_eqs!(disc_state, s, interiormap, N, u)
    interior = interiormap.I[interiormap.pde[u]]
    ndims(u, s) == 0 && return
    sd(i, j) = selectdim(interior, j, i)
    domain = setdiff(s.Igrid[u], interior)
    II1 = unitindices(N)
    for j in 1:N
        I1 = II1[j]
        edge = sd(1, j)
        offset = edge[1][j] - 1
        for k in 1:offset
            setdiff!(domain, vec(copy(edge) .- [I1 * k]))
        end
        edge = sd(size(interior, j), j)
        offset = size(s.discvars[u], j) - size(interior, j)
        for k in 1:offset
            setdiff!(domain, vec(copy(edge) .+ [I1 * k]))
        end
    end
    return append!(disc_state.bceqs, [s.discvars[u][I] ~ 0 for I in domain])
end

"""
Create a vector containing indices of the corners of the domain.
"""
@inline function findcorners(s::DiscreteSpace, lower, upper, u)
    args = remove(arguments(u), s.time)
    if any(lower .== 0) && any(upper .== 0)
        return CartesianIndex{2}[]
    end
    return reduce(
        vcat,
        vec.(
            map(0:3) do n
                dig = digits(n, base = 2, pad = 2)
                CartesianIndices(
                    Tuple(
                        map(enumerate(dig)) do (i, b)
                            x = args[i]
                            if b == 1
                                1:lower[i]
                            elseif b == 0
                                (length(s, x) - upper[i] + 1):length(s, x)
                            end
                        end
                    )
                )
            end
        )
    )
end

@inline function generate_corner_eqs!(disc_state, s, interiormap, pde)
    u = interiormap.var[pde]
    N = ndims(u, s)
    if N <= 1
        return
    elseif N == 2
        Icorners = findcorners(s, interiormap.lower[pde], interiormap.upper[pde], u)
        append!(disc_state.bceqs, [s.discvars[u][I] ~ 0 for I in Icorners])
    else
        generate_corner_eqs!(disc_state, s, interiormap, N, u)
    end
end
