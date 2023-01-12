########################################################################################
# Stencil interface
########################################################################################

function bc_deriv(D::DerivativeOperator, ::LowerBoundaryTrait, udisc, j, is, interior, ranges)
    weights = D.low_boundary_coefs[1]
    taps = 1:D.boundary_stencil_length
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_deriv(D::DerivativeOperator, ::UpperBoundaryTrait, udisc, j, is, interior, ranges)
    lenx = last(ranges[j])
    weights = D.high_boundary_coefs[1]
    taps = (lenx-D.boundary_stencil_length+1):lenx
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_var(::LowerBoundaryTrait, udisc, j, is, interior, ranges)
    weights = [1]
    taps = [1]
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_var(::UpperBoundaryTrait, udisc, j, is, interior, ranges)
    lenx = last(ranges[j])
    weights = [1]
    taps = [lenx]
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end


function boundary_value_rules(interior, s::DiscreteSpace{N,M,G}, boundary, derivweights) where {N,M,G<:EdgeAlignedGrid}
    u_, x_ = getvars(boundary)
    x = x_
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = params(u, s)
    j = findfirst(isequal(x_), args)

    boundary_vs = filter(v -> any(x -> safe_unwrap(x) isa Number, arguments(v)), boundary.depvars)
    non_boundary_vs = filter(v -> any(x -> !(safe_unwrap(x) isa Number), arguments(v)), boundary.depvars)

    depvarderivbcmaps = [(Differential(x_)^d)(v_) => bc_deriv(derivweights.halfoffsetmap[1][Differential(x_)^d], trait(boundary), s.discvars[u], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs, d in derivweights.orders[x_]]

    depvarbcmaps = [v_ => bc_deriv(derivweights.interpmap[x_], trait(boundary), s.discvars[u], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = unwrap.(filter(v -> !any(x -> safe_unwrap(x) isa Number, arguments(v)), boundary.depvars))

    integralbcmaps = generate_whole_domain_integration_rules(interior, s, integralvs, indexmap, nothing, x_)

    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2

        otherderivmaps = vec([(Differential(x__)^d)(u__) => bc_deriv(derivweights.halfoffsetmap[1][Differential(x__)^d], trait(boundary), s.discvars[depvar(u__, s)], x2i(s, u__, x__), get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for d in derivweights.orders[x_]])

        otherbcmaps = [u__ => bc_deriv(derivweights.interpmap[x__], trait(boundary), s.discvars[depvar(u__, s)], x2i(s, u__, x__), get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for u__ in boundary_vs]
        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    varrules = varmaps(s, interior, non_boundary_vs)

    return vcat(depvarderivbcmaps, depvarbcmaps, integralbcmaps, varrules)
end

function boundary_value_rules(interior, s::DiscreteSpace{N,M,G}, boundary, derivweights) where {N,M,G<:CenterAlignedGrid}
    u_, x_ = getvars(boundary)
    x = x_
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = params(u, s)

    boundary_vs = filter(v -> any(x -> safe_unwrap(x) isa Number, arguments(v)), boundary.depvars)
    non_boundary_vs = filter(v -> any(x -> !(safe_unwrap(x) isa Number), arguments(v)), boundary.depvars)

    depvarderivbcmaps = vec([(Differential(x_)^d)(v_) => bc_deriv(derivweights.map[Differential(x_)^d], trait(boundary), s.discvars[depvar(v_, s)], x2i(s, v_, x_), get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs, d in derivweights.orders[x_]])

    depvarbcmaps = [v_ => bc_var(trait(boundary), s.discvars[depvar(v_, s)], x2i(s, v_, x_), get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs]

    # Only make a map if the integral will actually come out to the same number of dimensions as the boundary value
    integralvs = unwrap.(filter(v -> !any(x -> safe_unwrap(x) isa Number, arguments(v)), boundary.depvars))

    integralbcmaps = generate_whole_domain_integration_rules(interior, s, integralvs, indexmap, nothing, x_)

    if boundary isa HigherOrderInterfaceBoundary
        u__ = boundary.u2
        x__ = boundary.x2
        otheru = depvar(u__, s)

        j = x2i(s, otheru, x__)

        otherderivmaps = vec([(Differential(x__)^d)(u__) => bc_deriv(derivweights.map[Differential(x__)^d], trait(boundary), s.discvars[depvar(u__, s)], j, get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for d in derivweights.orders[x_]])

        otherbcmaps = [u__ => bc_var(trait(boundary), s.discvars[depvar(u__, s)], j, get_is(u__, s), get_interior(u__, s, interior), get_ranges(u__, s)) for u__ in boundary_vs]
        depvarderivbcmaps = vcat(depvarderivbcmaps, otherderivmaps)
        depvarbcmaps = vcat(depvarbcmaps, otherbcmaps)
    end

    varrules = varmaps(s, interior, non_boundary_vs)

    return vcat(depvarderivbcmaps, depvarbcmaps, integralbcmaps, varrules)
end

function generate_bc_op_pair(s, b::AbstractEquationBoundary, interior, iboundary, derivweights)
    bc = b.eq

    u_, x_ = getvars(b)

    boundaryvalrules = boundary_value_rules(interior, s, b, derivweights)
    valrules = axiesvals(s, b, interior)
    rules = vcat(boundaryvalrules, valrules)

    ranges = map(params(u_, s)) do x
        if isequal(x, x_)
            offset(b, iboundary, length(s, x))
        else
            interior[x]
        end
    end

    Tuple(ranges) => broadcast_substitute(bc.lhs, rules)
end

function generate_bc_op_pair(s, b::InterfaceBoundary, interior, iboundary, derivweights)
    u_, x_ = getvars(b)

    isupper(boundary) && return nothing
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

    is = get_is(u_, s)
    idxs = map(1:length(is))
    if i == j
        1
    else
        is[i]
    end
    I = CartesianIndex(idxs)
    ranges = map(depvar(u_, s)) do x
        if x == x_
            1
        else
            interior[x]
        end
    end

    expr = disc1[I] - disc2[I+Ioffset]
    symindices = setdiff(1:ndims(u, s), [j])

    Tuple(ranges) => FillArrayOp(expr, filter(x -> x isa Sym, idxs), ranges[symindices])
end

function generate_bc_op_pair(s, b::AbstractInterpolatingBoundary, interior, iboundary, derivweights)
    u_, x_ = getvars(b)

    j = x2i(s, depvar(u_, s), x_)
    u = depvar(u_, s)

    udisc = s.discvars[u]
    D = derivweights.boundary[x_]
    if isupper(b)
        lenx = length(s, x_)
        boffset = offset(b, iboundary, lenx)

        weights = D.high_boundary_coefs[iboundary]
        taps = setdiff((lenx-D.boundary_stencil_length+1):lenx, [boffset])
    else
        weights = D.low_boundary_coefs[iboundary]
        taps = setdiff(1:D.boundary_stencil_length, [iboundary])
    end

    ranges = map(u) do x
        if x == x_
            1
        else
            interior[x]
        end
    end

    Tuple(ranges) => BoundaryDerivArrayOp(weights, taps, udisc, j, get_is(u_, s),
                                          get_interior(u, s, interior))
end

function generate_bc_op_pairs(s, boundaries, derivweights, interior)
    lowerboundaries = sort(filter(b -> !isupper(b), boundaries), by=ordering)
    upperboundaries = sort(filter(b -> isupper(b), boundaries), by=ordering)

    lowerpairs = map(enumerate(lowerboundaries)) do (iboundary, boundary)
        generate_bc_op_pair(s, boundary, interior, iboundary, derivweights)
    end
    upperpairs = map(enumerate(upperboundaries)) do (iboundary, boundary)
        generate_bc_op_pair(s, boundary, interior, iboundary, derivweights)
    end
    return filter(pair -> pair isa Pair, vcat(lowerpairs, upperpairs))
end

#TODO: Work out Extrap Eqs
