########################################################################################
# Stencil interface
########################################################################################

function bc_deriv(D::DerivativeOperator, ::LowerBoundary, udisc, j, is, interior, ranges)
    weights = D.low_boundary_coefs[1]
    taps = 1:D.boundary_stencil_length
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_deriv(D::DerivativeOperator, ::UpperBoundary, udisc, j, is, interior, ranges)
    lenx = last(ranges[j])
    weights = D.high_boundary_coefs[1]
    taps = (lenx-D.boundary_stencil_length+1):lenx
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_var(::LowerBoundary, udisc, j, is, interior, ranges)
    weights = [1]
    taps = [1]
    return BoundaryDerivArrayOp(weights, taps, udisc, j, is, interior)
end

function bc_var(::UpperBoundary, udisc, j, is, interior, ranges)
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

    xrule = [x_ => arguments(u_)[j]]
    boundary_vs = map(s.ū) do v
        substitute(v, xrule)
    end
    boundary_vs = filter(xs -> any(x -> safe_unwrap(x) isa Number, xs), map(arguments, boundary_vs))

    depvarderivbcmaps = [(Differential(x_)^d)(v_) => bc_deriv(derivweights.halfoffsetmap[1][Differential(x_)^d], boundary, s.discvars[u], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs, d in derivweights.orders[x_]]

    depvarbcmaps = [v_ => bc_deriv(derivweights.interpmap[x_], boundary, s.discvars[u], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs]

    return vcat(depvarderivbcmaps, depvarbcmaps)
end

function boundary_value_rules(interior, s::DiscreteSpace{N,M,G}, boundary, derivweights, indexmap) where {N,M,G<:CenterAlignedGrid}
    u_, x_ = getvars(boundary)
    x = x_
    # depvarbcmaps will dictate what to replace the variable terms with in the bcs
    # replace u(t,0) with u₁, etc

    u = depvar(u_, s)
    args = params(u, s)
    j = findfirst(isequal(x_), args)

    xrule = [x_ => arguments(u_)[j]]
    boundary_vs = map(s.ū) do v
        substitute(v, xrule)
    end
    boundary_vs = filter(xs -> any(x -> safe_unwrap(x) isa Number, xs), map(arguments, boundary_vs))

    depvarderivbcmaps = vec([(Differential(x_)^d)(v_) => bc_deriv(derivweights.map[Differential(x_)^d], boundary, s.discvars[depvar(v_, s)], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs, d in derivweights.orders[x_]])

    depvarbcmaps = [v_ => bc_var(boundary, s.discvars[depvar(v_, s)], j, get_is(v_, s), get_interior(v_, s, interior), get_ranges(v_, s)) for v_ in boundary_vs]

    return vcat(depvarderivbcmaps, depvarbcmaps)
end

function generate_bc_op_pair(s, b::AbstractEquationBoundary, iboundary, derivweights)
    bc = boundary.eq

    boundaryvalrules = boundary_value_maps(interior, s, b, derivweights, s.indexmap)
    varrules = varrules(s, interior, depvars)
    valrules = axiesvals(s, b, interior)
    rules = vcat(boundaryvalrules, varrules, valrules)

    ranges = map(depvars(b.u, s)) do x
        if x == b.x
            offset(b, iboundary, length(s, x))
        else
            interior[x]
        end
    end

    Tuple(ranges) => broadcast_substitute(bc.lhs, rules)
end

function generate_bc_op_pair(s, b::PeriodicBoundary, iboundary, derivweights)
    u_, x_ = getvars(b)
    discu = s.discvars[depvar(u_, s)]
    j = x2i(s, depvar(u_, s), x_)
    # * Assume that the periodic BC is of the simple form u(t,0) ~ u(t,1)
    Ioffset = unitindex(N, j)*(length(s, x_) - 1)
    is = get_is(u_, s)
    idxs = map(1:length(is))
        if i == j
            1
        else
            is[i]
        end
    I = CartesianIndex(idxs)
    expr = discu[I] - discu[I + Ioffset]
    ranges = map(depvar(u_, s)) do x
        if x == x_
            1
        else
            interior[x]
        end
    end
    symindices = setdiff(1:ndims(u, s), [j])

    Tuple(ranges) => FillArrayOp(expr, filter(x -> x isa Sym, idxs), ranges[symindices])
end

function generate_bc_op_pair(s, b::AbstractInterpolatingBoundary, iboundary, derivweights)
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
    lowerboundaries = sort(filter(b -> b isa AbstractLowerBoundary, boundaries), by=ordering)
    upperboundaries = sort(filter(b -> b isa AbstractUpperBoundary, boundaries), by=ordering)

    lowerpairs = map(enumerate(lowerboundaries)) do (iboundary, boundary)
        generate_bc_op_pair(s, boundary, iboundary, derivweights)
    end
    upperpairs = map(enumerate(upperboundaries)) do (iboundary, boundary)
        generate_bc_op_pair(s, boundary, iboundary, derivweights)
    end
    return vcat(lowerpairs, upperpairs)
end

#TODO: Work out Extrap Eqs
