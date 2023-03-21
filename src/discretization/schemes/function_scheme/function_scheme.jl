function get_f_and_taps(F::FunctionalScheme, II, s, bs, jx, u)
    j, x = jx
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity
    haslower, hasupper = haslowerupper(bs, x)

    lower_point_count = length(F.lower)
    upper_point_count = length(F.upper)

    if (II[j] <= lower_point_count) & !haslower
        f = F.lower[II[j]]
        offset = 1 - II[j]
        Itap = [II + (i + offset) * I1 for i in 0:(F.boundary_points-1)]
    elseif (II[j] > (length(s, x) - upper_point_count)) & !hasupper
        f = F.upper[length(s, x)-II[j]+1]
        offset = length(s, x) - II[j]
        Itap = [II + (i + offset) * I1 for i in (-F.boundary_points+1):1:0]
    else
        f = F.interior
        Itap = [bwrap(II + i * I1, bs, s, jx) for i in half_range(F.interior_points)]
    end

    return f, Itap
end

function function_scheme(F::FunctionalScheme, II, s, bs, jx, u, ufunc)
    j, x = jx
    ndims(u, s) == 0 && return 0

    f, Itap = get_f_and_taps(F, II, s, bs, jx, u)
    if isnothing(f)
        error("Scheme $(F.name) applied to $u in direction of $x at point $II is not defined.")
    end
    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.
    u_disc = ufunc(u, Itap, x)
    ps = vcat(F.ps, params(s))
    t = s.time
    itap = map(I -> I[j], Itap)
    discx = @view s.grid[x][itap]
    dx = s.dxs[x]
    if F.is_nonuniform
        if dx isa AbstractVector
            dx = @views dx[itap[1:end-1]]
        end
    elseif dx isa AbstractVector
        error("Scheme $(F.name) not implemented for nonuniform dxs.")
    end

    return f(u_disc, ps, t, discx, dx)
end

@inline function generate_advection_rules(F::FunctionalScheme, II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    central_ufunc(u, I, x) = s.discvars[u][I]
    return reduce(safe_vcat, [[(Differential(x))(u) =>
                                        function_scheme(F,
                                                        Idx(II, s, u, indexmap), s,
                                                        filter_interfaces(bcmap[operation(u)][x]),
                                                        (x2i(s, u, x), x), u,
                                                        central_ufunc)
                                for x in ivs(u, s)]
                               for u in depvars], init=[])
end
