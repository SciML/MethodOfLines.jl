function get_f_taps_coords(F::FunctionalScheme, II, s, bs, jx, u)
    j, x = jx
    I1 = unitindex(ndims(u, s), j)
    haslower, hasupper = haslowerupper(bs, x)

    lower_point_count = length(F.lower)
    upper_point_count = length(F.upper)

    if (II[j] <= lower_point_count) & !haslower
        f = F.lower[II[j]]
        offset = 1 - II[j]
        Iraw = [II + (i + offset) * I1 for i in 0:(F.boundary_points - 1)]
        Itap = Iraw
    elseif (II[j] > (length(s, x) - upper_point_count)) & !hasupper
        f = F.upper[length(s, x) - II[j] + 1]
        offset = length(s, x) - II[j]
        Iraw = [II + (i + offset) * I1 for i in (-F.boundary_points + 1):1:0]
        Itap = Iraw
    else
        # Single-wrap invariant: periodic u[1] ≡ u[N] leaves N-1 distinct nodes, so
        # N-1 >= interior_points. Deliberately conservative for one-sided interfaces.
        if (haslower || hasupper) && (length(s, x) - 1 < F.interior_points)
            error(
                "Scheme $(F.name) requires at least $(F.interior_points + 1) grid points in $x to wrap its stencil across an interface or periodic boundary; got $(length(s, x))."
            )
        end
        f = F.interior
        # Iraw: unwrapped, feeds bcoord; Itap: wrapped, feeds u lookup.
        Iraw = [II + i * I1 for i in half_range(F.interior_points)]
        Itap = [bwrap(I, bs, s, jx) for I in Iraw]
    end

    return f, Itap, Iraw
end

function get_f_and_taps(F::FunctionalScheme, II, s, bs, jx, u)
    f, Itap, _ = get_f_taps_coords(F, II, s, bs, jx, u)
    return f, Itap
end

function function_scheme(F::FunctionalScheme, II, s, bs, jx, u, ufunc)
    j, x = jx
    ndims(u, s) == 0 && return 0

    f, Itap, Iraw = get_f_taps_coords(F, II, s, bs, jx, u)
    if isnothing(f)
        error("Scheme $(F.name) applied to $u in direction of $x at point $II is not defined.")
    end
    u_disc = ufunc(u, Itap, x)
    ps = vcat(F.ps, params(s))
    t = Num(s.time)
    dx = s.dxs[x]
    if isempty(bs)
        itap = map(I -> I[j], Itap)
        discx = @view s.grid[x][itap]
        if F.is_nonuniform
            if dx isa AbstractVector
                dx = @views dx[itap[1:(end - 1)]]
            end
        elseif dx isa AbstractVector
            error("Scheme $(F.name) not implemented for nonuniform dxs.")
        end
    else
        # Wrapped taps: grid[x][itap] is non-monotonic; bcoord gives chart-transition coords.
        discx = [bcoord(I, bs, s, jx) for I in Iraw]
        if F.is_nonuniform
            if dx isa AbstractVector
                dx = diff(discx)
            end
        elseif dx isa AbstractVector
            error("Scheme $(F.name) not implemented for nonuniform dxs.")
        end
    end

    return f(u_disc, ps, t, discx, dx)
end

@inline function generate_advection_rules(
        F::FunctionalScheme, II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms
    )
    central_ufunc(u, I, x) = s.discvars[u][I]
    return reduce(
        safe_vcat,
        [
            [
                    (Differential(x))(u) => function_scheme(
                        F,
                        Idx(II, s, u, indexmap), s,
                        filter_interfaces(bcmap[operation(u)][x]),
                        (x2i(s, u, x), x), u,
                        central_ufunc
                    )
                    for x in ivs(u, s)
                ]
                for u in depvars
        ],
        init = []
    )
end
