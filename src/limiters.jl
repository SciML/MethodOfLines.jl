struct FluxLimiterGenerator
    expr
    u
    x
end

superbee(r) = max(0.0, min(2r, 1), min(r, 2))
van_leer(r) = (r + abs(r)) / (1 + abs(r))
minmod(r) = min(1, max(0, r))

# Change this const to change the limiter
const fl_func = superbee

"""
From the bopk "Finite Volume Methods for Hyperbolic Problems" by Randall Leveque, pages 114-115
Implements a TVD flux limiter by creating symbolic equations for the correction terms
"""
function limiter_rule(II, fl, s, dt, derivweights, b)

    j = x2i(s, fl.u, fl.x)
    I1 = unitindex(ndims(fl.u, s), j)
    wp(I) = wrapperiodic(I, s, b, fl.u, (j, fl.x))
    dx = s.dxs[fl.x] isa AbstractVector ? IfElse.ifelse(fl.expr > 0, s.dxs[fl.x][wp(II - I1)[j]], s.dxs[fl.x][II[j]]) : s.dxs[fl.x]

    v = fl.expr * dt / dx
    discu = s.discvars[fl.u]
    wind_ufunc(v, I, x) = s.discvars[v][I]
    #! Needs correction for boundary handling including periodic
    theta_low = IfElse.ifelse(fl.expr > 0, discu[wp(II - I1)] - discu[wp(II - 2 * I1)], discu[wp(II + I1)] - discu[II]) / (discu[II] - discu[wp(II - I1)])
    theta_high = IfElse.ifelse(fl.expr > 0, discu[II] - discu[wp(II - I1)], discu[wp(II + 2 * I1)] - discu[wp(II + I1)]) / (discu[wp(II + I1)] - discu[II])

    reverse_difference = upwind_difference(1, II, s, pmap.map[operation(fl.u)][fl.x], derivweights, (j, fl.x), fl.u, ufunc, true)
    forward_difference = upwind_difference(1, II, s, pmap.map[operation(fl.u)][fl.x], derivweights, (j, fl.x), fl.u, ufunc, false)

    correction_term = fl.expr * IfElse.ifelse(fl.expr > 0, -0.5 * (1 - v), 0.5 * (1 + v)) * (fl_func(theta_high) * forward_difference - fl_func(theta_low) * reverse_difference)
    #! This is not correct in general. Need to sub in to the equation to get the correct expression in general.
    return correction_term
end

@inline function generate_limiter_winding_rules(II, s, dt, depvars, derivweights, pmap, indexmap, terms)
    wind_ufunc(v, I, x) = s.discvars[v][I]
    # for all independent variables and dependant variables
    rules = vcat(#Catch multiplication
        reduce(vcat, [reduce(vcat, [[@rule *(~~a, $(Differential(x)^d)(u), ~~b) => FluxLimiterGenerator(*(~a..., ~b...), u, x) for d in (
            let orders = derivweights.orders[x]
                orders[isodd.(orders)]
            end
        )] for x in params(u, s)]) for u in depvars]),

        #Catch division and multiplication, see issue #1
        reduce(vcat, [reduce(vcat, [[@rule /(*(~~a, $(Differential(x)^d)(u), ~~b), ~c) => FluxLimiterGenerator(*(~a..., ~b...) / ~c, u, x) for d in (
            let orders = derivweights.orders[x]
                orders[isodd.(orders)]
            end
        )] for x in params(u, s)]) for u in depvars])
    )

    wind_rules = []

    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(wind_rules, t => limiter_rule(II, r(t), s, dt, derivweights, pmap[operation(r(t).u)][r(t).x]))
            end
        end
    end

    return (vcat(wind_rules, vec(mapreduce(vcat, depvars) do u
            mapreduce(vcat, params(u, s)) do x
                j = x2i(s, u, x)
                let orders = derivweights.orders[x]
                    oddorders = orders[isodd.(orders)]
                    # for all odd orders
                    if length(oddorders) > 0
                        map(oddorders) do d
                            (Differential(x)^d)(u) => upwind_difference(d, Idx(II, s, u, indexmap), s, pmap.map[operation(u)][x], derivweights, (j, x), u, wind_ufunc, true)
                        end
                    else
                        []
                    end
                end
            end
        end)), wind_exprs)


end

# Set all terms not to be limited to 0
function generate_null_rules(terms, time)
    S = Symbolics
    SU = SymbolicUtils
    map(terms) do t
        # Might not catch cases where the time derivative term is not simple
        if S.istree(t) && SU.operation(t) isa Differential && isequal(SU.operation(u).x, time)
            nothing => 0
        else
            t => 0
        end
    end
end

"""
`generate_finite_difference_rules`

Generate a vector of finite difference rules to dictate what to replace variables in the `pde` with at the gridpoint `II`.

Care is taken to make sure that the rules only use points that are actually in the discretized grid by progressively up/downwinding the stencils when the gridpoint `II` is close to the boundary.

There is a genral catch all ruleset that uses the cartesian centered difference scheme for derivatives, and simply the discretized variable at the given gridpoint for particular variables.

There are of course more specific schemes that are used to improve stability/speed/accuracy when particular forms are encountered in the PDE. These rules are applied first to override the general ruleset.

##Currently implemented special cases are as follows:
    - Spherical derivatives
    - Nonlinear laplacian uses a half offset centered scheme for the inner derivative to improve stability
    - Spherical nonlinear laplacian.
    - Up/Downwind schemes to be used for odd ordered derivatives multiplied by a coefficient, downwinding when the coefficient is positive, and upwinding when the coefficient is negative.

Please submit an issue if you know of any special cases which impact stability or accuracy that are not implemented, with links to papers and/or code that demonstrates the special case.
"""
function generate_null_rules(II, s, depvars, pde, derivweights, pmap, indexmap)

    terms = split_terms(pde, s.x̄)

    null_rules = generate_null_rules(terms, s.time)
    # Because winding needs to know about multiplying terms, we can't split the terms into additive and multiplicative terms.
    winding_rules = generate_limiter_winding_rules(II, s, dt, depvars, derivweights, pmap, indexmap, terms)

    return vcat(vec(winding_rules), vec(null_rules))
end

function generate_limiter_func(eqs, s, dt, states, parameters)
    innerfunc = build_function(eqs, states, dt, parameters)
    return stage_limiter!(u, integrator, p, t) = innerfunc(u, integrator.dt, p)
end
