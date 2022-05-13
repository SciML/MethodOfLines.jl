struct FluxLimiterGenerator
    expr
    u
    x
end

superbee(r) = max(0., min(2r, 1), min(r, 2))
van_leer(r) = (r + abs(r)) / (1 + abs(r))
minmod(r) = min(1, max(0, r))

# Change this const to change the limiter
const fl_func = superbee

"""
From the bopk "Finite Volume Methods for Hyperbolic problems", pages 114-115
Implements a TVD flux limiter by creating symbolic equations for the correction terms
"""
function build_limeter_eq(II, fls, s, dt, derivweights, pmap, indexmap)
    correction_terms = map(fls) do fl
        j = x2i(s, fl.u, fl.x)
        dx = s.dxs[fl.x] isa AbstractVector ? IfElse.ifelse(fl.expr > 0, s.dxs[fl.x][II[j]-1], s.dxs[fl.x][II[j]]) : s.dxs[fl.x]

        v = fl.expr*dt/dx
        discu = s.discvars[fl.u]
        wind_ufunc(v, I, x) = s.discvars[v][I]
        I1 =
        #! Needs correction for boundary handling including periodic
        theta_low = IfElse.ifelse(fl.expr > 0, discu[II-I1] - discu[II-2*I1], discu[II+I1] - discu[II]) / (discu[II] - discu[II-I1])
        theta_high = IfElse.ifelse(fl.expr > 0, discu[II] - discu[II-I1], discu[II+2*I1] - discu[II+I1]) / (discu[II+I1] - discu[II])

        reverse_difference = upwind_difference(1, II, s, pmap.map[operation(fl.u)][fl.x], derivweights, (j,fl.x), fl.u, ufunc, true)
        forward_difference = upwind_difference(1, II, s, pmap.map[operation(fl.u)][fl.x], derivweights, (j,fl.x), fl.u, ufunc, false)

        fl.expr*IfElse.ifelse(fl.expr > 0, -0.5*(1-v), 0.5*(1+v))*(fl_func(theta_high)*forward_difference - fl_func(theta_low)*reverse_difference)
    end
    return discu[II] ~ discu[II] + sum(correction_terms)
end


function generate_limiter_func(eqs, s, dt, states, parameters)
    innerfunc = build_function(eqs, states, dt, parameters)
    return stage_limiter!(u, integrator, p, t) = innerfunc(u, integrator.dt, p)
end
