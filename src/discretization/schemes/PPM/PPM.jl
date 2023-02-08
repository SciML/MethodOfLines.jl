function ppm_interp(Im2, Im1, II, Ip1, Ip2, udisc, dx::Number) 
    u_m2 = udisc[Im2]
    u_m1 = udisc[Im1]
    u_0 = udisc[II]
    u_p1 = udisc[Ip1]
    u_p2 = udisc[Ip2]

    ap = (7/12) * (u_0 + u_p1) - (1/12) * (u_p2 - u_m1)
    am = (7/12) * (u_m1 + u_0) - (1/12) * (u_p1 - u_m2)
    am, ap, u_0
end

function ppm_interp(Im2, Im1, II, Ip1, Ip2, udisc, dx::AbstractVector) 
    u_m2 = udisc[Im2]
    u_m1 = udisc[Im1]
    u_0 = udisc[II]
    u_p1 = udisc[Ip1]
    u_p2 = udisc[Ip2]

    # CW 1.7
    d_a0 = (dx[II] / (dx[Im1] + dx[II] + dx[Ip1])) * ((2 * dx[Im1] + dx[II]) * (u_p1 - u_0) / (dx[Ip1] + dx[II]) + (dx[II] + 2 * dx[Ip1]) * (u_0 - u_m1) / (dx[Im1] + dx[II]))
    d_am1 = (dx[Im1] / (dx[Im2] + dx[Im1] + dx[II])) * ((2 * dx[Im2] + dx[Im1]) * (u_0 - u_m1) / (dx[II] + dx[Im1]) + (dx[Im1] + 2 * dx[II]) * (u_m1 - u_m2) / (dx[Im2] + dx[Im1]))

    # CW 1.8
    ud_p1 = u_p1 - u_0
    ud_0 = u_0 - u_m1
    ud_m1 = u_m1 - u_m2
    if sign(ud_p1) == sign(ud_0)
        d_a0 = min(abs(d_a0), min(2 * abs(ud_p1), 2 * abs(ud_0))) * sign(d_a0)
    else
        d_a0 = 0
    end
    if sign(ud_0) == sign(ud_m1)
        d_am1 = min(abs(d_am1), min(2 * abs(ud_0), 2 * abs(ud_m1))) * sign(d_am1)
    else
        d_am1 = 0
    end

    coeffp1 = dx[II] / (dx[II] + dx[Ip1])
    coeffp2 = 1 / (dx[Im1] + dx[II] + dx[Ip1] + dx[Ip2])
    coeffp3 = (2 * dx[Ip1] * dx[II]) / (dx[II] + dx[Ip1])
    coeffp4 = (dx[Im1] + dx[II]) / (2 * dx[II] + dx[Ip1])
    coeffp5 = (dx[Ip2] + dx[Ip1]) / (2 * dx[Ip1] + dx[II])
    coeffp6 = dx[II] * (dx[Im1] + dx[II]) / (2 * dx[II] + dx[Ip1])
    coeffp7 = dx[Ip1] * (dx[Ip1] + dx[Ip2]) / (dx[II] + 2 * dx[Ip1])

    coeffm1 = dx[Im1] / (dx[Im1] + dx[II])
    coeffm2 = 1 / (dx[Im2] + dx[Im1] + dx[II] + dx[Ip1])
    coeffm3 = (2 * dx[II] * dx[Im1]) / (dx[Im1] + dx[II])
    coeffm4 = (dx[Im2] + dx[Im1]) / (2 * dx[Im1] + dx[II])
    coeffm5 = (dx[Ip1] + dx[II]) / (2 * dx[II] + dx[Im1])
    coeffm6 = dx[Im1] * (dx[Im2] + dx[Im1]) / (2 * dx[Im1] + dx[II])
    coeffm7 = dx[II] * (dx[II] + dx[Ip1]) / (dx[Im1] + 2 * dx[II])

    ap = u_0 + coeffp1 * (u_p1 - u_0) + coeffp2 * (coeffp3 * (coeffp4 - coeffp5) * (u_p1 - u_0) - coeffp6 * ud_p1 + coeffp7 * ud_0)
    am = u_m1 + coeffm1 * (u_0 - u_m1) + coeffm2 * (coeffm3 * (coeffm4 - coeffm5) * (u_0 - u_m1) - coeffm6 * ud_0 + coeffm7 * ud_m1)
    am, ap, u_0
end

"""
Corrects values between zone boundaries as per CW84 1.10
Slightly inefficient in the case that cond is true, because ifelse can't handle symbolic tuples very well
"""
function ppm_zone_boundary_correct(am, ap, a0)
    C1 = (ap - am) * (a0 - (am + ap)/2)
    C2 = (ap - am)^2 / 6
    cond = sign(ap - a0) != sign(a0 - am)
    aL = ifelse(cond, a0, ifelse(C1 > C2, 3*a0 - 2*ap, am))
    aR = ifelse(cond, a0, ifelse(-C2 > C1, 3*a0 - 2*am, ap))
    aL, aR
end

"""
Computes the spatial derivative as per CW84 1.11-1.13.
True PPM requires knowledge of u and Δt, which we don't have;
instead, we use the Courant number as an interpolation hyperparameter
and always use the positive-advection version of PPM.
"""
function ppm_du(aL, aR, a0, courant)
    a6 = 6 * (a0 - (aL + aR) / 2)
    return aR - (courant / 2) * (aR - aL - (1 - 2 * courant / 3) * a6)
end

function ppm_dudx(am, ap, ap2, a0, a1, Cx, dx::Number, II, Ip1)
    return (ppm_du(ap, ap2, a1, Cx / dx) - ppm_du(am, ap, a0, Cx / dx)) / dx
end

function ppm_dudx(am, ap, ap2, a0, a1, Cx, dx::AbstractVector, II, Ip1)
    return (ppm_du(ap, ap2, a1, Cx / dx[Ip1]) - ppm_du(am, ap, a0, Cx / dx[II])) / dx[II]
end

"""
Implements the piecewise parabolic method for fixed dx.
Uses Equation 1.9 from Collela and Woodward, 1984. 
"""
function ppm(II::CartesianIndex, s::DiscreteSpace, ppmscheme::PPMScheme, bs, jx, u, dx::Union{Number,AbstractVector})
    j, x = jx
    I1 = unitindex(ndims(u, s), j)

    Im2 = bwrap(II - 2I1, bs, s, jx)
    Im1 = bwrap(II - I1, bs, s, jx)
    Ip1 = bwrap(II + I1, bs, s, jx)
    Ip2 = bwrap(II + 2I1, bs, s, jx)
    Ip3 = bwrap(II + 3I1, bs, s, jx)
    is = map(I -> I[j], [Im1, II, Ip1, Ip2])
    for i in is
        if i < 1
            return nothing
        elseif i > length(s, x)
            return nothing
        end
    end

    if Ip3[j] > length(s, x)
        Ip3 = Ip2
    end
    if Im2[j] < 1
        Im2 = Im1
    end

    am, ap, a0 = ppm_interp(Im2, Im1, II, Ip1, Ip2, s.discvars[u], dx) # dispatches to number or abstractvector
    _, ap2, a1 = ppm_interp(Im1, II, Ip1, Ip2, Ip3, s.discvars[u], dx)
    am, ap = ppm_zone_boundary_correct(am, ap, a0)
    _, ap2 = ppm_zone_boundary_correct(ap, ap2, a1)
    
    ppm_dudx(am, ap, ap2, a0, a1, ppmscheme.Cx, dx, II, Ip1)
end 

function generate_PPM_rules(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    return reduce(safe_vcat, [[(Differential(x))(u) => ppm(Idx(II, s, u, indexmap), s, derivweights.advection_scheme, filter_interfaces(bcmap[operation(u)][x]), (x2i(s, u, x), x), u, s.dxs[x]) for x in params(u, s)] for u in depvars], init = [])
end