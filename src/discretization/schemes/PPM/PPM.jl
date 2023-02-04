"""
Implements the piecewise parabolic method for fixed dx.
Uses Equation 1.9 from Collela and Woodward, 1984. 
"""
function ppm(II::CartesianIndex, s::DiscreteSpace, ppmscheme::PPMScheme, bs, jx, u, dx::Number)
    j, x = jx
    I1 = unitindex(ndims(u, s), j)

    udisc = s.discvars[u]

    Im2 = bwrap(II - 2I1, bs, s, jx)
    Im1 = bwrap(II - I1, bs, s, jx)
    Ip1 = bwrap(II + I1, bs, s, jx)
    Ip2 = bwrap(II + 2I1, bs, s, jx)
    is = map(I -> I[j], [Im2, Im1, Ip1, Ip2])
    for i in is
        if i < 1
            return nothing
        elseif i > length(s, x)
            return nothing
        end
    end

    u_m2 = udisc[Im2]
    u_m1 = udisc[Im1]
    u_0 = udisc[II]
    u_p1 = udisc[Ip1]
    u_p2 = udisc[Ip2]

    ap = (7/12) * (u_0 + u_p1) - (1/12) * (u_p2 - u_m1)
    am = (7/12) * (u_m1 + u_0) - (1/12) * (u_p1 - u_m2)
    return (ap - am) / dx
end 

function ppm(II::CartesianIndex, s::DiscreteSpace, b, jx, u, dx::AbstractVector)
    j, x = jx
    I1 = unitindex(ndims(u, s), j)

    udisc = s.discvars[u]

    Im2 = bwrap(II - 2I1, bs, s, jx)
    Im1 = bwrap(II - I1, bs, s, jx)
    Ip1 = bwrap(II + I1, bs, s, jx)
    Ip2 = bwrap(II + 2I1, bs, s, jx)
    is = map(I -> I[j], [Im2, Im1, Ip1, Ip2])
    for i in is
        if i < 1
            return nothing
        elseif i > length(s, x)
            return nothing
        end
    end

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
    return (ap - am) / dx[II]
end

function generate_PPM_rules(II::CartesianIndex, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    return reduce(safe_vcat, [[(Differential(x))(u) => ppm(Idx(II, s, u, indexmap), s, derivweights.advection_scheme, filter_interfaces(bcmap[operation(u)][x]), (x2i(s, u, x), x), u, s.dxs[x]) for x in params(u, s)] for u in depvars], init = [])
end