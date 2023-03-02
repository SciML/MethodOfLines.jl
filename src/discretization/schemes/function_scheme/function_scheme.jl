function function_scheme(F::FunctionScheme{is_nonuniform}, II, s, bs, jx, u, ufunc) where {is_nonuniform <: Val{false}}


    # Tap points of the stencil, this uses boundary_point_count as this is equal to half the stencil size, which is what we want.
    u_disc = ufunc(u, Itap, x)
    ps = params(s)
    dx = s.dxs[x]
    t = s.time

    return f(u_disc, ps, t, x, dx)
end


function get_f_and_taps(F::FunctionScheme{is_nonuniform}, II, s, bs, jx, u, ufunc) where {is_nonuniform <: Val{false}}
    j, x = jx
    ndims(u, s) == 0 && return 0
    # unit index in direction of the derivative
    I1 = unitindex(ndims(u, s), j)
    # offset is important due to boundary proximity
    haslower, hasupper = haslowerupper(bs, x)

    lower_point_count = length(f.lower)
    upper_point_count = length(f.upper)

    if (II[j] <= D.lower_point_count) & !haslower
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
