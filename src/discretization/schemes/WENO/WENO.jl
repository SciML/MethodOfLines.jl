function weno_f(u, p, t, x, dx)
    ε = p[1]

    u_m2 = u[1]
    u_m1 = u[2]
    u_0 = u[3]
    u_p1 = u[4]
    u_p2 = u[5]

    γm1 = 1 / 10
    γm2 = 3 / 5
    γm3 = 3 / 10

    β1 = 13 * (u_0 - 2 * u_p1 + u_p2)^2 / 12 + (3 * u_0 - 4 * u_p1 + u_p2)^2 / 4
    β2 = 13 * (u_m1 - 2 * u_0 + u_p1)^2 / 12 + (u_m1 - u_p1)^2 / 4
    β3 = 13 * (u_m2 - 2 * u_m1 + u_0)^2 / 12 + (u_m2 - 4 * u_m1 + 3 * u_0)^2 / 4

    ωm1 = γm1 / (ε + β1)^2
    ωm2 = γm2 / (ε + β2)^2
    ωm3 = γm3 / (ε + β3)^2

    wm_denom = ωm1 + ωm2 + ωm3
    wm1 = ωm1 / wm_denom
    wm2 = ωm2 / wm_denom
    wm3 = ωm3 / wm_denom

    γp1 = 3 / 10
    γp2 = 3 / 5
    γp3 = 1 / 10

    ωp1 = γp1 / (ε + β1)^2
    ωp2 = γp2 / (ε + β2)^2
    ωp3 = γp3 / (ε + β3)^2

    wp_denom = ωp1 + ωp2 + ωp3
    wp1 = ωp1 / wp_denom
    wp2 = ωp2 / wp_denom
    wp3 = ωp3 / wp_denom

    hm1 = (11 * u_0 - 7 * u_p1 + 2 * u_p2) / 6
    hm2 = (5 * u_0 - u_p1 + 2 * u_m1) / 6
    hm3 = (2 * u_0 + 5 * u_m1 - u_m2) / 6

    hp1 = (2 * u_0 + 5 * u_p1 - u_p2) / 6
    hp2 = (5 * u_0 + 2 * u_p1 - u_m1) / 6
    hp3 = (11 * u_0 - 7 * u_m1 + 2 * u_m2) / 6

    hp = wp1 * hp1 + wp2 * hp2 + wp3 * hp3
    hm = wm1 * hm1 + wm2 * hm2 + wm3 * hm3

    return (hp - hm) / dx
end

"""
`WENOScheme` of Jiang and Shu
## Keyword Arguments
- `epsilon`: A quantity used to prevent vanishing denominators in the scheme, defaults to `1e-6`. More sensetive problems will benefit from a smaller value. It is defined as a functional scheme.
"""
function WENOScheme(epsilon = 1e-6)
    boundry_f = [nothing, nothing]
    return FunctionalScheme{5, 0}(weno_f, boundary_f, boundary_f, false, [epsilon], name = "WENO")
end
