
"""
`spherical_diffusion`

for terms of the form `r^-2*Dr(r^2*Dr(u(t, r)))`

Based on https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf

See scheme 1 in appendix A. The r = 0 case is treated in a later appendix
"""
function spherical_diffusion(innerexpr, II, derivweights, s, indexmap, bcmap, depvars, r, u)
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    bs = filter_interfaces(bcmap[operation(u)][r])

    D_1 = derivweights.map[Differential(r)]
    D_2 = derivweights.map[Differential(r)^2]

    #TODO!: Update this to use indvars of the pde
    # What to replace parameter x with given I
    _rsubs(x, I) = x => s.grid[x][I[s.x2i[x]]]
    # Full rules for substituting parameters in the inner expression
    function rsubs(I)
        safe_vcat([v => s.discvars[v][I] for v in depvars], [_rsubs(x, I) for x in s.x̄])
    end
    # Discretization func for u
    ufunc_u(v, I, x) = s.discvars[v][I]

    # 2nd order finite difference in u
    exprhere = Num(substitute(innerexpr, rsubs(II)))
    # Catch the r ≈ 0 case
    if isapprox(Symbolics.unwrap(substitute(r, _rsubs(r, II))), 0, atol = 1e-6)
        D_2_u = central_difference(D_2, II, s, bs, (s.x2i[r], r), u, ufunc_u)
        return 6exprhere * D_2_u # See appendix B of the paper
    end
    D_1_u = central_difference(D_1, II, s, bs, (s.x2i[r], r), u, ufunc_u)
    # See scheme 1 in appendix A of the paper

    return exprhere * (D_1_u / substitute(r, _rsubs(r, II)) + cartesian_nonlinear_laplacian(
        innerexpr, II, derivweights, s, indexmap, bcmap, depvars, r, u))
end

@inline function generate_spherical_diffusion_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars,
        derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    rules = reduce(safe_vcat,
        [vec([@rule *(~~a, 1 / (r^2), ($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))), ~~b) => *(
                  ~a...,
                  spherical_diffusion(
                      *(~c..., ~d..., ~e..., Num(1)), Idx(II, s, u, indexmap),
                      derivweights, s, indexmap, bcmap, depvars, r, u),
                  ~b...)
              for r in ivs(u, s)]) for u in depvars],
        init = [])

    rules = safe_vcat(rules,
        reduce(safe_vcat,
            [vec([@rule /(*(~~a, $(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e)), ~~b), (r^2)) => *(
                      ~a...,
                      ~b...,
                      spherical_diffusion(
                          *(~c..., ~d..., ~e..., Num(1)), Idx(II, s, u, indexmap),
                          derivweights, s, indexmap, bcmap, depvars, r, u))
                  for r in ivs(u, s)]) for u in depvars],
            init = []))

    rules = safe_vcat(rules,
        reduce(safe_vcat,
            [vec([@rule /(($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))), (r^2)) => spherical_diffusion(
                      *(~c..., ~d..., ~e..., Num(1)), Idx(II, s, u, indexmap),
                      derivweights, s, indexmap, bcmap, depvars, r, u)
                  for r in ivs(u, s)]) for u in depvars],
            init = []))

    spherical_diffusion_rules = []
    for t in terms
        for r in rules
            try
                if r(t) !== nothing
                    push!(spherical_diffusion_rules, t => r(t))
                end
            catch e
                rethrow(e)
            end
        end
    end
    return spherical_diffusion_rules
end
