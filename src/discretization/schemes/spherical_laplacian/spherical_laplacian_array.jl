
########################################################################################
# Stencil interface
########################################################################################
#TODO: Decouple this from old discretization interface
function spherical_diffusion(innerexpr, interior, derivweights, s, bs, depvars, r, u)
    interior = get_interior(u, s, interior)
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    D_1 = derivweights.map[Differential(r)]
    D_2 = derivweights.map[Differential(r)^2]

    j = x2i(s, u, r)

    valmaps = arrayvalmaps(s, u, depvars, interior)

    exprarr = broadcast_substitute(innerexpr, valmaps)

    D_1_u = central_difference(D_1, interior, s, bs, (j, r), u, s.discvars[u])
    rarr = broadcast_substitute(Num(r), valmaps)

    out = exprarr .* (D_1_u ./ rarr) .+
          cartesian_nonlinear_laplacian(innerexpr, interior, derivweights, s, bs, depvars, r, u)

    # Catch the r ≈ 0 case
    ks = findall(_r -> _r ≈ 0, s.grid[r])
    if length(ks) > 0
        r0deriv = 3exprarr .* central_difference(D_2, interior, s, bs, (j, r), u, s.discvars[u])
        r0pairs = map(ks) do k
            op = selectdim(r0deriv, j, k)
            prepare_boundary_op((op, k), interior, j)
        end
        return Construct_ArrayMaker(interior, vcat(Tuple(interior) => out, r0pairs))
    else
        return out
    end

end

@inline function generate_spherical_diffusion_rules(interior, s::DiscreteSpace, depvars, derivweights::DifferentialDiscretizer, bcmap, indexmap, terms)
    rules = reduce(vcat,
                   [vec([@rule *(~~a, 1 / (r^2),
                                 ($(Differential(r))(*(~~c, (r^2),
                                 ~~d,
                                 $(Differential(r))(u), ~~e))),
                                 ~~b) => *(~a...,
                                           spherical_diffusion(*(~c..., ~d..., ~e..., Num(1)),
                                                               interior, derivweights, s,
                                                               bcmap[operation(u)][r], depvars,
                                                               r, u),
                                            ~b...)
                         for r in params(u, s)])
                    for u in depvars],
                   init = [])

    rules = vcat(rules, reduce(vcat, [vec([@rule /(*(~~a, $(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e)), ~~b), (r^2)) => *(~a..., ~b..., spherical_diffusion(*(~c..., ~d..., ~e..., Num(1)), interior, derivweights, s, bcmap[operation(u)][r], depvars, r, u))
                                           for r in params(u, s)]) for u in depvars], init = []))

    rules = vcat(rules, reduce(vcat, [vec([@rule /(($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))), (r^2)) => spherical_diffusion(*(~c..., ~d..., ~e..., Num(1)), interior, derivweights, s, bcmap[operation(u)][r], depvars, r, u)
                                           for r in params(u, s)]) for u in depvars], init = []))

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
