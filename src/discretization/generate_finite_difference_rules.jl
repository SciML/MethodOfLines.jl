"""
`interpolate_discrete_param`

Interpolate gridpoints by taking the average of the values of the discrete points, or if the offset is outside the grid, extrapolate the value with dx.
"""
@inline function interpolate_discrete_param(i, s, itap, x, bpc)
    if i+itap > (length(s, x) - bpc)
        return s.grid[x][i+itap]-s.dxs[x]*.5        

    else
        return s.grid[x][i+itap]+s.dxs[x]*.5        
    end   
end

"""
`cartesian_nonlinear_laplacian`

Differential(x)(expr(x)*Differential(x)(u(x)))

Given an internal multiplying expression `expr`, return the correct finite difference equation for the nonlinear laplacian at the location in the grid given by `II`.

The inner derivative is discretized with the half offset centered scheme, giving the derivative at interpolated grid points offset by dx/2 from the regular grid. 

The outer derivative is discretized with the centered scheme, giving the nonlinear laplacian at the grid point `II`.
For first order returns something like this:
`d/dx( a du/dx ) ~ (a(x+1/2) * (u[i+1] - u[i]) - a(x-1/2) * (u[i] - u[i-1]) / dx^2`

For 4th order, returns something like this:
```
first_finite_diffs = [a(x-3/2)*finitediff(u, i-3/2), 
                      a(x-1/2)*finitediff(u, i-1/2), 
                      a(x+1/2)*finitediff(u, i+1/2), 
                      a(x+3/2)*finitediff(u, i+3/2)]

dot(central_finite_diff_weights, first_finite_diffs)
```

where `finitediff(u, i)` is the finite difference at the interpolated point `i` in the grid.

And so on.
"""
function cartesian_nonlinear_laplacian(expr, II, derivweights, s, x, u)
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf 
    # See scheme 1, namely the term without the 1/r dependence. See also #354 and #371 in DiffEqOperators, the previous home of this package.
    
    jx = j, x = (s.x2i[x], x)
    @assert II[j] != 1 "The nonlinear laplacian is only defined on the interior of the grid, it is unsupported in boundary conditions."
    @assert II[j] != length(s, x) "The nonlinear laplacian is only defined on the interior of the grid, it is unsupported in boundary conditions."

    D_inner = derivweights.halfoffsetmap[Differential(x)]
    inner_interpolater = derivweights.interpmap[x]

    # Get the outer weights and stencil. clip() essentially removes a point from either end of the grid, for this reason this function is only defined on the interior, not in bcs
    outerweights, outerstencil = get_half_offset_weights_and_stencil(D_inner, clip(II, s, j, D_inner.boundary_point_count), s, jx, length(s, x) - 1)

    # Get the correct weights and stencils for this II
    inner_deriv_weights_and_stencil = [get_half_offset_weights_and_stencil(D_inner, I, s, jx) for I in outerstencil]
    interp_weights_and_stencil = [get_half_offset_weights_and_stencil(inner_interpolater, I, s, jx) for I in outerstencil]

    # map variables to symbolically inerpolated/extrapolated expressions
    map_vars_to_interpolated(stencil, weights) = [v => dot(weights, s.discvars[v][stencil]) for v in s.ū]

    # Map parameters to interpolated values. Using simplistic extrapolation/interpolation for now as grids are uniform
    map_params_to_interpolated(itap) = x => interpolate_discrete_param(II[j], s, itap[j]-II[j], x, D_inner.boundary_point_count)

    # Take the inner finite difference
    inner_difference = [dot(inner_weights, s.discvars[u][inner_stencil]) for (inner_weights, inner_stencil) in inner_deriv_weights_and_stencil]
    
    # Symbolically interpolate the multiplying expression
    interpolated_expr = [Num(substitute(substitute(expr, map_vars_to_interpolated(interpstencil, interpweights)), map_params_to_interpolated.(interpstencil))) 
                        for (interpweights, interpstencil) in interp_weights_and_stencil]
 
    # multiply the inner finite difference by the interpolated expression, and finally take the outer finite difference
    return dot(outerweights, inner_difference .* interpolated_expr)
end

"""
`cartesian_nonlinear_laplacian`

Differential(x)(expr(x)*Differential(x)(u(x)))

Given an internal multiplying expression `expr`, return the correct finite difference equation for the nonlinear laplacian at the location in the grid given by `II`.

The inner derivative is discretized with the half offset centered scheme, giving the derivative at interpolated grid points offset by dx/2 from the regular grid. 

The outer derivative is discretized with the centered scheme, giving the nonlinear laplacian at the grid point `II`.
For first order returns something like this:
`d/dx( a du/dx ) ~ (a(x+1/2) * (u[i+1] - u[i]) - a(x-1/2) * (u[i] - u[i-1]) / dx^2`

For 4th order, returns something like this:
```
first_finite_diffs = [a(x-3/2)*finitediff(u, i-3/2), 
                      a(x-1/2)*finitediff(u, i-1/2), 
                      a(x+1/2)*finitediff(u, i+1/2), 
                      a(x+3/2)*finitediff(u, i+3/2)]

dot(central_finite_diff_weights, first_finite_diffs)
```

where `finitediff(u, i)` is the finite difference at the interpolated point `i` in the grid.

And so on.
"""
function spherical_diffusion(innerexpr, II, derivweights, s, r, u)
    # Based on the paper https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf 
    D_1 = derivweights.map[Differential(r)]
    D_2 = derivweights.map[Differential(r)^2]

    # What to replace parameter x with given I
    _rsubs(x, I) = x => s.grid[x][I[s.x2i[x]]]
    # Full rules for substituting parameters in the inner expression
    rsubs(I) = vcat([v => s.discvars[v][I] for v in s.ū], [_rsubs(x, I) for x in s.x̄])
    # Discretization func for u
    ufunc_u(v, I, x) = s.discvars[v][I]

    # 2nd order finite difference in u
    exprhere = Num(substitute(innerexpr, rsubs(II)))
    # Catch the r ≈ 0 case
    if Symbolics.unwrap(substitute(r, _rsubs(r, II))) ≈ 0
        D_2_u = central_difference(D_2, II, s, (s.x2i(r), r), u, ufunc_u)
        return 3exprhere*D_2_u # See appendix B of the paper
    end
    D_1_u = central_difference(D_1, II, s, (s.x2i[r], r), u, ufunc_u)
    # See scheme 1 in appendix A of the paper
    
    return exprhere*(D_1_u/Num(substitute(r, _rsubs(r, II))) + cartesian_nonlinear_laplacian(innerexpr, II, derivweights, s, r, u))
end

@inline function generate_cartesian_rules(II, s, derivweights, terms)
    central_ufunc(u, I, x) = s.discvars[u][I]
    return reduce(vcat, [[(Differential(x)^d)(u) => central_difference(derivweights.map[Differential(x)^d], II, s, (j,x), u, central_ufunc) for d in derivweights.orders[x], u in s.ū] for (j,x) in enumerate(s.x̄)])
end

@inline function generate_upwinding_rules(II, s, derivweights, terms)    
    #forward_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j],II[j]+1]])
    #reverse_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j]-1,II[j]]])
    # upwinding_rules = [@rule(*(~~a, $(Differential(s.x̄[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
    #                         *(~~a..., ~~b..., dot(reverse_weights(II,j),s.ū[k][central_neighbor_idxs(II,j)[1:2]])),
    #                         *(~~a..., ~~b..., dot(forward_weights(II,j),s.ū[k][central_neighbor_idxs(II,j)[2:3]]))))
    #                         for j in 1:nparams(s), k in 1:length(pdesys.s.ū)]

end

@inline function generate_nonlinlap_rules(II, s, derivweights, terms)
    rules = [@rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => *(~~c,cartesian_nonlinear_laplacian(*(a..., b...), II, derivweights, s, x, u), ~~d) for x in s.x̄, u in s.ū]

    rules = [@rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => cartesian_nonlinear_laplacian(*(a..., b...), II, derivweights, s, x, u) for x in s.x̄, u in s.ū]

    rules = vcat(vec(rules), vec([@rule ($(Differential(x))($(Differential(x))(u)/~a)) => cartesian_nonlinear_laplacian(1/~a, II, derivweights, s, x, u) for x in s.x̄, u in s.ū]))
    
    nonlinlap_rules = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(nonlinlap_rules, t => r(t))
            end
        end
    end
    return nonlinlap_rules
end

@inline function generate_spherical_diffusion_rules(II, s, derivweights, terms)
    rules = vec([@rule *(~~a, 1/(r^2), ($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))), ~~b) => *(~a..., spherical_diffusion(*(~c..., ~d..., ~e...), II, derivweights, s, r, u), ~b...)
            for r in s.x̄, u in s.ū])

    rules = vcat(rules, vec([@rule *(~~a, (r^2)^-2, ($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))), ~~b) => *(~a..., spherical_diffusion(*(~c..., ~d..., ~e...), II, derivweights, s, r, u), ~b...)
            for r in s.x̄, u in s.ū]))

    rules = vcat(rules, vec([@rule /(($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))), (r^2)) => spherical_diffusion(*(~c..., ~d..., ~e...), II, derivweights, s, r, u)
    for r in s.x̄, u in s.ū]))

    spherical_diffusion_rules = []
    for t in terms
        for r in rules
            if r(t) !== nothing
                push!(spherical_diffusion_rules, t => r(t))
            end
        end
    end
    return spherical_diffusion_rules    
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

##Planned special cases include:
    - Up/Downwind schemes to be used for odd ordered derivatives multiplied by a coefficient, downwinding when the coefficient is positive, and upwinding when the coefficient is negative.

Please submit an issue if you know of any special cases which impact stability or accuracy that are not implemented, with links to papers and/or code that demonstrates the special case.
"""
function generate_finite_difference_rules(II, s, pde, derivweights)

    terms = split_additive_terms(pde)

    # Standard cartesian centered difference scheme
    central_deriv_rules_cartesian = generate_cartesian_rules(II, s, derivweights, terms)

    # Nonlinear laplacian scheme
    nonlinlap_rules = generate_nonlinlap_rules(II, s, derivweights, terms)

    # Spherical diffusion scheme
    spherical_diffusion_rules = generate_spherical_diffusion_rules(II, s, derivweights, terms)
    
    return vcat(vec(spherical_diffusion_rules), vec(nonlinlap_rules), vec(central_deriv_rules_cartesian))
end

