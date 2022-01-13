"""
`interpolate_discrete_param`

Interpolate gridpoints by taking the average of the values of the discrete points, or if the offset is outside the grid, extrapolate the value with dx.
"""
function interpolate_discrete_param(II, s, itap, j, x)
    # * This will need to be updated to dispatch on grid type when grids become more general
    offset = itap+1/2
    if (II[j]+itap) < 1
        return s.grid[x][1]+s.dxs[x]*offset
    elseif (II[j]+itap) > (length(x) -  1)
        return s.grid[x][length(x)]+s.dxs[x]*offset
    else
        return (s.grid[x][II[j]+itap]+s.grid[x][II[j]+itap+1])/2
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

    jx = j, x = (s.x2i(x), x)
    inner_interpolater, D_inner = derivweights.nonlinlapmap[x]

    # Get the outer weights and stencil to generate the required 
    outerweights, outerstencil = get_half_offset_weights_and_stencil(D_inner, II, s, 0, jx)
    # Index offsets of each stencil in the inner finite difference to get the correct stencil for each needed half grid point, 0 corresopnds to x+1/2
    itaps = getindex.(outerstencil, (j,))
    
    # Get the correct weights and stencils for this II
    inner_deriv_weights_and_stencil = [get_half_offset_weights_and_stencil(D_inner, II, s, itap, jx) for itap in itaps]
    interp_weights_and_stencil = [get_half_offset_weights_and_stencil(inner_interpolater, II, s, itap, jx) for itap in itaps]

    # map variables to symbolically inerpolated/extrapolated expressions
    map_vars_to_interpolated(stencil, weights) = [v => dot(weights, s.discvars[v][stencil]) for v in s.vars]

    # Map parameters to interpolated values. Using simplistic extrapolation/interpolation for now as grids are uniform
    map_params_to_interpolated(itap) = [z => interpolate_discrete_param(II, s, itap, i, z) for (i,z) in s.nottime]

    # Take the inner finite difference
    inner_difference = [dot(inner_weights, s.discvars[u][inner_stencil]) for (inner_weights, inner_stencil) in inner_deriv_weights_and_stencil]
    
    # Symbolically interpolate the multiplying expression
    interpolated_expr = [Num(substitute(substitute(expr, map_vars_to_interpolated(interpstencil, interpweights)), map_params_to_interpolated(itap))) 
                        for (itap,(interpweights, interpstencil)) in zip(itaps, interp_weights_and_stencil)]
 
    # multiply the inner finite difference by the interpolated expression, and finally take the outer finite difference
    return dot(weights, inner_difference .* interpolated_expr)
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
    _rsubs(x, I) = x => s.grid[x][I[s.x2i(x)]]
    # Full rules for substituting parameters in the inner expression
    rsubs(I) = vcat([v => s.discvars[v][I] for v in s.vars], [_rsubs(x, I) for x in params(s)])
    # Discretization func for u
    ufunc_u(v, I, x) = s.discvars[v][I]

    # 2nd order finite difference in u
    exprhere = substitute(innerexpr, rsubs(II))
    # Catch the r ≈ 0 case
    if substitute(r, _rsubs(r, II)) ≈ 0
        D_2_u = central_difference(D_2, II, s, (s.x2i(r), r), u, ufunc_u)
        return 3exprhere*D_2_u # See appendix B of the paper

    D_1_u = central_difference(D_1, II, s, (s.x2i[r], r), u, ufunc_u)
    # See scheme 1 in appendix A of the paper
    
    return exprhere*(D_1_u/substitute(r, _rsubs(r, II)) + cartesian_nonlinear_laplacian(innerexpr, II, derivweights, s, r, u))
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

Please submit an issue if you know of any special cases that are not implemented, with links to papers and/or code that demonstrates the special case.
"""
function generate_finite_difference_rules(II, s, pde, derivweights)

    valrules = vcat([u => s.discvars[u][II] for u in s.vars],
                    [x => s.grid[x][II[s.x2i[x]]] for x in params(s)])
    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.nottime), (k,u) in enumerate(s.vars)]

    central_ufunc(u, I, x) = s.discvars[u][I]
    central_deriv_rules_cartesian = Array{Pair{Num,Num},1}()
    for x in s.nottime
        j = s.x2i[x]
        rs = [(Differential(x)^d)(u) => central_difference(derivweights.map[Differential(x)^d], II, s, (j,x), u, central_ufunc) for d in derivweights.orders[x], u in s.vars]

        central_deriv_rules_cartesian = vcat(central_deriv_rules_cartesian, rs)
    end

    # TODO: upwind rules needs interpolation into `@rule`
    #forward_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j],II[j]+1]])
    #reverse_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j]-1,II[j]]])
    # upwinding_rules = [@rule(*(~~a, $(Differential(s.nottime[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
    #                         *(~~a..., ~~b..., dot(reverse_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[1:2]])),
    #                         *(~~a..., ~~b..., dot(forward_weights(II,j),s.vars[k][central_neighbor_idxs(II,j)[2:3]]))))
    #                         for j in 1:nparams(s), k in 1:length(pdesys.s.vars)]

    ## Discretization of non-linear laplacian.

    

    cartesian_deriv_rules = [@rule ($(Differential(x))(*(~~a, $(Differential(x))(u), ~~b))) => cartesian_nonlinear_laplacian(*(a..., b...), II, derivweights, s, x, u) for x in s.nottime, u in s.vars]

    cartesian_deriv_rules = vcat(vec(cartesian_deriv_rules),vec(
                            [@rule ($(Differential(x))($(Differential(x))(u)/~a)) => cartesian_nonlinear_laplacian(1/~a, II, derivweights, s, x, u) for x in s.nottime, u in s.vars]))

    spherical_deriv_rules = [@rule *(~~a, (r^-2), ($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))), ~~b) =>
            *(~a..., spherical_diffusion(*(~c..., ~d..., ~e...), II, derivweights, s, r, u), ~b...)
            for r in s.nottime, u in s.vars]

    rhs_arg = istree(pde.rhs) && (SymbolicUtils.operation(pde.rhs) == +) ? SymbolicUtils.arguments(pde.rhs) : [pde.rhs]
    lhs_arg = istree(pde.lhs) && (SymbolicUtils.operation(pde.lhs) == +) ? SymbolicUtils.arguments(pde.lhs) : [pde.lhs]
    nonlinlap_rules = []
    for t in vcat(lhs_arg,rhs_arg)
        for r in cartesian_deriv_rules
            if r(t) !== nothing
                push!(nonlinlap_rules, t => r(t))
            end
        end
        for r in spherical_deriv_rules
            if r(t) !== nothing
                push!(nonlinlap_rules, t => r(t))
            end
        end
    end

    rules = vcat(vec(nonlinlap_rules),
                vec(central_deriv_rules_cartesian),
                vec(central_deriv_rules_spherical),
                valrules)
    return rules
end

