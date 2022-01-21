function calculate_weights_cartesian(order::Int, x0::T, xs::AbstractVector, idxs::AbstractVector) where T<:Real
    # Cartesian domain: use Fornberg
    calculate_weights(order, x0, vec(xs[idxs]))
end
function calculate_weights_spherical(order::Int, x0::T, x::AbstractVector, idxs::AbstractVector) where T<:Real
    # Spherical domain: see #367
    # https://web.mit.edu/braatzgroup/analysis_of_finite_difference_discretization_schemes_for_diffusion_in_spheres_with_variable_diffusivity.pdf
    # Only order 2 is implemented
    @assert order == 2
    # Only 2nd order discretization is implemented
    # We can't activate this assertion for now because the rules try to create the spherical Laplacian
    # before checking whether there is a spherical Laplacian
    # this could be fixed by dispatching on domain type when we have different domain types
    # but for now everything is an Interval
    # @assert length(x) == 3
    # TODO: nonlinear diffusion in a spherical domain
    i = idxs[2]
    dx1 = x[i] - x[i-1]
    dx2 = x[i+1] - x[i]
    i0 = i - 1 # indexing starts at 0 in the paper and starts at 1 in julia
    1 / (i0 * dx1 * dx2) * [i0-1, -2i0, i0+1]
end

function generate_finite_difference_rules(II, s, pde, discretization)
    approx_order = discretization.centered_order
    I1 = oneunit(first(s.Igrid))
    Imin(order) = first(s.Igrid) + I1 * (order ÷ 2)
    Imax(order) = last(s.Igrid) - I1 * (order ÷ 2)
    stencil(j, order) = CartesianIndices(Tuple(map(x -> -x:x, (1:length(s.x̄) .== j) * (order ÷ 2))))
    # Use max and min to apply buffers
    central_neighbor_idxs(II,j,order) = stencil(j,order) .+ max(Imin(order),min(II,Imax(order)))
    central_weights_cartesian(d_order,II,j) = calculate_weights_cartesian(d_order, s.grid[j][II[j]], s.grid[j], vec(map(i->i[j],central_neighbor_idxs(II,j,approx_order))))
    central_deriv(d_order, II,j,k) = dot(central_weights(d_order, II,j),s.discvars[k][central_neighbor_idxs(II,j,approx_order)])

    central_deriv_cartesian(d_order,II,j,k) = dot(central_weights_cartesian(d_order,II,j),s.discvars[k][central_neighbor_idxs(II,j,approx_order)])

    # spherical Laplacian has a hardcoded order of 2 (only 2nd order is implemented)
    # both for derivative order and discretization order
    central_weights_spherical(II,j) = calculate_weights_spherical(2, s.grid[j][II[j]], s.grid[j], vec(map(i->i[j], central_neighbor_idxs(II,j,2))))
    central_deriv_spherical(II,j,k) = dot(central_weights_spherical(II,j),s.discvars[k][central_neighbor_idxs(II,j,2)])

    # get a sorted list derivative order such that highest order is first. This is useful when substituting rules
    # starting from highest to lowest order.
    d_orders(order) = reverse(sort(collect(union(differential_order(pde.rhs, order), differential_order(pde.lhs, order)))))

    # central_deriv_rules = [(Differential(s)^2)(u) => central_deriv(2,II,j,k) for (j,s) in enumerate(s.x̄), (k,u) in enumerate(ū)]
    central_deriv_rules_cartesian = Array{Pair{Num,Num},1}()
    for (j,x) in enumerate(s.x̄)
        rs = [(Differential(x)^d)(u) => central_deriv_cartesian(d,II,j,k) for d in d_orders(x), (k,u) in enumerate(ū)]
        for r in rs
            push!(central_deriv_rules_cartesian, r)
        end
    end

    central_deriv_rules_spherical = [Differential(x)(x^2*Differential(x)(u))/x^2 => central_deriv_spherical(II,j,k)
                                    for (j,x) in enumerate(s.x̄), (k,u) in enumerate(ū)]

    valrules = vcat([ū[k] => s.discvars[k][II] for k in 1:length(ū)],
                    [s.x̄[j] => s.grid[j][II[j]] for j in 1:nparams(s)])

    # TODO: upwind rules needs interpolation into `@rule`
    forward_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j],II[j]+1]])
    reverse_weights(II,j) = calculate_weights(discretization.upwind_order, 0.0, s.grid[j][[II[j]-1,II[j]]])
    # upwinding_rules = [@rule(*(~~a,$(Differential(s.x̄[j]))(u),~~b) => IfElse.ifelse(*(~~a..., ~~b...,)>0,
    #                         *(~~a..., ~~b..., dot(reverse_weights(II,j),ū[k][central_neighbor_idxs(II,j)[1:2]])),
    #                         *(~~a..., ~~b..., dot(forward_weights(II,j),ū[k][central_neighbor_idxs(II,j)[2:3]]))))
    #                         for j in 1:nparams(s), k in 1:length(pdesys.ū)]

    ## Discretization of non-linear laplacian.
    # d/dx( a du/dx ) ~ (a(x+1/2) * (u[i+1] - u[i]) - a(x-1/2) * (u[i] - u[i-1]) / dx^2
    reverse_finite_difference(II, j, k) = -dot(reverse_weights(II, j), s.discvars[k][central_neighbor_idxs(II, j, approx_order)[1:2]]) / s.dxs[j]
    forward_finite_difference(II, j, k) = dot(forward_weights(II, j), s.discvars[k][central_neighbor_idxs(II, j, approx_order)[2:3]]) / s.dxs[j]
    # TODO: improve interpolation of g(x) = u(x) for calculating u(x+-dx/2)
    interpolate_discrete_depvar(II, s, j, k, l) = sum([s.discvars[k][central_neighbor_idxs(II, j, approx_order)][i] for i in (l == 1 ? [2,3] : [1,2])]) / 2.
    # iv_mid returns middle space values. E.g. x(i-1/2) or y(i+1/2).
    interpolate_discrete_indvar(II, j, l) = (s.grid[j][II[j]] + s.grid[j][II[j]+l]) / 2.0
    # Dependent variable rules
    map_vars_to_discrete(II, j, k, l) = [ū[k] => interpolate_discrete_depvar(II, s, j, k, l) for k in 1:length(ū)]
    # Independent variable rules
    map_params_to_discrete(II, j, l) = [s.x̄[j] => interpolate_discrete_indvar(II, j, l) for j in 1:length(s.x̄)]
    # Replacement rules: new approach
    # Calc
    function discrete_cartesian(expr, j, k)
        disc_downwind = Num(substitute(substitute(expr, map_vars_to_discrete(II, j, k, -1)), map_params_to_discrete(II, j, -1)))
        disc_upwind = Num(substitute(substitute(expr, map_vars_to_discrete(II, j, k, 1)), map_params_to_discrete(II, j, 1)))
        return dot([disc_downwind, disc_upwind], 
                   [reverse_finite_difference(II, j, k), forward_finite_difference(II, j, k)])
    end

    cartesian_deriv_rules = [@rule ($(Differential(iv))(*(~~a, $(Differential(iv))(dv), ~~b))) => discrete_cartesian(*(a..., b...),j,k) for (j, iv) in enumerate(s.x̄) for (k, dv) in enumerate(ū)]

    cartesian_deriv_rules = vcat(vec(cartesian_deriv_rules),vec(
                            [@rule ($(Differential(iv))($(Differential(iv))(dv)/~a)) =>
                            discrete_cartesian(1/~a,j,k)
                            for (j, iv) in enumerate(s.x̄) for (k, dv) in enumerate(ū)]))

    spherical_deriv_rules = [@rule *(~~a, ($(Differential(iv))((iv^2)*$(Differential(iv))(dv))), ~~b) / (iv^2) =>
                                *(~a..., central_deriv_spherical(II, j, k), ~b...)
                                for (j, iv) in enumerate(s.x̄) for (k, dv) in enumerate(ū)]

    # r^-2 needs to be handled separately
    spherical_deriv_rules = vcat(vec(spherical_deriv_rules),vec(
                            [@rule *(~~a, (iv^-2) * ($(Differential(iv))((iv^2)*$(Differential(iv))(dv))), ~~b) =>
                                *(~a..., central_deriv_spherical(II, j, k), ~b...)
                                for (j, iv) in enumerate(s.x̄) for (k, dv) in enumerate(ū)]))

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