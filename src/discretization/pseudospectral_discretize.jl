# `PseudospectralDiscretization` pipeline.
#
# The parsing, boundary-condition and system-construction machinery is shared
# with `MOLFiniteDifference`; the differences are confined to grid
# construction (`construct_discrete_space`), the derivative operators
# (`construct_differential_discretizer`), interior sizing (no ghost-point
# padding: spectral rows always span the whole direction, and higher-order
# periodic matching conditions are redundant), and the rule generation
# (`generate_finite_difference_rules`).

function PDEBase.interface_errors(
        pdesys::PDESystem, v::PDEBase.VariableMap, discretization::PseudospectralDiscretization
    )
    for x in v.x̄
        @assert haskey(discretization.dxs, Num(x))||haskey(discretization.dxs, x) "Variable $x has no collocation specification"
    end
    return
end

function PDEBase.check_boundarymap(
        boundarymap, v::PDEBase.VariableMap, discretization::PseudospectralDiscretization
    )
    bs = flatten_vardict(boundarymap)
    for b in bs
        if b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}
            isequal(b.x, b.x2) || throw(
                ArgumentError(
                    "PseudospectralDiscretization does not yet support multi-domain interface boundary conditions like $(b.eq), only same-variable periodic conditions are supported."
                )
            )
        end
    end
    for x in v.x̄
        spec = discretization.dxs[x]
        for uop in keys(boundarymap)
            ubs = boundarymap[uop][x]
            hasperiodic = any(
                b -> b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}, ubs
            )
            hastruncating = any(
                b -> b isa AbstractTruncatingBoundary &&
                    !(b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}), ubs
            )
            if spec isa FourierCollocation
                hasperiodic || throw(
                    ArgumentError(
                        "`FourierCollocation` for $x requires a periodic boundary condition `u(t, a) ~ u(t, b)` on every dependent variable in that direction; none was found for $uop."
                    )
                )
                hastruncating && throw(
                    ArgumentError(
                        "Periodic direction $x discretized with `FourierCollocation` does not accept additional truncating boundary conditions like $(first(filter(b -> b isa AbstractTruncatingBoundary && !(b isa Union{InterfaceBoundary, HigherOrderInterfaceBoundary}), ubs)).eq)."
                    )
                )
            else
                hasperiodic && throw(
                    ArgumentError(
                        "Direction $x has a periodic boundary condition but is discretized with $(spec isa AbstractVector ? "a custom grid" : string(typeof(spec))). Periodic directions require `x => FourierCollocation(n)`."
                    )
                )
            end
        end
    end
    return
end

function PDEBase.should_transform(
        pdesys::PDESystem, disc::PseudospectralDiscretization, boundarymap
    )
    # The collocation evaluator handles nested derivative terms like
    # `Dx(a(u) * Dx(u))` or `Dx(u^2)` natively, so the FD-oriented
    # transformation (auxiliary variables etc.) is unnecessary.
    return false
end

"""
`spectral_clip_interior!!`: higher-order periodic matching conditions are
satisfied identically by the trigonometric interpolant, so they must not
truncate the interior - the corresponding equations are skipped in
`discretize_equation!` as well.
"""
clip_interior!!(lower, upper, s, b, discretization) = clip_interior!!(lower, upper, s, b)
function clip_interior!!(
        lower, upper, s, b::HigherOrderInterfaceBoundary,
        discretization::PseudospectralDiscretization
    )
    return nothing
end

function validate_interface_orders(
        pdes, boundarymap, discretization::PseudospectralDiscretization
    )
    return nothing
end

"""
No stencil extents: spectral rows always span the entire direction, so no
ghost-point padding is ever required.
"""
function calculate_stencil_extents(
        s, u, discretization::PseudospectralDiscretization, orders, bcmap
    )
    n = length(remove(arguments(u), s.time))
    return zeros(Int, n), zeros(Int, n)
end

function PDEBase.construct_discrete_space(
        vars::PDEBase.VariableMap, discretization::PseudospectralDiscretization
    )
    x̄ = vars.x̄
    depvars = vars.ū
    nspace = length(x̄)

    axies = Dict(
        map(x̄) do x
            spec = discretization.dxs[x]
            a, b = vars.intervals[x]
            x => spectral_grid(spec, a, b)
        end
    )
    grid = axies
    dxs = Dict(x => diff(axies[x]) for x in x̄)

    Iaxies = [
        u => CartesianIndices(
            (
                (
                    axes(axies[x])[1]
                        for x in remove(arguments(u), vars.time)
                )...,
            )
        )
            for u in depvars
    ]
    depvarsdisc = discretize_dep_vars(depvars, grid, vars)

    return DiscreteSpace{nspace, length(depvars), CenterAlignedGrid}(
        vars, Dict(depvarsdisc), axies, grid, dxs, Dict(Iaxies), Dict(Iaxies), nothing
    )
end

function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace,
        derivweights::SpectralDifferentialDiscretizer, indexmap,
        discretization::PseudospectralDiscretization
    )
    boundaryvalfuncs = generate_boundary_val_funcs(
        s, depvars, bcmap, indexmap, derivweights
    )
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    for boundary in eqvarbcs
        if boundary isa HigherOrderInterfaceBoundary
            # Periodic derivative matching is automatic for the trigonometric
            # interpolant. Verify the condition is identically satisfied rather
            # than emitting a redundant equation.
            eqs = generate_bc_eqs(
                s, boundaryvalfuncs, boundary, interiormap,
                Dict(ivs(depvar(boundary.u, s), s) .=> eachindex(ivs(depvar(boundary.u, s), s)))
            )
            all(eq -> isequal(eq.lhs, eq.rhs), eqs) || throw(
                ArgumentError(
                    "Periodic boundary condition $(boundary.eq) is inconsistent with the `FourierCollocation` discretization: derivative matching is automatic for the trigonometric interpolant and cannot express a jump."
                )
            )
            continue
        end
        generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end
    generate_corner_eqs!(disc_state, s, interiormap, pde)

    interior = interiormap.I[pde]
    if isempty(interior)
        push!(
            disc_state.eqs,
            discretize_equation_at_point(
                CartesianIndex(), s, depvars, pde, derivweights,
                bcmap, eqvar, indexmap, boundaryvalfuncs
            )
        )
    else
        for II in interior
            push!(
                disc_state.eqs,
                discretize_equation_at_point(
                    II, s, depvars, pde, derivweights,
                    bcmap, eqvar, indexmap, boundaryvalfuncs
                )
            )
        end
    end
    return disc_state.eqs
end
