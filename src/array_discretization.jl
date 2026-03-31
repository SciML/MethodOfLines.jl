"""
    _to_scalar_interiormap(interiormap)

Convert an `InteriorMap` whose `.I` dict stores `[(lo, hi), …]` tuples
(as produced by `generate_interior` for `ArrayDiscretization`) into one
that stores `CartesianIndices` so that the existing boundary-equation code
can consume it unchanged.
"""
function _to_scalar_interiormap(interiormap)
    scalar_I = Dict()
    for (pde, ranges) in interiormap.I
        if ranges isa AbstractVector && !isempty(ranges) && first(ranges) isa Tuple
            cart_ranges = Tuple(r[1]:r[2] for r in ranges)
            scalar_I[pde] = CartesianIndices(cart_ranges)
        else
            scalar_I[pde] = ranges
        end
    end
    return InteriorMap(
        interiormap.var, interiormap.pde, scalar_I,
        interiormap.lower, interiormap.upper, interiormap.stencil_extents
    )
end

"""
    discretize_equation!(disc_state, pde, interiormap, eqvar, bcmap, depvars,
                         s, derivweights, indexmap,
                         discretization::MOLFiniteDifference{G, D <: ArrayDiscretization})

Array-based discretisation of a single PDE.

Boundary equations (BCs, extrapolation, corners) are handled identically to the
scalar path (using a CartesianIndices-based interior map).  Interior equations
are generated via a *template-instantiation* strategy: substitution rules are
built once with a symbolic index variable and the PDE is symbolically
transformed once, then the resulting template is instantiated at each interior
grid point.
"""
function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
        discretization::MOLFiniteDifference{G, D}
    ) where {G, D <: ArrayDiscretization}

    # Convert tuple-range interior map to CartesianIndices for boundary code
    scalar_interiormap = _to_scalar_interiormap(interiormap)
    should_validate = discretization.disc_strategy.validate

    # ── boundary handling (uses scalar-compatible interior map) ───────────────
    boundaryvalfuncs = generate_boundary_val_funcs(
        s, depvars, bcmap, indexmap, derivweights
    )
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    for boundary in eqvarbcs
        if boundary isa InterfaceBoundary
            generate_bc_eqs_arrayop!(disc_state, s, boundaryvalfuncs, scalar_interiormap, boundary)
        elseif boundary isa AbstractTruncatingBoundary
            generate_bc_eqs_arrayop!(disc_state, s, boundaryvalfuncs, scalar_interiormap, boundary, derivweights; validate=should_validate)
        else
            generate_bc_eqs!(disc_state, s, boundaryvalfuncs, scalar_interiormap, boundary)
        end
    end
    generate_extrap_eqs!(
        disc_state, pde, eqvar, s, derivweights, scalar_interiormap, bcmap
    )
    generate_corner_eqs!(
        disc_state, s, scalar_interiormap, ndims(s.discvars[eqvar]), eqvar
    )

    # ── interior equations ───────────────────────────────────────────────────
    interior_ranges = interiormap.I[pde]   # [(lo, hi), …] for ArrayDiscretization

    eqs = if length(interior_ranges) == 0
        # No spatial dimensions — fall back to scalar point discretisation
        II = CartesianIndex()
        [discretize_equation_at_point(
            II, s, depvars, pde, derivweights, bcmap,
            eqvar, indexmap, boundaryvalfuncs
        )]
    else
        generate_array_interior_eqs(
            s, depvars, pde, derivweights, bcmap, eqvar,
            indexmap, boundaryvalfuncs, interior_ranges;
            validate=should_validate
        )
    end

    return vcat!(disc_state.eqs, eqs)
end
