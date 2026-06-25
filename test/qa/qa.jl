using SciMLTesting, MethodOfLines, Test
using JET

# Aqua/JET/ExplicitImports findings tracked at
# https://github.com/SciML/MethodOfLines.jl/issues/574
run_qa(
    MethodOfLines;
    explicit_imports = true,
    # Aqua sub-checks with genuine findings, marked broken pending fixes (issue #574):
    #   :ambiguities       — 48 method ambiguities involving MethodOfLines methods
    #   :undefined_exports — MethodOfLines.grid_align (a kwarg/field name, not a binding)
    #   :stale_deps        — OrdinaryDiffEq declared but not loaded (uses split subpackages)
    #   :deps_compat       — LinearAlgebra has no [compat] entry
    #   :piracies          — call-operator methods on PDEBase.InterfaceBoundary /
    #                        PDEBase.AbstractBoundary in interface_boundary.jl
    aqua_broken = (:ambiguities, :undefined_exports, :stale_deps, :deps_compat, :piracies),
    # JET: the 10 possible errors previously tracked in #574 no longer reproduce on the
    # current ModelingToolkit 11 / SciMLBase 3 stack, so JET.test_package runs as a hard
    # check (jet_broken left at its default of false).
    # ExplicitImports per-check ignore-lists (names owned by / non-public in other
    # packages; they become clean once those packages export/declare-public these names).
    ei_kwargs = (;
        all_explicit_imports_via_owners = (;
            ignore = (
                :Interval,                       # owner IntervalSets, re-exported by PDEBase
                :get_bcs, :get_dvs, :get_eqs,    # owner ModelingToolkitBase, re-exported by ModelingToolkit
                :get_ivs, :get_unknowns,
                :unwrap,                         # owner SymbolicUtils, re-exported by Symbolics
            ),
        ),
        all_qualified_accesses_via_owners = (;
            ignore = (
                :ProblemTypeCtx,                 # owner ModelingToolkitBase, accessed via ModelingToolkit
                :unwrap,                         # owner SymbolicUtils, accessed via Symbolics
            ),
        ),
        all_explicit_imports_are_public = (;
            ignore = (
                # non-public in PDEBase:
                :Interval, :add_metadata!, :error_analysis, :get_ops, :insert, :remove,
                :sym_dot, :unitindex, :unitindices, :update_varmap!, :vcat!,
                # non-public in Symbolics:
                :diff2term, :unwrap,
                # non-public in ModelingToolkit:
                :get_bcs, :get_dvs, :get_eqs, :get_ivs, :get_unknowns,
            ),
        ),
        all_qualified_accesses_are_public = (;
            ignore = (
                # non-public in Base:
                :var"@propagate_inbounds", :AbstractCartesianIndex, :OneTo,
                # non-public in SciMLBase:
                :AbstractDiscretizationMetadata, :AbstractODESolution, :AbstractPDESolution,
                :PDESolution, :observed,
                # non-public in PDEBase:
                :EquationState, :cardinalize_eqs!, :check_boundarymap,
                :construct_differential_discretizer, :construct_disc_state,
                :construct_discrete_space, :construct_var_equation_mapping,
                :discretize_equation!, :generate_ic_defaults, :generate_metadata,
                :get_discvars, :get_eqvar, :interface_errors, :parse_bcs,
                :should_transform, :transform_pde_system!,
                # non-public in ModelingToolkit:
                :ProblemTypeCtx,
                # non-public in IfElse:
                :ifelse,
                # non-public in RuntimeGeneratedFunctions:
                :init,
                # non-public in Symbolics:
                :unwrap, :variable,
            ),
        ),
    ),
    # no_implicit_imports: ~80 names reach MethodOfLines via `using <BigDep>` (SciMLBase,
    # DiffEqBase, ModelingToolkit, Symbolics, SymbolicUtils, PDEBase, DomainSets,
    # Interpolations, ...). Making them all explicit is a large, risky refactor; kept
    # broken pending issue #574.
    ei_broken = (:no_implicit_imports,),
)
