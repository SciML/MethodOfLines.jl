using SciMLTesting, MethodOfLines, Test
using JET

run_qa(
    MethodOfLines;
    explicit_imports = true,
    reexports_allow = (:discretize, :symbolic_discretize),
    aqua_kwargs = (; persistent_tasks = (; tmax = 300)),
    ei_kwargs = (;
        no_stale_explicit_imports = (;
            # `@register_array_symbolic` expands to `@wrapped` in this module.
            ignore = (Symbol("@wrapped"),),
        ),
        all_explicit_imports_are_public = (;
            # These are PDEBase/Symbolics developer hooks required to implement the
            # discretizer extension points; their owners have not declared them public.
            ignore = (
                :error_analysis, :get_ops, :insert, :remove, :sym_dot,
                :symbolic_to_float, :unitindex, :unitindices, :update_varmap!, :vcat!,
            ),
        ),
        all_qualified_accesses_are_public = (;
            # These are required by the Base/PDEBase/SciMLBase developer interfaces;
            # their owners do not declare them public, and they are not user-facing API.
            ignore = (
                :AbstractCartesianIndex, :EquationState, :PDESolution,
                :cardinalize_eqs!, :observed, :parse_bcs,
            ),
        ),
    ),
)
