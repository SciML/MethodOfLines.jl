module MethodOfLines
    using LinearAlgebra
    using SciMLBase
    using DiffEqBase
    using DiffEqOperators
    using ModelingToolkit
    using ModelingToolkit: operation, istree, arguments, variable
    using SymbolicUtils, Symbolics
    using SymbolicUtils: operation, arguments
    using IfElse
    import DomainSets

    include("discretization/fornberg.jl")

    include("interface/grid_types.jl")
    include("interface/scheme_types.jl")
    include("interface/MOLFiniteDifference.jl")

    include("discretization/discretize_vars.jl")
    include("MOL_utils.jl")
    include("system_parsing/interior_map.jl")

    include("discretization/differential_discretizer.jl")
    include("system_parsing/bcs/boundary_types.jl")

    include("system_parsing/bcs/periodic_map.jl")

    include("discretization/generate_finite_difference_rules.jl")

    include("discretization/generate_bc_eqs.jl")

    include("error_analysis.jl")
    include("MOL_discretization.jl")

    export MOLFiniteDifference, discretize, symbolic_discretize, ODEFunctionExpr, generate_code, grid_align, edge_align, center_align, get_discrete

end
