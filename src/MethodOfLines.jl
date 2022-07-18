module MethodOfLines
using LinearAlgebra
using SciMLBase
using DiffEqBase
using ModelingToolkit
using ModelingToolkit: operation, istree, arguments, variable
using SymbolicUtils, Symbolics
using SymbolicUtils: operation, arguments
using IfElse
using StaticArrays
import DomainSets

# Interface
include("interface/grid_types.jl")
include("interface/scheme_types.jl")
include("interface/MOLFiniteDifference.jl")

include("discretization/discretize_vars.jl")
include("MOL_utils.jl")
include("system_parsing/interior_map.jl")

# Weight calculation
include("discretization/schemes/fornberg_calculate_weights.jl")
include("discretization/derivative_operator.jl")
include("discretization/schemes/centered_difference/centered_diff_weights.jl")
include("discretization/schemes/upwind_difference/upwind_diff_weights.jl")
include("discretization/schemes/half_offset_weights.jl")

include("discretization/differential_discretizer.jl")

# System Parsing
include("system_parsing/bcs/parse_boundaries.jl")

include("system_parsing/bcs/periodic_map.jl")

# Schemes
include("discretization/schemes/centered_difference/centered_difference.jl")
include("discretization/schemes/upwind_difference/upwind_difference.jl")
include("discretization/schemes/half_offset_centred_difference.jl")
include("discretization/schemes/nonlinear_laplacian/nonlinear_laplacian.jl")
include("discretization/schemes/spherical_laplacian/spherical_laplacian.jl")

# System Discretization
include("discretization/generate_finite_difference_rules.jl")

include("discretization/generate_bc_eqs.jl")

# Main
include("error_analysis.jl")
include("MOL_discretization.jl")

export MOLFiniteDifference, discretize, symbolic_discretize, ODEFunctionExpr, generate_code, grid_align, edge_align, center_align, get_discrete

end
