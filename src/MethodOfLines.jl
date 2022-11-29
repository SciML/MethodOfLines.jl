module MethodOfLines
using LinearAlgebra
using SciMLBase
using DiffEqBase
using ModelingToolkit
using ModelingToolkit: operation, istree, arguments, variable, get_metadata
using SymbolicUtils, Symbolics
using Symbolics: wrap, unwrap, solve_for, expand_derivatives, diff2term, setname, rename,
    similarterm, symtype, ArrayOp, ArrayMaker, scalarize
using SymbolicUtils: operation, arguments
using IfElse
using StaticArrays
using Interpolations
using Latexify
import DomainSets

# To Extend
import SciMLBase.wrap_sol
import Base.display
import Base.isequal
import Base.getindex
import Base.checkindex
import Base.checkbounds
import Base.getproperty
import Base.ndims

# Interface
include("interface/grid_types.jl")
include("interface/scheme_types.jl")
include("interface/disc_strategy_types.jl")
include("interface/MOLFiniteDifference.jl")
include("discretization/derivative_operator.jl")

# Utils
include("discretization/array_form/stencil_utils.jl")
include("MOL_utils.jl")
include("broadcast_substitute.jl")

# System Parsing
include("system_parsing/variable_map.jl")
include("system_parsing/bcs/parse_boundaries.jl")
include("system_parsing/bcs/periodic_map.jl")
include("system_parsing/pde_system_transformation.jl")

# Var Discretization and interior map
include("discretization/discretize_vars.jl")
include("system_parsing/interior_map.jl")

# Weight calculation
include("discretization/schemes/fornberg_calculate_weights.jl")
include("discretization/schemes/centered_difference/centered_diff_weights.jl")
include("discretization/schemes/upwind_difference/upwind_diff_weights.jl")
include("discretization/schemes/half_offset_weights.jl")
include("discretization/schemes/extrapolation_weights.jl")
include("discretization/differential_discretizer.jl")

# Interface handling
include("discretization/interface_boundary.jl")

# Scalarized Schemes
include("discretization/schemes/centered_difference/centered_difference.jl")
include("discretization/schemes/upwind_difference/upwind_difference.jl")
include("discretization/schemes/half_offset_centred_difference.jl")
include("discretization/schemes/nonlinear_laplacian/nonlinear_laplacian.jl")
include("discretization/schemes/spherical_laplacian/spherical_laplacian.jl")
include("discretization/schemes/WENO/WENO.jl")

# Array Schemes
include("discretization/schemes/centered_difference/centered_difference_array.jl")
include("discretization/schemes/upwind_difference/upwind_difference_array.jl")
include("discretization/array_form/half_offset_centered_difference_array.jl")
#TODO: Properly seperate laplacian schemes
include("discretization/schemes/WENO/WENO_array.jl")

# System Discretization
include("discretization/generate_finite_difference_rules.jl")
include("discretization/generate_ic_defaults.jl")

# Scalarized Discretization
include("discretization/generate_bc_eqs.jl")
include("scalar_discretization.jl")

# Array Discretization
include("discretization/generate_array_bc_eqs.jl")
include("array_discretization.jl")

# Solution Interface
include("interface/solution/MOLMetadata.jl")
include("interface/solution/solution_utils.jl")
include("interface/solution/common.jl")
include("interface/solution/timedep.jl")
include("interface/solution/timeindep.jl")

# Main
include("error_analysis.jl")
include("MOL_discretization.jl")

export MOLFiniteDifference, discretize, symbolic_discretize, ODEFunctionExpr, generate_code,
    grid_align, edge_align, center_align, get_discrete
export UpwindScheme, WENOScheme

end
