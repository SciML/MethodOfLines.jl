module MethodOfLines
using LinearAlgebra
using SciMLBase
using DiffEqBase
using ModelingToolkit
using ModelingToolkit: operation, iscall, arguments, variable, get_unknowns,
    parameters, varmap_to_vars, get_eqs, get_bcs, get_dvs,
    get_ivs
using SymbolicIndexingInterface
using SymbolicUtils, Symbolics
using Symbolics: unwrap, symbolic_linear_solve, expand_derivatives, diff2term, setname,
    rename
using SymbolicUtils: operation, arguments, getmetadata
using IfElse
using StaticArrays
using Interpolations
using Latexify
using PrecompileTools
using DomainSets
using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# See here for the main `symbolic_discretize` and `generate_system` functions
using PDEBase
using PDEBase: unitindices, unitindex, remove, insert, sym_dot, VariableMap, depvar, x2i,
    d_orders, vcat!, update_varmap!, get_ops

# staggered changes
using PDEBase: cardinalize_eqs!, make_pdesys_compatible, parse_bcs, generate_system,
    Interval
using PDEBase: error_analysis, add_metadata!

# To Extend
import PDEBase.interface_errors
import PDEBase.check_boundarymap
import PDEBase.should_transform
import PDEBase.transform_pde_system!
import PDEBase.construct_discrete_space
import PDEBase.construct_disc_state
import PDEBase.construct_var_equation_mapping
import PDEBase.construct_differential_discretizer
import PDEBase.discretize_equation!
import PDEBase.generate_ic_defaults
import PDEBase.generate_metadata
import PDEBase.pde_substitute
import PDEBase.symbolic_discretize

import PDEBase.get_time
import PDEBase.get_eqvar
import PDEBase.get_discvars
import PDEBase.depvar
import PDEBase.x2i
import Base.display
import Base.isequal
import Base.getindex
import Base.checkindex
import Base.checkbounds
import Base.getproperty
import Base.ndims

import SciMLBase.discretize

# Interface
include("interface/grid_types.jl")
include("interface/scheme_types.jl")
include("interface/callbacks.jl")
include("interface/disc_strategy_types.jl")
include("interface/MOLFiniteDifference.jl")

include("discretization/discretize_vars.jl")
include("MOL_utils.jl")
include("system_parsing/interior_map.jl")

# Solution Interface
include("interface/solution/MOLMetadata.jl")
include("interface/solution/solution_utils.jl")
include("interface/solution/common.jl")
include("interface/solution/timedep.jl")
include("interface/solution/timeindep.jl")

# Weight calculation
include("discretization/schemes/fornberg_calculate_weights.jl")
include("discretization/derivative_operator.jl")
include("discretization/schemes/centered_difference/centered_diff_weights.jl")
include("discretization/schemes/upwind_difference/upwind_diff_weights.jl")
include("discretization/schemes/half_offset_weights.jl")
include("discretization/schemes/extrapolation_weights.jl")
include("discretization/differential_discretizer.jl")
include("discretization/schemes/callbacks/callback_rules.jl")

# System Parsing
include("system_parsing/pde_system_transformation.jl")

# Interface handling
include("discretization/interface_boundary.jl")

# Schemes
include("discretization/schemes/function_scheme/function_scheme.jl")
include("discretization/schemes/centered_difference/centered_difference.jl")
include("discretization/schemes/2nd_order_mixed_deriv/2nd_order_mixed_deriv.jl")
include("discretization/schemes/upwind_difference/upwind_difference.jl")
include("discretization/schemes/half_offset_centred_difference.jl")
include("discretization/schemes/nonlinear_laplacian/nonlinear_laplacian.jl")
include("discretization/schemes/spherical_laplacian/spherical_laplacian.jl")
include("discretization/schemes/WENO/WENO.jl")
include("discretization/schemes/integral_expansion/integral_expansion.jl")

# System Discretization
include("discretization/generate_finite_difference_rules.jl")
include("discretization/generate_bc_eqs.jl")
include("discretization/generate_ic_defaults.jl")
include("discretization/staggered_discretize.jl")

# Main
include("scalar_discretization.jl")
include("MOL_discretization.jl")

## PrecompileTools - temporarily disabled for MTK v11 compatibility debugging
# include("precompile.jl")

# Export
export MOLFiniteDifference, discretize, symbolic_discretize, ODEFunctionExpr, generate_code,
    grid_align, edge_align, center_align, get_discrete, chebyspace
export UpwindScheme, WENOScheme, FunctionalScheme, MOLDiscCallback

end
