module MethodOfLines
import LinearAlgebra
using LinearAlgebra: I, cond, dot
import SciMLBase
using SciMLBase: DAEProblem, NonlinearProblem, ODEFunction, ODEProblem, SplitODEProblem
import DiffEqBase
import ModelingToolkit
using ModelingToolkit: get_unknowns,
    get_eqs, get_bcs, get_dvs,
    get_ivs
import ModelingToolkitBase
using ModelingToolkitBase: @named, @parameters, PDESystem, complete, initialization_equations,
    mtkcompile, unknowns
import SymbolicIndexingInterface
using SymbolicIndexingInterface: NotSymbolic, symbolic_type
import SymbolicUtils
using SymbolicUtils: @rule, hasmetadata, setmetadata, substitute, term
import Symbolics
using Symbolics: @variables, @wrapped, Differential, Equation, Integral, Num, terms
using Symbolics: unwrap, symbolic_linear_solve, expand_derivatives, diff2term,
    symbolic_to_float
using SymbolicUtils: operation, arguments, iscall, getmetadata, unwrap_const
import StaticArrays
using StaticArrays: SVector
import Interpolations
using Interpolations: Gridded, Linear, interpolate
import Latexify
using Latexify: latexify
import PrecompileTools
using PrecompileTools: @compile_workload, @setup_workload
import DomainSets
using DomainSets: boundary, interior
import RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# See here for the main `symbolic_discretize` and `generate_system` functions
import PDEBase
using PDEBase: AbstractBoundary, AbstractCartesianDiscreteSpace,
    AbstractDifferentialDiscretizer, AbstractEquationSystemDiscretization,
    AbstractTruncatingBoundary, AbstractVarEqMapping, HigherOrderInterfaceBoundary,
    InterfaceBoundary, LowerBoundary, UpperBoundary, all_ivs, depvars, ex2term,
    filter_interfaces, flatten_vardict, get_depvars, get_time, getvars,
    has_derivatives, has_interfaces, haslowerupper, isupper, pde_substitute,
    safe_unwrap, split_additive_terms, split_terms, subs_alleqs!, subsmatch
using PDEBase: unitindices, unitindex, remove, insert, sym_dot, VariableMap, depvar, x2i,
    d_orders, vcat!, update_varmap!, get_ops

# staggered changes
using DomainSets: Interval
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
import SciMLBase.symbolic_discretize

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
include("discretization/schemes/WENO/nonuniform_weno.jl")
include("discretization/schemes/WENO/WENO.jl")
include("discretization/schemes/integral_expansion/integral_expansion.jl")

# System Discretization
include("discretization/generate_finite_difference_rules.jl")
include("discretization/generate_bc_eqs.jl")
include("discretization/generate_ic_defaults.jl")
include("discretization/staggered_discretize.jl")

# Main
include("discretization/discretize_equations.jl")
include("dae_discretization.jl")
include("MOL_discretization.jl")

## PrecompileTools
include("precompile.jl")

# Export
export MOLFiniteDifference, discretize, symbolic_discretize, ODEFunctionExpr, generate_code,
    edge_align, center_align, get_discrete, chebyspace
export UpwindScheme, WENOScheme, FunctionalScheme, MOLDiscCallback

end
