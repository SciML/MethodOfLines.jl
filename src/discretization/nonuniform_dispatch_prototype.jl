# MethodOfLines.jl - Dispatch Infrastructure for Non-Uniform Grids

module MethodOfLinesDispatch

export get_operator_weights
export WENOScheme, UpwindScheme, NeumannBoundary, PeriodicBoundary
export UnsupportedNonUniformGridError

# 1. EXCEPTIONS
struct UnsupportedNonUniformGridError <: Exception
    operator_type::Type
    axis_name::Symbol
end

function Base.showerror(io::IO, e::UnsupportedNonUniformGridError)
    print(io, "MethodOfLines Architecture Guard: The operator [", e.operator_type, 
              "] is not implemented for non-uniform grids on axis [:", e.axis_name, "].")
end

# 2. DISPATCH TRAITS (Grid Classification)
abstract type GridType end
struct Uniform <: GridType end
struct NonUniform <: GridType end

# Resolve GridType statically based on array type
@inline GridType(::Type{<:AbstractRange}) = Uniform()
@inline GridType(::Type{<:AbstractVector}) = NonUniform()

# 3. OPERATOR HIERARCHY
abstract type AbstractOperator end
abstract type AbstractDiscretizationScheme <: AbstractOperator end
abstract type AbstractBoundaryOperator <: AbstractOperator end

struct WENOScheme <: AbstractDiscretizationScheme end
struct UpwindScheme <: AbstractDiscretizationScheme end 
struct NeumannBoundary <: AbstractBoundaryOperator end
struct PeriodicBoundary <: AbstractBoundaryOperator end 

# 4. INTERNAL ROUTING (Specialized Methods)

# Interior Schemes
@noinline function _dispatch_interior(::NonUniform, axis::Symbol, scheme::AbstractDiscretizationScheme)
    throw(UnsupportedNonUniformGridError(typeof(scheme), axis))
end
@inline _dispatch_interior(::Uniform, ::Symbol, ::AbstractDiscretizationScheme) = :legacy_uniform_engine
@inline _dispatch_interior(::NonUniform, ::Symbol, ::WENOScheme) = :dynamic_nonuniform_weno_engine

# Boundary Operators
@noinline function _dispatch_boundary(::NonUniform, axis::Symbol, op::AbstractBoundaryOperator)
    throw(UnsupportedNonUniformGridError(typeof(op), axis))
end
@inline _dispatch_boundary(::Uniform, ::Symbol, ::AbstractBoundaryOperator) = :legacy_uniform_boundary
@inline _dispatch_boundary(::NonUniform, ::Symbol, ::NeumannBoundary) = :nonuniform_fornberg_boundary

# 5. PUBLIC API
"""
    get_operator_weights(axis_name, domain, operator)

Internal API to route operator weight calculations based on grid properties.
Returns the appropriate computational engine with zero runtime overhead.
"""
@inline function get_operator_weights(axis::Symbol, domain::T, op::AbstractOperator) where {T <: AbstractVector}
    if op isa AbstractDiscretizationScheme
        return _dispatch_interior(GridType(T), axis, op)
    else
        return _dispatch_boundary(GridType(T), axis, op)
    end
end

end # module

# TEST RUNNER & PERFORMANCE VERIFICATION
using .MethodOfLinesDispatch
using BenchmarkTools

function verify_dispatch_performance()
    println("--- DISPATCH SYSTEM VERIFICATION ---")
    
    # Setup
    ux = 0.0:0.1:1.0
    ny = [0.0, 0.1, 0.5, 1.0]
    
    # 1. Logic Check
    println("Routing (Uniform X)    : ", get_operator_weights(:x, ux, WENOScheme()))
    println("Routing (Non-Uniform Y): ", get_operator_weights(:y, ny, WENOScheme()))
    
    # 2. Safety Check
    try 
        get_operator_weights(:y, ny, UpwindScheme()) 
    catch e 
        println("Safety Net Triggered   : Success")
    end

    # 3. Performance Check
    print("Performance (Target 1.2ns): ")
    @btime get_operator_weights(:y, $ny, $(WENOScheme()))
end

verify_dispatch_performance()