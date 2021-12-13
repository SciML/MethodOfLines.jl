abstract type AbstractEquation end

struct PartialDifferentialEquation <: AbstractEquation
    lhs::Term
    rhs::Term
    PartialDifferentialEquation(eq) = new(eq.lhs, eq.rhs)
end

struct BoundaryValueEquation <: AbstractEquation
    lhs::Term
    rhs::Term
    BoundaryValueEquation(eq) = new(eq.lhs, eq.rhs)
end

# function Base.getproperty(pde::T, name::Symbol) where T <: AbstractEquation
#     @show name
#     if name == :lhs
#         return (pde.eq).lhs
#     elseif name == :rhs
#         return (pde.eq).rhs
#     else

#     end
# end

ModelingToolkit.substitute(pair, pde::PartialDifferentialEquation) = PartialDifferentialEquation(susbtitute(pde.lhs ~ pde.rhs, pair))

ModelingToolkit.substitute(pair, bc::BoundaryValueEquation) = BoundaryValueEquation(susbtitute(bc.lhs ~ bc.rhs, pair))

unwrap(eq::AbstractEquation) = eq.lhs ~ eq.rhs
unwrap(eqs::AbstractArray{T}) where {T<:AbstractEquation} = map(eq -> unwrap(eq), eqs)