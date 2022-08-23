"""
$(TYPEDEF)

A type used to store data about a PDESystem, and how it was discretized by MethodOfLines.jl.
Used to unpack the solution.

- `discretespace`: a DiscreteSpace object, used in the discretization.
- `disc`: a Discretization object, used in the discretization. Usually a
          MOLFiniteDifference object.
- `pdesys`: a PDESystem object, used in the discretization.
"""
struct MOLMetadata{Ds,Disc,PDE} <: SciMLBase.AbstractDiscretizationMetadata
    discretespace::Ds
    disc::Disc
    pdesys::PDE
end

function MOLMetadata(discretespace, disc, pdesys)
    return MOLMetadata{typeof(discretespace),typeof(disc),typeof(pdesys)}(discretespace,
        disc, pdesys)
end

PDESolution(sol, metadata::MOLMetadata) = metadata.discretespace.time isa Nothing ? PDENoTimeSolution(sol, metadata) : PDETimeSeriesSolution(sol, metadata)
