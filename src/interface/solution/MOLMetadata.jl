"""
`MOLMetadata`

A type used to store data about a PDESystem, and how it was discretized by MethodOfLines.jl.
Used to unpack the solution.

- `discretespace`: a DiscreteSpace object, used in the discretization.
- `disc`: a Discretization object, used in the discretization. Usually a
          MOLFiniteDifference object.
- `pdesys`: a PDESystem object, used in the discretization.
"""
struct MOLMetadata{hasTime, Ds,Disc,PDE, M, Strat} <: SciMLBase.AbstractDiscretizationMetadata{hasTime}
    discretespace::Ds
    disc::Disc
    pdesys::PDE
    use_ODAE::Bool
    metadata::M
    function MOLMetadata(discretespace, disc, pdesys, use_ODAE, metadata = nothing)
        metaref = Ref{Any}()
        metaref[] = metadata
        if discretespace.time isa Nothing
            hasTime = Val(false)
        else
            hasTime = Val(true)
        end
        return new{hasTime, typeof(discretespace),
                   typeof(disc), typeof(pdesys),
                   typeof(metaref)}(discretespace,
                                                                                 disc, pdesys, use_ODAE,
                                                                                 metaref)
    end
end

function add_metadata!(meta::MOLMetadata, metadata)
    meta.metadata[] = metadata
end
