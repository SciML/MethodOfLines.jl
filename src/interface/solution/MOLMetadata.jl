"""
`MOLMetadata`

A type used to store data about a PDESystem, and how it was discretized by MethodOfLines.jl.
Used to unpack the solution.

- `discretespace`: a DiscreteSpace object, used in the discretization.
- `disc`: a Discretization object, used in the discretization. Usually a
          MOLFiniteDifference object.
- `pdesys`: a PDESystem object, used in the discretization.
"""
struct MOLMetadata{hasTime, Ds, Disc, PDE, M, C, Strat} <:
    SciMLBase.AbstractDiscretizationMetadata{hasTime}
    discretespace::Ds
    disc::Disc
    pdesys::PDE
    use_ODAE::Bool
    metadata::M
    complexmap::C
    u0::Vector
    function MOLMetadata(
            discretespace, disc, pdesys, boundarymap, complexmap,
            metadata = nothing, u0 = []
        )
        metaref = Ref{Any}()
        metaref[] = metadata
        if discretespace.time isa Nothing
            hasTime = Val(false)
        else
            hasTime = Val(true)
        end
        use_ODAE = disc.use_ODAE
        return new{
            hasTime, typeof(discretespace),
            typeof(disc), typeof(pdesys),
            typeof(metaref), typeof(complexmap), typeof(disc.disc_strategy),
        }(
            discretespace,
            disc, pdesys, use_ODAE,
            metaref, complexmap, u0
        )
    end
end

function PDEBase.generate_metadata(
        s::DiscreteSpace, disc::MOLFiniteDifference, pdesys::PDESystem,
        boundarymap, complexmap, u0 = []
    )
    return MOLMetadata(s, disc, pdesys, boundarymap, complexmap, nothing, u0)
end

# function PDEBase.generate_metadata(s::DiscreteSpace, disc::MOLFiniteDifference{G,D}, pdesys::PDESystem, boundarymap, metadata=nothing) where {G<:StaggeredGrid}
#     return MOLMetadata(s, disc, pdesys, boundarymap, metadata)
# end
