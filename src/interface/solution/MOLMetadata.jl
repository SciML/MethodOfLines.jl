"""
`MOLMetadata`

A type used to store data about a PDESystem, and how it was discretized by MethodOfLines.jl.
Used to unpack the solution.

- `discretespace`: a DiscreteSpace object, used in the discretization.
- `disc`: a Discretization object, used in the discretization. Usually a
          MOLFiniteDifference object.
- `pdesys`: a PDESystem object, used in the discretization.
"""
struct MOLMetadata{hasTime, Ds, Disc, PDE, M, C, Strat, U0} <:
    SciMLBase.AbstractDiscretizationMetadata{hasTime}
    discretespace::Ds
    disc::Disc
    pdesys::PDE
    use_ODAE::Bool
    metadata::M
    complexmap::C
    u0::U0
    function MOLMetadata(
            discretespace, disc, pdesys, boundarymap, complexmap, u0 = nothing
        )
        metaref = Ref{Any}()
        metaref[] = nothing
        if discretespace.time isa Nothing
            hasTime = Val(false)
        else
            hasTime = Val(true)
        end
        use_ODAE = disc.use_ODAE
        if use_ODAE
            bcivmap = reduce(
                (d1, d2) -> mergewith(vcat, d1, d2), collect(values(boundarymap))
            )
            allbcs = let v = discretespace.vars
                mapreduce(x -> bcivmap[x], vcat, v.x̄)
            end
            if all(bc -> bc.order > 0, allbcs)
                use_ODAE = false
            end
        end
        return new{
            hasTime, typeof(discretespace),
            typeof(disc), typeof(pdesys),
            typeof(metaref), typeof(complexmap), typeof(disc.disc_strategy),
            typeof(u0),
        }(
            discretespace,
            disc, pdesys, use_ODAE,
            metaref, complexmap, u0
        )
    end
end

function PDEBase.generate_metadata(
        s::DiscreteSpace, disc::MOLFiniteDifference, pdesys::PDESystem,
        boundarymap, complexmap, u0 = nothing
    )
    return MOLMetadata(s, disc, pdesys, boundarymap, complexmap, u0)
end

# PDEBase's discretize function checks hasproperty(metadata, :u0) to retrieve IC defaults.
# The u0 data is stored in a dedicated field, separate from the generic `metadata` Ref
# (which gets overwritten by add_metadata! before u0 is accessed).
function Base.hasproperty(m::MOLMetadata, s::Symbol)
    if s === :u0
        return getfield(m, :u0) !== nothing
    else
        return hasfield(typeof(m), s)
    end
end

# function PDEBase.generate_metadata(s::DiscreteSpace, disc::MOLFiniteDifference{G,D}, pdesys::PDESystem, boundarymap, metadata=nothing) where {G<:StaggeredGrid}
#     return MOLMetadata(s, disc, pdesys, boundarymap, metadata)
# end
