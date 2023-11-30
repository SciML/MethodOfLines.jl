"""
`MOLMetadata`

A type used to store data about a PDESystem, and how it was discretized by MethodOfLines.jl.
Used to unpack the solution.

- `discretespace`: a DiscreteSpace object, used in the discretization.
- `disc`: a Discretization object, used in the discretization. Usually a
          MOLFiniteDifference object.
- `pdesys`: a PDESystem object, used in the discretization.
"""
struct MOLMetadata{hasTime,Ds,Disc,PDE,M,C,Strat} <: SciMLBase.AbstractDiscretizationMetadata{hasTime}
    discretespace::Ds
    disc::Disc
    pdesys::PDE
    use_ODAE::Bool
    metadata::M
    complexmap::C
    function MOLMetadata(discretespace, disc, pdesys, boundarymap, complexmap, metadata=nothing)
        metaref = Ref{Any}()
        metaref[] = metadata
        if discretespace.time isa Nothing
            hasTime = Val(false)
        else
            hasTime = Val(true)
        end
        use_ODAE = disc.use_ODAE
        if use_ODAE
            bcivmap = reduce((d1, d2) -> mergewith(vcat, d1, d2), collect(values(boundarymap)))
            allbcs = let v = discretespace.vars
                mapreduce(x -> bcivmap[x], vcat, v.x̄)
            end
            if all(bc -> bc.order > 0, allbcs)
                use_ODAE = false
            end
        end
        return new{hasTime,typeof(discretespace),
            typeof(disc),typeof(pdesys),
            typeof(metaref),typeof(complexmap),typeof(disc.disc_strategy)}(discretespace,
            disc, pdesys, use_ODAE,
            metaref, complexmap)
    end
end

function PDEBase.generate_metadata(s::DiscreteSpace, disc::MOLFiniteDifference, pdesys::PDESystem, boundarymap, complexmap, metadata=nothing)
    @show complexmap
    return MOLMetadata(s, disc, pdesys, boundarymap, complexmap, metadata)
end
