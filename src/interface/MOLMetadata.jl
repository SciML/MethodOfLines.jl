"""
`MOLMetadata`

A type used to store data about a PDESystem, and how it was discretized by MethodOfLines.jl. Used to unpack the solution.

- `discretespace`: a DiscreteSpace object, used in the discretization.
- `disc`: a Discretization object, used in the discretization. Usually a   MOLFiniteDifference object.
"""
struct MOLMetadata{N, M, Ds, Disc, PDE}
    discretespace::Ds
    disc::Disc
    pdesys::PDE
end

function MOLMetadata(discretespace, disc, pdesys)
    return MOLMetadata{typeof(discretespace), typeof(disc), typeof(pdesys)}(discretespace, disc, pdesys)
end

struct MOLWrapper{M, Sys}
    metadata::M
    odesys::Sys
end

function MOLWrapper(metadata, system)
    return MOLWrapper{typeof(metadata), typeof(system)}(metadata, system)
end
