### INITIAL AND BOUNDARY CONDITIONS ###

abstract type AbstractBoundary end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundary <: AbstractTruncatingBoundary
    u
    x
end

struct UpperBoundary <: AbstractTruncatingBoundary
    u
    x
end

struct PeriodicBoundary <: AbstractBoundary
    u
    x
end

getvars(b::AbstractBoundary) = (b.u, b.x)

struct BoundaryHandler{hasperiodic}
    boundaries::Dict{Num, AbstractBoundary}
end

# Which interior end to remove
whichboundary(::LowerBoundary) = (1, 0)
whichboundary(::UpperBoundary) = (0, 1)
whichboundary(::PeriodicBoundary) = (1, 0)

@inline function clip_interior(lower, upper, b::AbstractBoundary, x2i)
    clip = whichboundary(b)
    dim = x2i[b.x]

    lower[dim] = lower[dim] + clip[1]
    upper[dim] = upper[dim] - clip[2]
end


# indexes for Iedge depending on boundary type
isupper(::LowerBoundary) = false
isupper(::UpperBoundary) = true
isupper(::PeriodicBoundary) = false