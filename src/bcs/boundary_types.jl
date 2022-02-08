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

# indexes for Iedge depending on boundary type
idx(::LowerBoundary) = 1
idx(::UpperBoundary) = 2
idx(::PeriodicBoundary) = 1