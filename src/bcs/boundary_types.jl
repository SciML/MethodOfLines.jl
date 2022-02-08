### INITIAL AND BOUNDARY CONDITIONS ###

abstract type AbstractBoundary{u, x} end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundary{u, x} <: AbstractTruncatingBoundary
end

struct UpperBoundary{u, x} <: AbstractTruncatingBoundary
end

struct PeriodicBoundary{u, x} <: AbstractBoundary
end

struct BoundaryHandler{hasperiodic}
    boundaries::Dict{Num, AbstractBoundary}
end

# indexes for Iedge depending on boundary type
idx(::LowerBoundary) = 1
idx(::UpperBoundary) = 2
idx(::PeriodicBoundary) = 1