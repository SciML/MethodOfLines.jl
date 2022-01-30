### INITIAL AND BOUNDARY CONDITIONS ###

abstract type AbstractBoundary end

abstract type AbstractTruncatingBoundary <: AbstractBoundary end

abstract type AbstractExtendingBoundary <: AbstractBoundary end

struct LowerBoundary <: AbstractTruncatingBoundary
end

struct UpperBoundary<: AbstractTruncatingBoundary
end

struct CompleteBoundary <: AbstractTruncatingBoundary
end

struct PeriodicBoundary <: AbstractBoundary
end

struct BoundaryHandler{hasperiodic}
    boundaries::Dict{Num, AbstractBoundary}
end