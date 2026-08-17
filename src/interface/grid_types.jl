abstract type AbstractGrid end

"""
    CenterAlignedGrid()

Grid alignment strategy that places the first and last grid points on the domain
boundaries. Use the singleton [`center_align`](@ref) as the `grid_align` value in
[`MOLFiniteDifference`](@ref).

This is a stateless marker type and has no fields.
"""
struct CenterAlignedGrid <: AbstractGrid
end

"""
    EdgeAlignedGrid()

Grid alignment strategy that places grid points half a spacing from the domain
boundaries. Use the singleton [`edge_align`](@ref) as the `grid_align` value in
[`MOLFiniteDifference`](@ref), for example when edge-centered values improve
Neumann boundary accuracy.

This is a stateless marker type and has no fields.
"""
struct EdgeAlignedGrid <: AbstractGrid
end

"""
    StaggeredGrid()

Grid alignment strategy for variables whose grid locations are staggered relative
to the primary grid. Pass an instance as `grid_align` to
[`MOLFiniteDifference`](@ref) and specify the corresponding variable alignment
when constructing the discretization.

This is a stateless marker type and has no fields.
"""
struct StaggeredGrid <: AbstractGrid
end

"""
    center_align

Grid alignment value for center-aligned finite difference grids.
"""
const center_align = CenterAlignedGrid()

"""
    edge_align

Grid alignment value for edge-aligned finite difference grids.
"""
const edge_align = EdgeAlignedGrid()
const stagger_align = StaggeredGrid()

abstract type AbstractVarAlign end

struct CenterAlignedVar <: AbstractVarAlign
end

struct EdgeAlignedVar <: AbstractVarAlign
end
