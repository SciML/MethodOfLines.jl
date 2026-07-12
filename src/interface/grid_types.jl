abstract type AbstractGrid end

struct CenterAlignedGrid <: AbstractGrid
end

struct EdgeAlignedGrid <: AbstractGrid
end

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
