abstract type AbstractGrid end

struct CenterAlignedGrid <: AbstractGrid 
end

struct EdgeAlignedGrid <: AbstractGrid 
end

struct StaggeredGrid <: AbstractGrid
end

const center_align=CenterAlignedGrid()
const edge_align=EdgeAlignedGrid()
const stagger_align=StaggeredGrid()


abstract type AbstractVarAlign end

struct CenterAlignedVar <: AbstractVarAlign
end

struct StaggeredVar <: AbstractVarAlign
end
