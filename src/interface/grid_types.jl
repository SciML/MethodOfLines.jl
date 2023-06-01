abstract type AbstractGrid end

struct CenterAlignedGrid <: AbstractGrid 
end

struct EdgeAlignedGrid <: AbstractGrid 
end

struct StaggeredGrid <: AbstractGrid
end

const center_align=CenterAlignedGrid()
const edge_align=EdgeAlignedGrid()
const stagger=StaggeredGrid()
