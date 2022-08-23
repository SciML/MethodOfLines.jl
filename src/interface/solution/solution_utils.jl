# TODO: Check if the grid is uniform and use the supported higher order interpolations

function build_interpolation(umap, ivs, ivgrid)
    return Dict(map(keys(umap)) do k
        args = arguments(k)
        nodes = Tuple((
            map(args) do arg
                i = findfirst(arg, ivs)
                @assert i !== nothing "Independent variable $arg
                                       not found in ivs"
                ivgrid[i]
            end
        )...)
        k => interpolate(nodes, umap[k], Gridded(Linear()))
    end)
end
