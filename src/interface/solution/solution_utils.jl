# TODO: Check if the grid is uniform and use the supported higher order interpolations

function build_interpolation(umap, ivs, ivgrid)
    return Dict(map(collect(keys(umap))) do k
        args = arguments(k.val)
        nodes = (map(args) do arg
                i = findfirst(isequal(arg), ivs)
                @assert i !== nothing "Independent variable $arg not found in ivs $ivs."
                collect(ivgrid[i])
            end...,)
        k => interpolate(nodes, umap[k], Gridded(Linear()))
    end)
end

sym_to_index(sym, syms) = findfirst(isequal(sym), syms)
