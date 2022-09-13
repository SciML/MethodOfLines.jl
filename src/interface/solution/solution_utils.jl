# TODO: Check if the grid is uniform and use the supported higher order interpolations

function build_interpolation(umap, ivs, ivgrid, pdesys)
    return Dict(map(collect(keys(umap))) do k
        args = arguments(k.val)
        nodes = (map(args) do arg
                i = findfirst(isequal(arg), ivs)
                @assert i !== nothing "Independent variable $arg not found in ivs $ivs."
                collect(ivgrid[i])
            end...,)
        @assert all(map(pair -> isequal(first(pair[1]), pair[2]), zip(axes.(nodes), axes(umap[k])))) "Please ensure that the order that your independant variables appear in your PDESystem matches the order that they appear in the argument signature of your dependent variables. For example, if you have a PDESystem with independent variables `[t, x, y]`, and a dependent variable `u(t, x, y)`, then you should use `u(t, x, y)` instead of `u(t, y, x)` in your sytstem definition.\nFound a problem with the dependent variable $k, given that your ivs are $(pdesys.ivs)."
        k => interpolate(nodes, umap[k], Gridded(Linear()))
    end)
end

sym_to_index(sym, syms) = findfirst(isequal(sym), syms)
