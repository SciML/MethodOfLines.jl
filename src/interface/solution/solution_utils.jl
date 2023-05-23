# TODO: Check if the grid is uniform and use the supported higher order interpolations

function build_interpolation(umap, dvs, ivs, ivgrid, sol, pdesys, replaced_vars)
    return mapreduce(vcat, Num.(dvs)) do k
        args = arguments(k.val)
        nodes = (map(args) do arg
                i = findfirst(isequal(arg), ivs)
                @assert i !== nothing "Independent variable $arg not found in ivs $ivs."
                collect(ivgrid[i])
            end...,)
        @assert all(map(pair -> isequal(first(pair[1]), pair[2]), zip(axes.(nodes), axes(umap[k])))) "Please ensure that the order that your independant variables appear in your PDESystem matches the order that they appear in the argument signature of your dependent variables. For example, if you have a PDESystem with independent variables `[t, x, y]`, and a dependent variable `u(t, x, y)`, then you should use `u(t, x, y)` instead of `u(t, y, x)` in your sytstem definition.\nFound a problem with the dependent variable $k, given that your ivs are $(pdesys.ivs)."
        kreplaced = get(replaced_vars, k, nothing)
        if all(length.(nodes) .> 1)
            interp = interpolate(nodes, umap[k], Gridded(Linear()))
        else
            i = findfirst(a -> a <= 1, length.(nodes))
            @warn "Solution has length 1 in dimension $(ivs[i]). Interpolation will not be possible for variable $k. Solution return code is $(sol.retcode)."
            interp = nothing
        end
        if isnothing(kreplaced)
            [k => interp]
        else
            [k => interp, kreplaced => interp]
        end
    end |> Dict
end

sym_to_index(sym, syms) = findfirst(isequal(sym), syms)
