# TODO: Check if the grid is uniform and use the supported higher order interpolations

function build_interpolation(
        umap, dvs, ivs, ivgrid, sol, pdesys, replaced_vars, discretespace
    )
    return mapreduce(vcat, Num.(dvs)) do k
        args = arguments(k.val)
        nodes = (
            map(args) do arg
                i = findfirst(isequal(arg), ivs)
                @assert i !== nothing "Independent variable $arg not found in ivs $ivs."
                isequal(arg, discretespace.time) ? collect(ivgrid[i]) :
                    collect(ivgrid[i]) .+ staggered_shift(discretespace, k.val, arg)
            end...,
        )
        @assert all(
            map(
                pair -> isequal(first(pair[1]), pair[2]), zip(axes.(nodes), axes(umap[k]))
            )
        ) "Please ensure that the order that your independent variables appear in your PDESystem matches the order that they appear in the argument signature of your dependent variables. For example, if you have a PDESystem with independent variables `[t, x, y]`, and a dependent variable `u(t, x, y)`, then you should use `u(t, x, y)` instead of `u(t, y, x)` in your sytstem definition.\nFound a problem with the dependent variable $k, given that your ivs are $(get_ivs(pdesys))."
        kreplaced = get(replaced_vars, k, nothing)
        if all(length.(nodes) .> 1)
            interp = interpolate(nodes, umap[k], Gridded(Linear()))
            # staggered points start half a step inside the domain edge, so evaluating on
            # the grid (`sol(t, :)`) reaches half a step past the first point
            shifted = arg -> !isequal(arg, discretespace.time) &&
                !iszero(staggered_shift(discretespace, k.val, arg))
            any(shifted, args) &&
                (interp = Interpolations.extrapolate(interp, Interpolations.Line()))
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

# function build_interpolation(umap, dvs, ivs, ivgrid, sol{G}, pdesys, replaced_vars) where {G<:StaggeredGrid}
# end

sym_to_index(sym, syms) = findfirst(isequal(sym), syms)

iscomplex(A::SciMLBase.AbstractPDESolution) = !isnothing(A.disc_data.complexmap)
