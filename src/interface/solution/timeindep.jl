function SciMLBase.PDENoTimeSolution(
        sol::SciMLBase.NonlinearSolution{T}, metadata::MOLMetadata) where {T}
    odesys = sol.prob.f.sys

    pdesys = metadata.pdesys
    discretespace = metadata.discretespace
    # Extract axies
    ivs = [discretespace.x̄...]
    ivgrid = ((discretespace.grid[x] for x in ivs)...,)
    dvs = discretespace.ū
    # Reshape the solution to flat arrays
    umap = mapreduce(vcat, dvs) do u
        let discu = discretespace.discvars[u]
            solu = map(CartesianIndices(discu)) do I
                i = sym_to_index(discu[I], get_unknowns(odesys))
                # Handle Observed
                if i !== nothing
                    sol.u[i]
                else
                    SciMLBase.observed(sol, discu[I])
                end
            end
            out = zeros(T, size(discu)...)
            for I in CartesianIndices(discu)
                out[I] = solu[I]
            end
            # Deal with any replaced variables
            ureplaced = get(discretespace.vars.replaced_vars, u, nothing)
            if isnothing(ureplaced)
                [Num(u) => out]
            else
                [Num(u) => out, ureplaced => out]
            end
        end
    end |> Dict
    # Build Interpolations
    interp = build_interpolation(
        umap, dvs, ivs, ivgrid, sol, pdesys, discretespace.vars.replaced_vars)

    return SciMLBase.PDENoTimeSolution{
        T, length(discretespace.ū), typeof(umap), typeof(metadata),
        typeof(sol), typeof(ivgrid), typeof(ivs), typeof(pdesys.dvs), typeof(sol.prob), typeof(sol.alg),
        typeof(interp), typeof(sol.stats)}(umap, sol, ivgrid, ivs,
        pdesys.dvs, metadata, sol.prob, sol.alg,
        interp, sol.retcode, sol.stats)
end
