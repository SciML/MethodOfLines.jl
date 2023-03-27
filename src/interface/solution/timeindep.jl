function SciMLBase.PDENoTimeSolution(sol::SciMLBase.NonlinearSolution{T}, metadata::MOLMetadata) where {T}
    odesys = sol.prob.f.sys

    pdesys = metadata.pdesys
    discretespace = metadata.discretespace
    # Extract axies
    ivs = [discretespace.x̄...]
    ivgrid = ((discretespace.grid[x] for x in ivs)...,)
    # Reshape the solution to flat arrays
    umap = Dict(map(discretespace.ū) do u
        let discu = discretespace.discvars[u]
            solu = map(CartesianIndices(discu)) do I
                i = sym_to_index(discu[I], odesys.states)
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
            Num(u) => out
        end
    end)
    # Build Interpolations
    interp = build_interpolation(umap, ivs, ivgrid, sol, pdesys)

    return SciMLBase.PDENoTimeSolution{T,length(discretespace.ū),typeof(umap),typeof(metadata),
        typeof(sol),typeof(ivgrid),typeof(ivs),typeof(pdesys.dvs),typeof(sol.prob),typeof(sol.alg),
        typeof(interp), typeof(stats)}(umap, sol, ivgrid, ivs,
        pdesys.dvs, metadata, sol.prob, sol.alg,
        interp, sol.retcode, sol.stats)
end
