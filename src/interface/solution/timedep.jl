
function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.ODESolution{T}, metadata::MOLMetadata) where {T}
    try
        odesys = sol.prob.f.sys

        pdesys = metadata.pdesys
        discretespace = metadata.discretespace

        ivs = [discretespace.time, discretespace.x̄...]
        ivgrid = ((isequal(discretespace.time, x) ? sol.t : discretespace.grid[x] for x in ivs)...,)
        # Reshape the solution to flat arrays, faster to do this eagerly.
        umap = Dict(map(discretespace.ū) do u
            let discu = discretespace.discvars[u]
                solu = map(CartesianIndices(discu)) do I
                    i = sym_to_index(discu[I], odesys.states)
                    # Handle Observed
                    if i !== nothing
                        sol[i, :]
                    else
                        SciMLBase.observed(sol, discu[I], :)
                    end
                end
                out = zeros(T, length(sol.t), size(discu)...)
                for I in CartesianIndices(discu)
                    out[:, I] .= solu[I]
                end
                Num(u) => out
            end
        end)
        # Build Interpolations
        interp = build_interpolation(umap, ivs, ivgrid)

        return SciMLBase.PDETimeSeriesSolution{T,length(discretespace.ū),typeof(umap),typeof(metadata),
            typeof(sol),typeof(sol.errors),typeof(sol.t),typeof(ivgrid),
            typeof(ivs),typeof(pdesys.dvs),typeof(sol.prob),typeof(sol.alg),
            typeof(interp)}(umap, sol, sol.errors, sol.t, ivgrid, ivs,
            pdesys.dvs, metadata, sol.prob, sol.alg,
            interp, sol.dense, sol.tslocation,
            sol.retcode)
    catch e
        rethrow(e)
        return sol, e
    end
end
