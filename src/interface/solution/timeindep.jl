# If the solution is already a PDENoTimeSolution, just return it unchanged.
# This handles cases where wrap_sol is called multiple times (e.g., during error handling).
function SciMLBase.PDENoTimeSolution(
        sol::SciMLBase.PDENoTimeSolution, metadata::MOLMetadata
    )
    return sol
end

function SciMLBase.PDENoTimeSolution(
        sol::SciMLBase.NonlinearSolution{T}, metadata::MOLMetadata
    ) where {T}
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
            idxs, obs, obsvals = grid_indices_and_observed(sol, odesys, discu)
            out = zeros(T, size(discu)...)
            for I in CartesianIndices(discu)
                idxs[I] === nothing || (out[I] = sol.u[idxs[I]])
            end
            for (k, I) in enumerate(obs)
                out[I] = obsvals[k]
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
        umap, dvs, ivs, ivgrid, sol, pdesys, discretespace.vars.replaced_vars,
        discretespace
    )
    pdedvs = get_dvs(pdesys)
    return SciMLBase.PDENoTimeSolution{
        T, length(discretespace.ū), typeof(umap), typeof(metadata),
        typeof(sol), typeof(ivgrid), typeof(ivs), typeof(pdedvs), typeof(sol.prob), typeof(sol.alg),
        typeof(interp), typeof(sol.stats),
    }(
        umap, sol, ivgrid, ivs,
        pdedvs, metadata, sol.prob, sol.alg,
        interp, sol.retcode, sol.stats
    )
end
