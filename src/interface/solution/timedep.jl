
function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.AbstractODESolution{T}, metadata::MOLMetadata) where {T}
    try
        odesys = sol.prob.f.sys
        Main.xx[] = sol
        pdesys = metadata.pdesys
        discretespace = metadata.discretespace

        ivs = [discretespace.time, discretespace.x̄...]
        ivgrid = ((isequal(discretespace.time, x) ? sol.t : discretespace.grid[x] for x in ivs)...,)

        solved_states = if metadata.use_ODAE
            deriv_states = metadata.metadata[]
            states(odesys)[deriv_states]
        else
            states(odesys)
        end
        # Reshape the solution to flat arrays, faster to do this eagerly.
        umap = Dict(map(discretespace.ū) do u
            let discu = discretespace.discvars[u]
                solu = map(CartesianIndices(discu)) do I
                    i = sym_to_index(discu[I], solved_states)
                    # Handle Observed
                    if i !== nothing
                        sol[i, :]
                    else
                        SciMLBase.observed(sol, discu[I], :)
                    end
                end
                # Correct placement of time axis
                if isequal(arguments(u)[1], discretespace.time)
                    out = zeros(T, length(sol.t), size(discu)...)
                    for I in CartesianIndices(discu)
                        out[:, I] .= solu[I]
                    end
                elseif isequal(arguments(u)[end], discretespace.time)
                    out = zeros(T, size(discu)..., length(sol.t))
                    for I in CartesianIndices(discu)
                        out[I, :] .= solu[I]
                    end
                else
                    @assert false "The time variable must be the first or last argument of the dependent variable $u."
                end

                Num(u) => out
            end
        end)
        # Build Interpolations
        interp = build_interpolation(umap, ivs, ivgrid, sol, pdesys)

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
