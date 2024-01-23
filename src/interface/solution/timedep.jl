
function generate_ivgrid(discretespace, ivs, t, metadata::MOLMetadata{G}) where {G}
    return ((isequal(discretespace.time, x) ? t : discretespace.grid[x] for x in ivs)...,)
end

function generate_ivgrid(discretespace, ivs, t, metadata::MOLMetadata{G}) where {G<:StaggeredGrid}
    return #TODO ((isequal(discretespace.time, x) ? t : discretespace.grid[x] for x in ivs)...,)
end

function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.AbstractODESolution{T}, metadata::MOLMetadata) where {T}
    try
        odesys = sol.prob.f.sys
        pdesys = metadata.pdesys
        discretespace = metadata.discretespace

        ivs = [discretespace.time, discretespace.x̄...]
        ivgrid = generate_ivgrid(discretespace, ivs, sol.t, metadata)

        solved_states = if metadata.use_ODAE
            deriv_states = metadata.metadata[]
            states(odesys)[deriv_states]
        else
            states(odesys)
        end
        dvs = discretespace.ū
        # Reshape the solution to flat arrays, faster to do this eagerly.
        umap = mapreduce(vcat, dvs) do u
            let discu = discretespace.discvars[u]
                # Initialize output with correct placement of time axis.
                if isequal(arguments(u)[1], discretespace.time)
                    out = zeros(T, length(sol.t), size(discu)...)
                elseif isequal(arguments(u)[end], discretespace.time)
                    out = zeros(T, size(discu)..., length(sol.t))
                else
                    error("The time variable must be the first or last argument of the dependent variable $u.")
                end
                
                # Build the flat arrays with time in the outer loop.
                # It is important for time to be in the outer loop because
                # this function may access data interpolators which may be
                # optimized to sequentially step through time.
                for ti ∈ eachindex(sol.t)
                    Is = CartesianIndices(discu)
                    is = sym_to_index.(discu[Is], (solved_states,))
                    
                    # Handle Observed
                    if is[1] !== nothing
                        solu = sol[is, ti]
                    else
                        solu = SciMLBase.observed.((sol,), safe_unwrap.(discu[Is]), (ti,))
                    end

                    # Correct placement of time axis
                    if isequal(arguments(u)[1], discretespace.time)
                        out[ti, Is] .= solu
                    elseif isequal(arguments(u)[end], discretespace.time)
                        out[Is, ti] .= solu
                    else
                        @assert false "The time variable must be the first or last argument of the dependent variable $u."
                    end
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
        interp = build_interpolation(umap, dvs, ivs, ivgrid, sol, pdesys, discretespace.vars.replaced_vars)

        return SciMLBase.PDETimeSeriesSolution{T,length(discretespace.ū),typeof(umap),typeof(metadata),
            typeof(sol),typeof(sol.errors),typeof(sol.t),typeof(ivgrid),
            typeof(ivs),typeof(pdesys.dvs),typeof(sol.prob),typeof(sol.alg),
            typeof(interp), typeof(sol.stats)}(umap, sol, sol.errors, sol.t, ivgrid, ivs,
            pdesys.dvs, metadata, sol.prob, sol.alg,
            interp, sol.dense, sol.tslocation,
            sol.retcode, sol.stats)
    catch e
        rethrow(e)
        return sol, e
    end
end
