function generate_ivgrid(discretespace, ivs, t, metadata::MOLMetadata{G}) where {G}
    return ((isequal(discretespace.time, x) ? t : discretespace.grid[x] for x in ivs)...,)
end

function generate_ivgrid(
        discretespace, ivs, t, metadata::MOLMetadata{G}
    ) where {G <: StaggeredGrid}
    return #TODO ((isequal(discretespace.time, x) ? t : discretespace.grid[x] for x in ivs)...,)
end

# If the solution is already a PDETimeSeriesSolution, just return it unchanged.
# This handles cases where wrap_sol is called multiple times (e.g., during error handling).
function SciMLBase.PDETimeSeriesSolution(
        sol::SciMLBase.PDETimeSeriesSolution, metadata::MOLMetadata
    )
    return sol
end

function array_observed_solution(sol, discu, observed_equations)
    for eq in observed_equations
        lhs = Symbolics.wrap(eq.lhs)
        lhs isa AbstractArray || continue
        size(lhs) == size(discu) || continue
        entries = Symbolics.scalarize(lhs)
        all(isequal(safe_unwrap(entries[I]), safe_unwrap(discu[I])) for I in CartesianIndices(discu)) || continue
        evaluator = SymbolicIndexingInterface.observed(sol, eq.lhs)
        values = evaluator.(
            SymbolicIndexingInterface.state_values(sol),
            (SymbolicIndexingInterface.parameter_values(sol),),
            SymbolicIndexingInterface.current_time(sol)
        )
        return map(CartesianIndices(discu)) do I
            getindex.(values, (I,))
        end
    end
    return nothing
end

function SciMLBase.PDETimeSeriesSolution(
        sol::SciMLBase.AbstractODESolution{T}, metadata::MOLMetadata
    ) where {T}
    try
        hostsol = host_solution(sol)
        odesys = sol.prob.f.sys
        pdesys = metadata.pdesys
        discretespace = metadata.discretespace

        ivs = [discretespace.time, discretespace.x̄...]
        ivgrid = generate_ivgrid(discretespace, ivs, sol.t, metadata)

        dvs = discretespace.ū
        # Reshape the solution to flat arrays, faster to do this eagerly.
        umap = mapreduce(vcat, dvs) do u
            let discu = discretespace.discvars[u]
                # Unknowns are read by index: the observed function of a whole field is one
                # statement per grid point, and compiling it grows faster than the grid. Only a
                # field with no unknowns, such as a reduced order reconstruction, is evaluated
                # as an array.
                idxs = grid_indices(odesys, discu)
                solu = all(isnothing, idxs) ?
                    array_observed_solution(hostsol, discu, ModelingToolkitBase.observed(odesys)) :
                    nothing
                if solu === nothing
                    solu = map(i -> i === nothing ? T[] : hostsol[i, :], idxs)
                    obs = findall(isnothing, idxs)
                    if !isempty(obs)
                        obsvals = grid_observed_values(hostsol, discu, obs)
                        for (k, I) in enumerate(obs)
                            solu[I] = getindex.(obsvals, k)
                        end
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
                    error("The time variable must be the first or last argument of the dependent variable $u.")
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

        return SciMLBase.PDETimeSeriesSolution{
            T, length(discretespace.ū), typeof(umap), typeof(metadata),
            typeof(sol), typeof(sol.errors), typeof(sol.t), typeof(ivgrid),
            typeof(ivs), typeof(get_dvs(pdesys)), typeof(sol.prob), typeof(sol.alg),
            typeof(interp), typeof(sol.stats),
        }(
            umap, sol, sol.errors, sol.t, ivgrid, ivs,
            get_dvs(pdesys), metadata, sol.prob, sol.alg,
            interp, sol.dense, sol.tslocation,
            sol.retcode, sol.stats
        )
    catch e
        rethrow(e)
        return sol, e
    end
end
