
function generate_ivgrid(discretespace, ivs, t, metadata::MOLMetadata{G}) where {G}
    return ((isequal(discretespace.time, x) ? t : discretespace.grid[x] for x in ivs)...,)
end

function generate_ivgrid(
        discretespace, ivs, t, metadata::MOLMetadata{G}) where {G <: StaggeredGrid}
    return #TODO ((isequal(discretespace.time, x) ? t : discretespace.grid[x] for x in ivs)...,)
end

function SciMLBase.PDETimeSeriesSolution(
        sol::SciMLBase.AbstractODESolution{T}, metadata::MOLMetadata) where {T}
    try
        odesys = sol.prob.f.sys
        pdesys = metadata.pdesys
        discretespace = metadata.discretespace

        ivs = [discretespace.time, discretespace.x̄...]
        ivgrid = generate_ivgrid(discretespace, ivs, sol.t, metadata)

        solved_unknowns = if metadata.use_ODAE
            deriv_unknowns = metadata.metadata[]
            unknowns(odesys)[deriv_unknowns]
        else
            unknowns(odesys)
        end
        dvs = discretespace.ū
        # Reshape the solution to flat arrays, faster to do this eagerly.
        umap = mapreduce(vcat, dvs) do u
            let discu = discretespace.discvars[u]
                solu = map(CartesianIndices(discu)) do I
                    i = sym_to_index(discu[I], solved_unknowns)
                    # Handle Observed
                    if i !== nothing
                        sol[i, :]
                    else
                        SciMLBase.observed(sol, safe_unwrap(discu[I]), :)
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
            umap, dvs, ivs, ivgrid, sol, pdesys, discretespace.vars.replaced_vars)

        return SciMLBase.PDETimeSeriesSolution{
            T, length(discretespace.ū), typeof(umap), typeof(metadata),
            typeof(sol), typeof(sol.errors), typeof(sol.t), typeof(ivgrid),
            typeof(ivs), typeof(pdesys.dvs), typeof(sol.prob), typeof(sol.alg),
            typeof(interp), typeof(sol.stats)}(umap, sol, sol.errors, sol.t, ivgrid, ivs,
            pdesys.dvs, metadata, sol.prob, sol.alg,
            interp, sol.dense, sol.tslocation,
            sol.retcode, sol.stats)
    catch e
        rethrow(e)
        return sol, e
    end
end
