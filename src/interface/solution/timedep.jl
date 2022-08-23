
function PDETimeSeriesSolution(sol::ODESolution{T}, metadata::MOLMetadata) where {T}
    odesys = sol.prob.f.sys

    pdesys = metadata.pdesys
    discretespace = metadata.discretespace

    ivs = [discretespace.time, discretespace.x̄...]
    ivgrid = (isequal(discretespace.time, x) ? sol.t : discretespace.grid[x] for x in ivs)
    # Reshape the solution to flat arrays
    umap = Dict(map(discretespace.ū) do u
        let discu = discretespace.discvars[u]
            solu = map(CartesianIndices(discu)) do I
                i = sym_to_index(discu[I], odesys)
                # Handle Observed
                if i !== nothing
                    sol.u[i]
                else
                    observed(sol, discu[I], :)
                end
            end
            out = zeros(T, length(sol.t), size(discu)...)
            for I in CartesianIndices(discu)
                out[:, I] .= solu[I]
            end
            u => out
        end
    end)
    # Build Interpolations
    interp = build_interpolation(umap, ivs, ivgrid)

    return PDETimeSeriesSolution{T,length(discretespace.ū),typeof(umap),typeof(metadata),
        typeof(sol),typeof(sol.errors),typeof(sol.t),typeof(ivgrid),
        typeof(ivs),typeof(pdesys.dvs),typeof(sol.prob),typeof(sol.alg),
        typeof(interp)}(umap, sol, sol.errors, sol.t, ivgrid, ivs,
        pdesys.dvs, metadata, sol.prob, sol.alg,
        interp, sol.dense, sol.tslocation,
        sol.retcode)
end

Base.@propagate_inbounds function Base.getindex(A::PDETimeSeriesSolution{T,N,S,D},
    sym) where {T,N,S,D<:MOLMetadata}
    if issymbollike(sym) || all(issymbollike, sym)
        if sym isa AbstractArray
            return map(s -> A[s], collect(sym))
        end
        i = sym_to_index(sym, A.original_sol)
    else
        i = sym
    end

    iv = nothing
    dv = nothing
    if i === nothing
        iiv = sym_to_index(sym, A.ivs)
        if iiv !== nothing
            iv = A.ivs[iiv]
        end
        idv = sym_to_index(sym, A.dvs)
        if idv !== nothing
            dv = A.dvs[idv]
        end
        if issymbollike(sym) && iv !== nothing && isequal(sym, iv)
            A.ivdomain[iiv]
        elseif issymbollike(sym) && dv !== nothing && isequal(sym, dv)
            A.u[sym]
        else
            observed(A.original_sol, sym, :)
        end
    elseif i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        A.original_sol[i, :]
    else
        error("Invalid indexing of solution")
    end
end
