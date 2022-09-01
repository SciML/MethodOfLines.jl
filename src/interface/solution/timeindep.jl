function PDENoTimeSolution(sol::SciMLBase.ODESolution{T}, metadata::MOLMetadata) where {T}
    odesys = sol.prob.f.sys

    pdesys = metadata.pdesys
    discretespace = metadata.discretespace

    ivs = [discretespace.x̄...]
    ivgrid = (discretespace.grid[x] for x in ivs)
    # Reshape the solution to flat arrays
    umap = Dict(map(discretespace.ū) do u
        let discu = discretespace.discvars[u]
            solu = map(CartesianIndices(discu)) do I
                i = SciMLBase.sym_to_index(discu[I], odesys)
                # Handle Observed
                if i !== nothing
                    sol.u[i]
                else
                    SciMLBase.observed(sol, discu[I])
                end
            end
            out = zeros(T, size(discu)...)
            for I in CartesianIndices(discu)
                out[I] .= solu[I]
            end
            u => out
        end
    end)
    # Build Interpolations
    interp = build_interpolation(umap, ivs, ivgrid)

    return SciMLBase.PDENoTimeSolution{T,length(discretespace.ū),typeof(umap),typeof(metadata),
        typeof(sol),typeof(ivgrid),typeof(ivs),typeof(pdesys.dvs),typeof(sol.prob),typeof(sol.alg),
        typeof(interp)}(umap, sol, ivgrid, ivs,
        pdesys.dvs, metadata, sol.prob, sol.alg,
        interp, sol.retcode)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.PDENoTimeSolution{T,N,S,D},
    sym) where {T,N,S,D<:MOLMetadata}
    if SciMLBase.issymbollike(sym) || all(SciMLBase.issymbollike, sym)
        if sym isa AbstractArray
            return map(s -> A[s], collect(sym))
        end
        i = SciMLBase.sym_to_index(sym, A.original_sol)
    else
        i = sym
    end

    iv = nothing
    dv = nothing
    if i === nothing
        iiv = SciMLBase.sym_to_index(sym, A.ivs)
        if iiv !== nothing
            iv = A.ivs[iiv]
        end
        idv = SciMLBase.sym_to_index(sym, A.dvs)
        if idv !== nothing
            dv = A.dvs[idv]
        end
        if SciMLBase.issymbollike(sym) && iv !== nothing && isequal(sym, iv)
            A.ivdomain[iiv]
        elseif SciMLBase.issymbollike(sym) && dv !== nothing && isequal(sym, dv)
            A.u[sym]
        else
            SciMLBase.observed(A.original_sol, sym)
        end
    elseif i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        A.original_sol[i]
    else
        error("Invalid indexing of solution")
    end
end