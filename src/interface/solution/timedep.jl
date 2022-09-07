
function SciMLBase.PDETimeSeriesSolution(sol::SciMLBase.ODESolution{T}, metadata::MOLMetadata) where {T}
    try
        odesys = sol.prob.f.sys

        pdesys = metadata.pdesys
        discretespace = metadata.discretespace

        ivs = [discretespace.time, discretespace.x̄...]
        ivgrid = ((isequal(discretespace.time, x) ? sol.t : discretespace.grid[x] for x in ivs)...,)
        # Reshape the solution to flat arrays
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

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.PDETimeSeriesSolution{T,N,S,D},
    sym) where {T,N,S,D<:MOLMetadata}
    if SciMLBase.issymbollike(sym) || all(SciMLBase.issymbollike, sym)
        if sym isa AbstractArray
            return map(s -> A[s], collect(sym))
        end
        i = sym_to_index(sym, A.prob.f.sys.states)
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
        if SciMLBase.issymbollike(sym) && iv !== nothing && isequal(sym, iv)
            A.ivdomain[iiv]
        elseif SciMLBase.issymbollike(sym) && dv !== nothing && isequal(sym, dv)
            A.u[sym]
        else
            SciMLBase.observed(A.original_sol, sym, :)
        end
    elseif i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        A.original_sol[i, :]
    else
        error("Invalid indexing of solution")
    end
end

# Must be defined due to ambiguity for sol[1]
Base.@propagate_inbounds function Base.getindex(A::SciMLBase.PDETimeSeriesSolution{T,N,S,D},
    i::Int) where {T,N,S,D<:MOLMetadata}
    _getindex(A, i)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.PDETimeSeriesSolution{T,N,S,D},
    i::Colon) where {T,N,S,D<:MOLMetadata}
    _getindex(A, i)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.PDETimeSeriesSolution{T,N,S,D},
    i::AbstractArray) where {T,N,S,D<:MOLMetadata}
    _getindex(A, i)
end

# Must be defined due to ambiguity for sol[1]
Base.@propagate_inbounds function _getindex(A::SciMLBase.PDETimeSeriesSolution{T,N,S,D},
    i::Union{Colon,Int,<:AbstractArray}) where {T,N,S,D<:MOLMetadata}
    if i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        A.original_sol[i, :]
    else
        error("Invalid indexing of solution")
    end
end
