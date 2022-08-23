
function (sol::PDESolution{T,N,S,D})(args...;
    dv=nothing) where {T,N,S,D<:MOLMetadata}
    # Colon reconstructs on gridpoints
    args = map(enumerate(args)) do (i, arg)
        if arg isa Colon
            sol.ivdomain[i]
        else
            arg
        end
    end
    # If no dv is given, return interpolations for every dv
    if dv === nothing
        @assert length(args) == length(sol.ivs) "Not enough arguments for the number of
                                                     independent variables (including time),
                                                     got $(length(args)) expected
                                                     $(length(sol.ivs) + 1)."
        return map(dvs) do dv
            ivs = arguments(dv)
            is = map(ivs) do iv
                i = findfirst(iv, sol.ivs)
                @assert i !== nothing "Independent variable $(iv) in
                                       dependent variable $(dv) not found in the solution."
                i
            end

            sol.interp[dv](args[is]...)
        end
    end
    return sol.interp[dv](args...)
end


Base.@propagate_inbounds function Base.getindex(A::PDESolution{T,N,S,D}, sym,
    args...) where {T,N,S,D<:MOLMetadata}
    if issymbollike(sym)
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
            A.ivdomains[iiv]
        elseif issymbollike(sym) && dv !== nothing && isequal(sym, dv)
            A.u[sym][args...]
        else
            observed(A.original_sol, sym, args...)
        end
    elseif i isa Base.Integer || i isa AbstractRange || i isa AbstractVector{<:Base.Integer}
        A.original_sol[i, args...]
    else
        error("Invalid indexing of solution")
    end
end
