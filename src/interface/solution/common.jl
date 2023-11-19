
function (sol::SciMLBase.PDESolution{T,N,S,D})(args...;
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
        @assert length(args) == length(sol.ivs) "Not enough arguments for the number of independent variables  including time where appropriate, got $(length(args)) expected $(length(sol.ivs))."
        return map(sol.dvs) do dv
            arg_ivs = arguments(safe_unwrap(dv))
            is = map(arg_ivs) do arg_iv
                i = findfirst(isequal(arg_iv), sol.ivs)
                @assert i !== nothing "Independent variable $(arg_iv) in dependent variable $(dv) not found in the solution."
                i
            end
            if any(length.(sol.ivdomain) .== 1)
                is = drop_singleton_inds(is, sol.ivdomain)
            end
            sol.interp[dv](args[is]...)
        end
    end
    if any(length.(sol.ivdomain) .== 1)
        good_inds = get_good_inds(sol.ivdomain)
        return sol.interp[dv](args[good_inds]...)
    end
    return sol.interp[dv](args...)
end

function drop_singleton_inds(is, ivdomain)
    bad_is = findall(a->length(a) == 1, ivdomain)
    filter(a->!(a in bad_is), is)
end

function get_good_inds(ivdomain)
    findall(a->length(a) > 1, ivdomain)
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.PDESolution{T,N,S,D},
    sym) where {T,N,S,D<:MOLMetadata}
    iv = nothing
    dv = nothing
    iiv = sym_to_index(sym, A.ivs)
    if iiv !== nothing
        iv = A.ivs[iiv]
    end
    idv = sym_to_index(sym, A.dvs)
    if idv !== nothing
        dv = A.dvs[idv]
    elseif any(isequal(safe_unwrap(sym)), safe_unwrap.(collect(values(A.disc_data.discretespace.vars.replaced_vars))))
        dv = sym
    end
    if SciMLBase.issymbollike(sym) && iv !== nothing && isequal(sym, iv)
        A.ivdomain[iiv]
    elseif SciMLBase.issymbollike(sym) && dv !== nothing && isequal(sym, dv)
        A.u[sym]
    else
        error("Invalid indexing of solution. $sym not found in solution.")
    end
end

Base.@propagate_inbounds function Base.getindex(A::SciMLBase.PDESolution{T,N,S,D}, sym,
    args...) where {T,N,S,D<:MOLMetadata}
    iv = nothing
    dv = nothing
    iiv = sym_to_index(sym, A.ivs)
    if iiv !== nothing
        iv = A.ivs[iiv]
    end
    idv = sym_to_index(sym, A.dvs)
    if idv !== nothing
        dv = A.dvs[idv]
    end
    if SciMLBase.issymbollike(sym) && iv !== nothing && isequal(sym, iv)
        A.ivdomains[iiv][args...]
    elseif SciMLBase.issymbollike(sym) && dv !== nothing && isequal(sym, dv)
        A.u[sym][args...]
    else
        error("Invalid indexing of solution")
    end
end

function Base.display(pdesol::SciMLBase.PDESolution{T,N,S,D}) where {T, N, S, D <: MOLMetadata}
    sys = pdesol.disc_data.pdesys
    println("PDESolution:")
    println("  Return Code:")
    println("    $(pdesol.retcode)")
    println("  Dependent variables:")
    for key in keys(pdesol.u)
        println("    $(key): $(size(pdesol.u[key])) sized solution")
    end
    println("  Domain:")
    for (i, xdisc) in enumerate(pdesol.ivdomain)
        x = pdesol.ivs[i]
        if all([xdisc[i+1] - xdisc[i] ≈ xdisc[2] - xdisc[1] for i in 1:length(xdisc)-1])
            step = xdisc[2] - xdisc[1]
            println("    $(x) ∈ ($(xdisc[1]), $(xdisc[end])) with $(length(xdisc)) points, step size $(step)")
        else
            avgstep = sum([xdisc[i+1] - xdisc[i] for i in 1:length(xdisc)-1]) / (length(xdisc)-1)
            println("    $(x) ∈ ($(xdisc[1]), $(xdisc[end])) with $(length(xdisc)) non-uniform points. average step size $(avgstep)")
        end
    end
    println("  From system:")
    println("    Equations:")
    latexify(sys.eqs) |> display
    println("    Boundary/Initial Conditions:")
    latexify(sys.bcs) |> display
end

#=
# * Commented due to excessive compile time

# Due to method ambiguity, we have to define thia for every combination
#   of

# An algorithm that generates every permutation/combination of the given list of symbols
for l in 1:6
    ArgTypes = [Colon, AbstractVector, Int, StepRangeLen]

    argcombinations = map(0:4^(l)-1) do n
        inds = digits(n, base=4, pad=l) .+ 1
        map(i -> ArgTypes[i], inds)
    end
    for AT in argcombinations
        eval(quote
            function Base.getindex(A::SciMLBase.PDESolution{T,N,S,D},args::Tuple{$AT...}...; dv = nothing) where {T,N,S,D<:MOLMetadata}
                args = map(enumerate(args)) do (i, arg)
                    if arg isa Colon
                        sol.ivdomain[i]
                    else
                        arg
                    end
                end
                # If no dv is given, return interpolations for every dv
                if dv === nothing
                    @assert length(args) == length(sol.ivs) "Not enough arguments for the number of independent variables  including time where appropriate, got $(length(args)) expected $(length(sol.ivs))."
                    return map(sol.dvs) do dv
                        ivs = arguments(dv.val)
                        is = map(ivs) do iv
                            i = findfirst(isequal(arg), map(iv -> iv.val, sol.ivs))
                            @assert i !== nothing "Independent variable $(iv) in dependent variable $(dv) not found in the solution."
                            i
                        end

                        sol.u[dv][args[is]...]
                    end
                end
                return sol.u[dv][args...]
            end
        end)
    end
end
=#
