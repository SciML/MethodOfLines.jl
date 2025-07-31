function PDEBase.generate_ic_defaults(tconds, s::DiscreteSpace,
        ::MOLFiniteDifference{G, DS}) where {G, DS <: ScalarizedDiscretization}
    t = s.time
    if s.time !== nothing
        u0 = mapreduce(vcat, tconds) do ic
            if isupper(ic)
                throw(ArgumentError("Upper boundary condition $(ic.eq) on time variable is not supported, please use a change of variables `t => -τ` to make this an initial condition."))
            end

            args = ivs(depvar(ic.u, s), s)
            indexmap = Dict([args[i] => i for i in 1:length(args)])
            D = ic.order == 0 ? identity : (Differential(t)^ic.order)
            defaultvars = D.(s.discvars[depvar(ic.u, s)])
            broadcastable_rhs = [symbolic_linear_solve(ic.eq, D(ic.u))]
            out = substitute.(
                broadcastable_rhs, valmaps(s, depvar(ic.u, s), ic.depvars, indexmap))
            vec(defaultvars .=> substitute.(
                broadcastable_rhs, valmaps(s, depvar(ic.u, s), ic.depvars, indexmap)))
        end
    else
        u0 = []
    end
    return u0
end
