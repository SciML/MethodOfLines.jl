function generate_ic_defaults(boundarymap, s)
    t = s.time
    if s.time !== nothing
        tconds = mapreduce(u -> boundarymap[u][t], vcat, operation.(s.ū))
        u0 = mapreduce(vcat, tconds) do tc
            if isupper(tc)
                throw(ArgumentError("Upper boundary condition $(tc.eq) on time variable is not supported, please use a change of variables `t => -τ` to make this an initial condition."))
            end

            args = params(depvar(tc.u, s), s)
            indexmap = Dict([args[i]=>i for i in 1:length(args)])
            D = tc.order == 0 ? identity : (Differential(t)^tc.order)
            defaultvars = D.(s.discvars[depvar(tc.u, s)])
            broadcastable_rhs = (solve_for(tc.eq, D(tc.u)),)
            vec(defaultvars .=> substitute.(broadcastable_rhs, valmaps(s, depvar(tc.u,s), tc.depvars, indexmap)))
        end
    else
        u0 = []
    end
    return u0
end
