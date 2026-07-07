function PDEBase.generate_ic_defaults(
        tconds, s::DiscreteSpace,
        ::MOLFiniteDifference{G, DS}
    ) where {G, DS <: ScalarizedDiscretization}
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
            rhs_vals = pde_substitute.(
                broadcastable_rhs, Dict.(valmaps(s, depvar(ic.u, s), ic.depvars, indexmap))
            )
            # Substitution in SymbolicUtils v4 no longer constant-folds function calls, so an
            # IC like `u(0, x) ~ cos(x)` produces stuck terms like `cos(0.108...)`. MTK v11's
            # `varmap_to_vars` requires literal constants in the u0 map and reports any
            # unfolded value as "Initial condition underdefined". `symbolic_to_float` folds
            # fully-constant expressions to numbers and leaves genuinely symbolic values
            # (e.g. parameter-dependent ICs) untouched for MTK to resolve downstream.
            fold(v) = v isa Number ? v : symbolic_to_float(v)
            vec(defaultvars .=> fold.(rhs_vals))
        end
    else
        u0 = []
    end
    return u0
end
