function error_analysis(sys, e)
    if e isa ModelingToolkit.ExtraVariablesSystemException
        eqs = sys.eqs
        states = sys.states
        t = sys.iv

        rs = [Differential(t)(state) => state for state in states]
        extrastates = [state for state in states]
        extraeqs = [eq for eq in eqs]
        for r in rs
            for eq in extraeqs
                if subsmatch(eq.lhs, r) | subsmatch(eq.rhs, r)
                    setdiff!(extrastates, [r.second])
                    setdiff!(extraeqs, [eq])
                    break
                end
            end
        end
        println("The system of equations is:")
        println(eqs)
        println()
        println("There are $(length(states)) variables and $(length(eqs)) equations.\n")
        println("The variables without time derivatives are:")
        println(extrastates)
        println()
        println("The equations without time derivatives are:")
        println(extraeqs)
        rethrow(e)
    else
        rethrow(e)
    end
end
