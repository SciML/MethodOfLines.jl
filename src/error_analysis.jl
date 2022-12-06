function error_analysis(sys, e)
    eqs = sys.eqs
    states = sys.states
    t = sys.iv
    println("The system of equations is:")
    println(eqs)
    if e isa ModelingToolkit.ExtraVariablesSystemException

        rs = [Differential(t)(state) => state for state in states]
        extrastates = [state for state in states]
        extraeqs = [eq for eq in eqs]
        for r in rs
            for eq in extraeqs
                if subsmatch(eq.lhs, r) | subsmatch(eq.rhs, r)
                    extrastates = vec(setdiff(extrastates, [r.second]))
                    extraeqs = vec(setdiff(extraeqs, [eq]))
                    break
                end
            end
        end
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
