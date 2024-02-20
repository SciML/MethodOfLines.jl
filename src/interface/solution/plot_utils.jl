function quick_animate(sol::SciMLBase.PDETimeseriesSolution, u)
    solu = sol[u]
    s = sol.disc_data.discretespace

    @assert ndims(solu)==2 "Only 2D (1 space, 1 time) solutions are supported for animation."

    t = s.time
    discx̄ = map(remove(arguments(u), t)) do x
        sol[x]
    end

    findfirst(isequal(t), arguments(u)) == 1 ? false : true

    anim = @animate for i in 1:length(sol.t)
        indices = map(arguments(u)) do x
            if isequal(t, x)
                i
            else
                Colon()
            end
        end
        plot(discx̄..., solu[indices...])
    end
    return anim
end
