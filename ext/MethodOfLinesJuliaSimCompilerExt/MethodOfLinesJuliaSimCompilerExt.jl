module MethodOfLinesJuliaSimCompilerExt
using JuliaSimCompiler, MethodOfLines, PDEBase, SciMLBase, ModelingToolkit
using PDEBase: get_discvars, get_time, add_metadata!

function PDEBase.add_metadata!(sys::JuliaSimCompiler.IRSystem, meta)
	sys.info.parent.metadata.metadata[] = meta
end

function PDEBase.generate_system(disc_state::PDEBase.EquationState, s, u0, tspan, metadata,
	disc::MethodOfLines.MOLFiniteDifference)
	println("JuliaSimCompiler: generate_system")
	discvars = get_discvars(s)
	t = get_time(disc)
	name = metadata.pdesys.name
	pdesys = metadata.pdesys
	alleqs = vcat(disc_state.eqs, unique(disc_state.bceqs))
	alldepvarsdisc = vec(reduce(vcat, vec(unique(reduce(vcat, vec.(values(discvars)))))))

	defaults = Dict(pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? u0 :
					vcat(u0, pdesys.ps))
	ps = pdesys.ps === nothing || pdesys.ps === SciMLBase.NullParameters() ? Num[] :
		 first.(pdesys.ps)
	# Finalize
	# if haskey(metadata.disc.kwargs, :checks)
	#     checks = metadata.disc.kwargs[:checks]
	# else
	checks = true
	# end
	try
		if t === nothing
			# At the time of writing, NonlinearProblems require that the system of equations be in this form:
			# 0 ~ ...
			# Thus, before creating a NonlinearSystem we normalize the equations s.t. the lhs is zero.
			eqs = map(eq -> 0 ~ eq.rhs - eq.lhs, alleqs)
			sys = NonlinearSystem(eqs, alldepvarsdisc, ps, defaults = defaults, name = name,
				metadata = metadata, checks = checks)
			return sys, nothing
		else
			# * In the end we have reduced the problem to a system of equations in terms of Dt that can be solved by an ODE solver.
			sys = ODESystem(alleqs, t, alldepvarsdisc, ps, defaults = defaults, name = name,
				metadata = metadata, checks = checks)
			@show sys.eqs
			@show sys.observed
			sys = IRSystem(sys)
			return sys, tspan
		end
	catch e
		println("The system of equations is:")
		println(alleqs)
		println()
		println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
		println()
		rethrow(e)
	end
end

function SciMLBase.discretize(pdesys::PDESystem,
	discretization::MOLFiniteDifference;
	analytic = nothing, kwargs...)
	sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
	try
		simpsys = structural_simplify(sys)
		if tspan === nothing
			return prob = NonlinearProblem(simpsys, ones(length(simpsys.states));
				discretization.kwargs..., kwargs...)
		else
			# Use ODAE if nessesary
        
            prob = ODEProblem(simpsys, Pair[], tspan; discretization.kwargs...,
                kwargs...)
            if analytic === nothing
                return prob
            else
                f = ODEFunction(pdesys, discretization, analytic = analytic,
                    discretization.kwargs..., kwargs...)

                return ODEProblem(f, prob.u0, prob.tspan, prob.p;
                    discretization.kwargs..., kwargs...)
            end
		end
	catch e
		PDEBase.error_analysis(sys, e)
	end
end

function PDEBase.error_analysis(sys::IRSystem, e)
    rethrow(e)
    
end
end