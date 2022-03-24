# [How it works] (@id hiw)

MethodOfLines.jl makes heavy use of `Symbolics.jl` and `SymbolicUtils.jl`, namely it's rule matching features to recognize terms which require particular discretizations.

Given your discretization and `PDESystem`, we take each independent variable defined on the space to be discretized and create a corresponding range. We then take each dependant variable and create an array of symbolic variables to represent it in its discretized form. 

Next, the boundary conditions are discretized, creating an equation for each point on the boundary in terms of the discretized variables, replacing any space derivatives in the direction of the boundary with their upwind finite difference expressions.

After that, the system of PDEs is discretized, first matching each PDE to each dependant variable by which variable is highest order in each PDE, with precedance given to time derivatives. Then, the PDEs are discretized creating a finite difference equation for each point in their matched dependant variables discrete form, less the number of boundary equations. These equations are removed from around the boundary, so each PDE only has discrete equations on its variable's interior.

Now we have a system of equations which are either ODEs, linear, or nonlinear equations and an equal number of unknowns. See [here](@ref brusssys) for the system that is generated for the Brusselator at low point count. The structure of the system is simplified with `ModelingToolkit.structural_simplify`, and then either an `ODEProblem` or `NonlinearProblem` is returned. Under the hood, the `ODEProblem` generates a fast semidiscretization, written in julia with `RuntimeGeneratedFunctions`. See [here](@ref brusscode) for an example of the generated code for the Brusselator system at low point count. 
