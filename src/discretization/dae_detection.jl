"""
    is_implicit_dae(prob::ODEProblem) -> Bool

Return `true` when `prob` represents an implicit ODE/DAE system that requires
consistent initialization (singular mass matrix or MTK initialization data).
Matches the DAE detection used in OrdinaryDiffEq when `initialize_dae!` runs.
"""
function is_implicit_dae(prob::SciMLBase.ODEProblem)
    SciMLBase.has_initializeprob(prob.f) && return true
    return is_singular_mass_matrix(prob.f.mass_matrix)
end

function is_singular_mass_matrix(M)
    M === I && return false
    M isa UniformScaling && return false
    M isa Tuple && return false
    if M isa Diagonal
        return any(iszero, M.diag)
    elseif M isa AbstractMatrix
        return any(iszero, diag(M))
    end
    return false
end
