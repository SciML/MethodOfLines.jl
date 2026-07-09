"""
    is_implicit_dae(prob::ODEProblem) -> Bool

Return `true` when `prob` represents an implicit DAE with a singular numeric mass
matrix. Intended for MTK-compiled numeric `ODEProblem`s from `discretize`; symbolic
or non-numeric mass matrices conservatively return `false`.
"""
function is_implicit_dae(prob::SciMLBase.ODEProblem)
    return is_singular_mass_matrix(prob.f.mass_matrix)
end

function is_singular_mass_matrix(M)
    M === I && return false
    M isa UniformScaling && return false
    M isa Tuple && return false
    if M isa Diagonal
        any(d -> d isa Num, M.diag) && return false
        eltype(M.diag) <: Number || return false
        return any(iszero, M.diag)
    elseif M isa AbstractMatrix
        any(d -> d isa Num, M) && return false
        eltype(M) <: Number || return false
        return any(iszero, diag(M))
    end
    return false
end
