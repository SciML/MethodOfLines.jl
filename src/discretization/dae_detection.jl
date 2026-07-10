function safe_is_numeric_zero(x)
    x isa Num && return false
    x isa Bool && return !x
    x isa Number && return iszero(x)
    return false
end

_is_numeric_mass_entry(x) = x isa Number || x isa Bool

function _diagonal_mass_is_singular(diag)
    @inbounds for x in diag
        if safe_is_numeric_zero(x)
            return true
        end
    end
    return false
end

function _state_vector_numeric(u0)
    u0 isa Num && return false
    u0 isa Number && return true
    u0 isa AbstractArray || return false
    @inbounds for u in u0
        u isa Num && return false
        if !(u isa Number) && !(u isa Bool)
            return false
        end
    end
    return true
end

function _mass_matrix_numeric_or_identity(M)
    M === I && return true
    M isa UniformScaling && return _is_numeric_mass_entry(M.λ)
    M isa Tuple && return false
    if M isa Diagonal
        @inbounds for x in M.diag
            _is_numeric_mass_entry(x) || return false
        end
        return true
    elseif M isa AbstractMatrix
        @inbounds for x in M
            _is_numeric_mass_entry(x) || return false
        end
        return true
    end
    return false
end

function numeric_contract(prob::SciMLBase.ODEProblem)
    return _state_vector_numeric(prob.u0) &&
        _mass_matrix_numeric_or_identity(prob.f.mass_matrix)
end

function safe_has_initializeprob(f)
    return SciMLBase.has_initializeprob(f) === true
end

function safe_isdae_mass(M)
    M === I && return false
    M isa UniformScaling && return false
    M isa Tuple && return false
    if M isa Diagonal
        return _diagonal_mass_is_singular(M.diag)
    elseif M isa AbstractMatrix
        return _diagonal_mass_is_singular(diag(M))
    end
    return false
end

is_singular_mass_matrix(M) = safe_isdae_mass(M)

"""
    is_implicit_dae(prob::ODEProblem) -> Bool

`true` for numeric problems with singular mass matrix or MTK initialization metadata.
"""
function is_implicit_dae(prob::SciMLBase.ODEProblem)
    numeric_contract(prob) || return false
    if safe_has_initializeprob(prob.f)
        return true
    end
    return safe_isdae_mass(prob.f.mass_matrix)
end
