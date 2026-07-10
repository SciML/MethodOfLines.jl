# DAE detection for post-discretize numeric ODEProblems.
# Mirrors OrdinaryDiffEq solve init: isdae (singular mass) || has_initializeprob.
# Every predicate returns Bool; no try/catch control flow.

"""
    safe_is_numeric_zero(x) -> Bool

Return `true` only when `x` is a concrete numeric zero (not symbolic).
"""
function safe_is_numeric_zero(x)
    x isa Num && return false
    x isa Bool && return !x
    x isa Number && return iszero(x)
    return false
end

"""
    _is_numeric_mass_entry(x) -> Bool

Return `true` when `x` is a concrete numeric mass-matrix entry safe to test with `iszero`.
"""
_is_numeric_mass_entry(x) = x isa Number || x isa Bool

"""
    _diagonal_mass_is_singular(diag) -> Bool

Return `true` when the diagonal contains at least one numeric zero.
Symbolic entries are skipped (no boolean context on `Num`).
"""
function _diagonal_mass_is_singular(diag)
    @inbounds for x in diag
        if safe_is_numeric_zero(x)
            return true
        end
    end
    return false
end

"""
    _state_vector_numeric(u0) -> Bool

Return `true` when `u0` is a solve-ready numeric state (no symbolic `Num` entries).
"""
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

"""
    _mass_matrix_numeric_or_identity(M) -> Bool

Return `true` when `M` is identity or has only concrete numeric entries.
"""
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

"""
    numeric_contract(prob::ODEProblem) -> Bool

Return `true` when `prob` is a numeric, solve-ready `ODEProblem` from `discretize`.
Symbolic or mixed-type problems conservatively return `false`.
"""
function numeric_contract(prob::SciMLBase.ODEProblem)
    return _state_vector_numeric(prob.u0) &&
        _mass_matrix_numeric_or_identity(prob.f.mass_matrix)
end

"""
    safe_has_initializeprob(f) -> Bool

Bool-contract wrapper around `SciMLBase.has_initializeprob`.
"""
function safe_has_initializeprob(f)
    return SciMLBase.has_initializeprob(f) === true
end

"""
    safe_isdae_mass(M) -> Bool

Return `true` when `M` is a singular numeric mass matrix (OrdinaryDiffEq `isdae` mass criterion).
`I`, `UniformScaling`, and `Tuple` masses are treated as non-DAE.
"""
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

"""
    is_singular_mass_matrix(M) -> Bool

Alias for [`safe_isdae_mass`](@ref); retained for tests and backward compatibility.
"""
is_singular_mass_matrix(M) = safe_isdae_mass(M)

"""
    is_implicit_dae(prob::ODEProblem) -> Bool

Return `true` when `prob` requires DAE consistent initialization before `solve`.
Matches OrdinaryDiffEq: singular numeric mass matrix or MTK `initializeprob` metadata.
"""
function is_implicit_dae(prob::SciMLBase.ODEProblem)
    numeric_contract(prob) || return false
    if safe_has_initializeprob(prob.f)
        return true
    end
    return safe_isdae_mass(prob.f.mass_matrix)
end
