module MethodOfLinesAbstractFFTsExt

using MethodOfLines: MethodOfLines, FourierCollocation, apply_along
using AbstractFFTs: AbstractFFTs, plan_rfft, plan_irfft

"""
    FourierFFTDerivative

Whole-direction Fourier derivative applied with real FFTs: `irfft(mult .* rfft(X))`
along one axis, with `mult = (i k)^d` over the `rfft` modes and the Nyquist mode
zeroed, which is the mode-space form of the dense matrix `D1^d`. `mat` is that
matrix, used for element types the FFT plans do not cover (dual numbers during
Jacobian evaluation). Plans are cached per array size and axis.
"""
struct FourierFFTDerivative{T, M <: AbstractMatrix{T}}
    n::Int
    mult::Vector{Complex{T}}
    mat::M
    plans::Dict{Any, Any}
end

function MethodOfLines.fast_fourier_operator(spec::FourierCollocation, grid, d, mat)
    n = spec.n
    T = float(eltype(grid))
    try
        plan_rfft(zeros(T, n))
    catch e
        e isa InterruptException && rethrow(e)
        @debug "No AbstractFFTs backend available for $(typeof(mat)); using the dense Fourier differentiation matrix." exception = e
        return nothing
    end
    L = T(grid[end] - grid[1])
    k = (2 * T(π) / L) .* (0:(n ÷ 2))
    mult = (im .* k) .^ d
    iseven(n) && (mult[end] = 0)
    return FourierFFTDerivative{T, typeof(mat)}(n, mult, mat, Dict{Any, Any}())
end

function fft_plans!(op::FourierFFTDerivative{T}, X, J) where {T}
    key = (size(X), J)
    return get!(op.plans, key) do
        p = plan_rfft(X, J)
        Xh = p * X
        (p, plan_irfft(Xh, op.n, J))
    end
end

function MethodOfLines.apply_along(
        op::FourierFFTDerivative{T}, X::AbstractArray{T, N}, ::Val{J}
    ) where {T, N, J}
    size(X, J) == op.n || return apply_along(op.mat, X, Val(J))
    p, pinv = fft_plans!(op, X, J)
    Xh = p * X
    shape = ntuple(i -> i == J ? length(op.mult) : 1, N)
    Xh .*= reshape(op.mult, shape)
    return pinv * Xh
end

function MethodOfLines.apply_along(op::FourierFFTDerivative, X::AbstractArray, ::Val{J}) where {J}
    return apply_along(op.mat, X, Val(J))
end

end
