module MethodOfLinesAbstractFFTsExt

using MethodOfLines: MethodOfLines, ChebyshevCollocation, FourierCollocation, apply_along
using AbstractFFTs: AbstractFFTs, plan_rfft, plan_irfft
using LinearAlgebra: mul!

function has_fft_backend(T, n)
    try
        plan_rfft(zeros(T, n))
        return true
    catch e
        e isa InterruptException && rethrow(e)
        @debug "No AbstractFFTs backend available; using the dense differentiation matrix." exception = e
        return false
    end
end

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

function MethodOfLines.fast_spectral_operator(spec::FourierCollocation, grid, d, mat)
    n = spec.n
    T = float(eltype(grid))
    has_fft_backend(T, n) || return nothing
    L = T(grid[end] - grid[1])
    k = (2 * T(π) / L) .* (0:(n ÷ 2))
    mult = (im .* k) .^ d
    iseven(n) && (mult[end] = 0)
    return FourierFFTDerivative{T, typeof(mat)}(n, mult, mat, Dict{Any, Any}())
end

# Plans and buffers for one array size and axis: the forward plan, the inverse plan and
# the mode-space buffer. Shared across calls, so not safe for concurrent evaluation.
function fft_workspace!(op::FourierFFTDerivative, X, J)
    return get!(op.plans, (size(X), J)) do
        p = plan_rfft(X, J)
        Xh = p * X
        (p, plan_irfft(Xh, op.n, J), Xh)
    end
end

function MethodOfLines.apply_along(
        op::FourierFFTDerivative{T}, X::AbstractArray{T, N}, ::Val{J}
    ) where {T, N, J}
    size(X, J) == op.n || return apply_along(op.mat, X, Val(J))
    p, pinv, Xh = fft_workspace!(op, X, J)
    mul!(Xh, p, X)
    shape = ntuple(i -> i == J ? length(op.mult) : 1, N)
    Xh .*= reshape(op.mult, shape)
    Y = similar(X)
    mul!(Y, pinv, Xh)
    return Y
end

"""
    ChebyshevFFTDerivative

Whole-direction Chebyshev derivative on `n` Lobatto nodes through the Chebyshev
transform: the DCT-I of a line is the real FFT of its even extension, the
coefficients of the derivative follow from the backward recurrence
`b[k-1] = b[k+1] + 2k a[k]`, applied `order` times, and the same transform maps the
coefficients back to nodal values. `scale` carries the map from `[-1, 1]` to the
domain. Equivalent to the dense matrix `mat`, which is used for element types the
FFT plans do not cover. Plans are cached per array size and axis.
"""
struct ChebyshevFFTDerivative{T, M <: AbstractMatrix{T}}
    n::Int
    order::Int
    scale::T
    mat::M
    plans::Dict{Any, Any}
end

function MethodOfLines.fast_spectral_operator(spec::ChebyshevCollocation, grid, d, mat)
    n = spec.n
    n >= 3 || return nothing
    T = float(eltype(grid))
    has_fft_backend(T, 2 * (n - 1)) || return nothing
    scale = (-2 / T(grid[end] - grid[1]))^d
    return ChebyshevFFTDerivative{T, typeof(mat)}(n, d, scale, mat, Dict{Any, Any}())
end

# Writes the even (mirror) extension of `X` along axis `J` into `V`: `n` nodes become
# `2(n - 1)` samples of one period, so a real FFT along `J` is the DCT-I of each line.
function even_extension!(V, X, J)
    sz = size(X)
    n = sz[J]
    nb = prod(sz[1:(J - 1)])
    na = prod(sz[(J + 1):end])
    Xr = reshape(X, nb, n, na)
    Vr = reshape(V, nb, 2 * (n - 1), na)
    @inbounds for a in 1:na
        for k in 1:n, b in 1:nb
            Vr[b, k, a] = Xr[b, k, a]
        end
        for k in 1:(n - 2), b in 1:nb
            Vr[b, n + k, a] = Xr[b, n - k, a]
        end
    end
    return V
end

# Plan and buffers for one array size and axis: the extension `V`, its transform `U`,
# the coefficient array `A` and two lines for the recurrence. Shared across calls, so
# not safe for concurrent evaluation.
function fft_workspace!(op::ChebyshevFFTDerivative{T}, X::AbstractArray{T, N}, J) where {T, N}
    return get!(op.plans, (size(X), J)) do
        sz = size(X)
        V = zeros(T, ntuple(i -> i == J ? 2 * (sz[J] - 1) : sz[i], N))
        p = plan_rfft(V, J)
        (p, V, p * V, zeros(T, sz), zeros(T, sz[J]), zeros(T, sz[J]))
    end
end

# Coefficients of the derivative of `a`, both in the double-prime convention (first
# and last entries doubled), written into `out`.
function chebyshev_derivative_coefficients!(out::AbstractVector, a::AbstractVector)
    N = length(a) - 1
    out[N + 1] = 0
    out[N] = N * a[N + 1]
    @inbounds for k in (N - 1):-1:1
        out[k] = out[k + 2] + 2k * a[k + 1]
    end
    return out
end

# Applies the recurrence `order` times to every line along axis `J` of `A`, in place.
function chebyshev_derivative_coefficients!(A::AbstractArray, J, order, line, tmp)
    sz = size(A)
    nb = prod(sz[1:(J - 1)])
    na = prod(sz[(J + 1):end])
    Ar = reshape(A, nb, sz[J], na)
    @inbounds for a in 1:na, b in 1:nb
        for k in 1:sz[J]
            line[k] = Ar[b, k, a]
        end
        for _ in 1:order
            chebyshev_derivative_coefficients!(tmp, line)
            line, tmp = tmp, line
        end
        for k in 1:sz[J]
            Ar[b, k, a] = line[k]
        end
    end
    return A
end

function MethodOfLines.apply_along(
        op::ChebyshevFFTDerivative{T}, X::AbstractArray{T, N}, ::Val{J}
    ) where {T, N, J}
    size(X, J) == op.n || return apply_along(op.mat, X, Val(J))
    p, V, U, A, line, tmp = fft_workspace!(op, X, J)
    even_extension!(V, X, J)
    mul!(U, p, V)
    A .= real.(U) ./ (op.n - 1)
    chebyshev_derivative_coefficients!(A, J, op.order, line, tmp)
    even_extension!(V, A, J)
    mul!(U, p, V)
    Y = similar(X)
    Y .= real.(U) .* (op.scale / 2)
    return Y
end

function MethodOfLines.apply_along(
        op::Union{FourierFFTDerivative, ChebyshevFFTDerivative}, X::AbstractArray, ::Val{J}
    ) where {J}
    return apply_along(op.mat, X, Val(J))
end

end
