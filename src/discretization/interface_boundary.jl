struct RefCartesianIndex{N, AType} <: Base.AbstractCartesianIndex{N}
    I::CartesianIndex{N}
    A::AType
    RefCartesianIndex(I::CartesianIndex{N}, A = nothing) where {N} = new{N, typeof(A)}(I, A)
end

Base.getindex(A::Array, IR::RefCartesianIndex) = IR.A === nothing ? A[IR.I] : IR.A[IR.I]
Base.getindex(I::RefCartesianIndex, i::Int) = I.I[i]

function Base.getindex(A::Array, Is::Vector{<:RefCartesianIndex})
    return map(Is) do I
        A[I]
    end
end

Base.:+(I::RefCartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I.I + J.I, I.A)
Base.:-(I::RefCartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I.I - J.I, I.A)
Base.:+(I::RefCartesianIndex, J::CartesianIndex) = RefCartesianIndex(I.I + J, I.A)
Base.:-(I::RefCartesianIndex, J::CartesianIndex) = RefCartesianIndex(I.I - J, I.A)
Base.:+(I::CartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I + J.I, J.A)
Base.:-(I::CartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I - J.I, J.A)

boundary_index(I, s, b::InterfaceBoundary, jx) = wrapinterface(I, s, b, jx)
boundary_index(I, s, ::AbstractBoundary, jx) = I

function bwrap(I, bs, s, jx)
    for b in bs
        I = boundary_index(I, s, b, jx)
    end
    return I
end

@inline function _wrapperiodic(I, N, j, l)
    I1 = unitindex(N, j)
    # shift l-1: u[1] ~ u[end]
    if I[j] <= 1
        I = I + I1 * (l - 1)
    elseif I[j] > l
        I = I - I1 * (l - 1)
    end
    return I
end

"""
Wrap stencil indices across interface/periodic boundaries.
"""

function wrapinterface(
        I::RefCartesianIndex{N, Nothing}, s::DiscreteSpace,
        b::InterfaceBoundary, jx
    ) where {N}
    j, x = jx
    return _wrapinterface(I.I, s, b, j)
end

@inline function wrapinterface(
        I::RefCartesianIndex, s::DiscreteSpace, ::InterfaceBoundary, jx
    )
    return I
end

function wrapinterface(I, s, b::InterfaceBoundary, jx)
    j, x = jx

    return _wrapinterface(I, s, b, j)
end

function get_interface_vars(b, s, j)
    u = b.u
    u2 = b.u2
    discu2 = s.discvars[depvar(u2, s)]
    l1 = length(s, b.x)
    l2 = length(s, b.x2)
    N = ndims(u, s)
    I1 = unitindex(N, j)
    return I1, discu2, l1, l2
end

"""
    interface_layout_compatible(s, b, j)

Whether interface `b` may write a `CartesianIndex` of `b.u` into `b.u2`.

The pointwise wrap and face equations index the partner at the same slot `j`
plus a shift along that axis. That is valid only when `b.x2` occupies argument
position `j` in `b.u2`, the two variables have the same number of spatial
arguments, and every non-interface axis has the same discrete length. A
different layout is not remapped.
"""
function interface_layout_compatible(s, b, j)
    u = depvar(b.u, s)
    u2 = depvar(b.u2, s)
    j2 = x2i(s, u2, b.x2)
    (
        j2 !== nothing && j2 == j &&
            ndims(u, s) == ndims(u2, s)
    ) || return false
    disc1 = s.discvars[u]
    disc2 = s.discvars[u2]
    return all(i -> i == j || size(disc1, i) == size(disc2, i), 1:ndims(u, s))
end

function check_interface_layout(s, b, j)
    interface_layout_compatible(s, b, j) || throw(
        ArgumentError(
            "Interface $(b.eq) joins variables with incompatible layout. " *
                "The interface axis must occupy the same argument position in both " *
                "variables, and every other axis must have the same discrete length."
        )
    )
    return nothing
end

function _wrapinterface(I, s, b::InterfaceBoundary{Val{false}(), Val{true}()}, j)
    if I[j] <= 1
        check_interface_layout(s, b, j)
        u = b.u
        u2 = b.u2
        discu2 = s.discvars[depvar(u2, s)]
        l2 = length(s, b.x2)
        N = ndims(u, s)
        I1 = unitindex(N, j)
        I = I + (l2 - 1) * I1
        I = RefCartesianIndex(I, discu2)
    else
        return RefCartesianIndex(I)
    end
end

function _wrapinterface(I, s, b::InterfaceBoundary{Val{true}(), Val{false}()}, j)
    l1 = length(s, b.x)
    if I[j] > l1
        check_interface_layout(s, b, j)
        u = b.u
        u2 = b.u2
        discu2 = s.discvars[depvar(u2, s)]
        N = ndims(u, s)
        I1 = unitindex(N, j)
        I = I + (1 - l1) * I1
        return RefCartesianIndex(I, discu2)
    else
        return RefCartesianIndex(I)
    end
end

function _wrapinterface(I, s, b::InterfaceBoundary{B, B}, j) where {B}
    throw(ArgumentError("Interface $(b.eq) joins two variables at the same end of the domain, this is not supported. Please post an issue if you need this feature."))
end

"""
    bcoord(I, bs, s, jx)

Physical coordinate of raw tap index `I` in the differentiated grid's chart. Wrapped taps
use the exact interface chart transition: periodic shift = period length, contiguous
shift = 0. Result is strictly increasing across the seam.

Mirrored bit for bit by `array_periodic_coord` for self-periodic directions and by
`array_wrap_coord` for two-domain interfaces; keep them in lockstep.
"""
function bcoord(I, bs, s, jx)
    j, x = jx
    for b in bs
        c = _wrapcoord(I, s, b, j)
        c === nothing || return c
    end
    return s.grid[x][I[j]]
end

_wrapcoord(I, s, b::AbstractBoundary, j) = nothing

function _wrapcoord(I, s, b::InterfaceBoundary{Val{false}(), Val{true}()}, j)
    if I[j] <= 1
        grid1 = s.grid[b.x]
        grid2 = s.grid[b.x2]
        i2 = I[j] + length(grid2) - 1
        # grid2 upper edge ≡ grid1 lower edge
        return grid2[i2] - (grid2[end] - grid1[1])
    else
        return nothing
    end
end

function _wrapcoord(I, s, b::InterfaceBoundary{Val{true}(), Val{false}()}, j)
    grid1 = s.grid[b.x]
    if I[j] > length(grid1)
        grid2 = s.grid[b.x2]
        i2 = I[j] + 1 - length(grid1)
        # grid2 lower edge ≡ grid1 upper edge
        return grid2[i2] + (grid1[end] - grid2[1])
    else
        return nothing
    end
end
