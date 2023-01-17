struct RefCartesianIndex{N,AType} <: Base.AbstractCartesianIndex{N}
    I::CartesianIndex{N}
    A::AType
    RefCartesianIndex(I::CartesianIndex{N}, A=nothing) where {N} = new{N,typeof(A)}(I, A)
end

Base.getindex(A::AbstractArray, IR::RefCartesianIndex) = IR.A === nothing ? A[IR.I] : IR.A[IR.I]
Base.getindex(I::RefCartesianIndex, i::Int) = I.I[i]


function Base.getindex(A::AbstractArray, Is::Vector{<:RefCartesianIndex})
    map(Is) do I
        A[I]
    end
end

Base.:+(I::RefCartesianIndex, J::CartesianIndex) = RefCartesianIndex(I.I + J, I.A)
Base.:-(I::RefCartesianIndex, J::CartesianIndex) = RefCartesianIndex(I.I - J, I.A)
Base.:+(I::CartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I + J.I, J.A)
Base.:-(I::CartesianIndex, J::RefCartesianIndex) = RefCartesianIndex(I - J.I, J.A)

(b::InterfaceBoundary)(I, s, j, isx) = wrapinterface(I, s, b, j, isx)
(b::AbstractBoundary)(I, s, j) = I

# This is a bit of a hacky monad to make single dimensional grids look like arrays of the right dimension for the interface boundary conditions with the nonlinear laplacian
struct OrderedIndexArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    array::A
    index::Int
    function OrderedIndexArray(array::A, index::Int, N::Int) where {T, A<:AbstractArray{T}}
        new{T,N,A}(array, index)
    end
end

Base.getindex(o::OrderedIndexArray, I::CartesianIndex) = o[I[o.index]]
Base.getindex(o::OrderedIndexArray, is...) = o[CartesianIndex(is...)]

function bwrap(I, bs, s, j, isx=false)
    for b in bs
        I = b(I, s, j, isx)
    end
    return I
end

@inline function wrapinterface(I::RefCartesianIndex{N,Nothing}, s::DiscreteSpace, b::InterfaceBoundary, j, isx) where {N}
    return _wrapinterface(I.I, s, b, j, isx)
end

@inline function wrapinterface(I::RefCartesianIndex, s::DiscreteSpace, ::InterfaceBoundary, j, isx)
    return I
end

@inline function wrapinterface(I, s, b::InterfaceBoundary, j, isx)
    return _wrapinterface(I, s, b, j, isx)
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

function _wrapinterface(I, s, b::InterfaceBoundary{Val{false}(),Val{true}()}, j, isx)
    if I[j] <= 1
        u = b.u
        u2 = b.u2
        N = ndims(u, s)
        discu2 = if isx
            OrderedIndexArray(s.grid[b.x2], j, N)
        else
            s.discvars[depvar(u2, s)]
        end
        l2 = length(s, b.x2)
        I1 = unitindex(N, j)
        # update index
        I = I + (l2 - 1) * I1
        I = RefCartesianIndex(I, discu2)
    else
        return RefCartesianIndex(I)
    end
end

function _wrapinterface(I, s, b::InterfaceBoundary{Val{true}(),Val{false}()}, j, isx)
    l1 = length(s, b.x)
    if I[j] > l1
        u = b.u
        u2 = b.u2
        N = ndims(u, s)
        discu2 = if isx
            OrderedIndexArray(s.grid[b.x2], j, N)
        else
            s.discvars[depvar(u2, s)]
        end
        I1 = unitindex(N, j)
        # update index
        I = I + (1 - l1) * I1
        return RefCartesianIndex(I, discu2)
    else
        return RefCartesianIndex(I)
    end
end

function _wrapinterface(I, s, b::InterfaceBoundary{B,B}, j) where {B}
    throw(ArgumentError("Interface $(b.eq) joins two variables at the same end of the domain, this is not supported. Please post an issue if you need this feature."))
end
