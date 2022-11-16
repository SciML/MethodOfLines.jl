struct RefCartesianIndex{N, AType}
    I::CartesianIndex{N}
    A::AType
    RefCartesianIndex(I::CartesianIndex{N}, A::AType) where {N, AType} = new{N, AType}(I, A)
end

getindex(_, IR::RefCartesianIndex) = A[IR.I]

(b::InterfaceBoundary)(I, s, jx) = wrapinterface(I, s, b, jx)
(b::InterfaceBoundary)(I::RefCartesianIndex, s, jx) = I
(b::AbstractBoundary)(I, s, jx) = I

function bwrap(I, bs, s, jx)
    for b in bs
        I = b(I, s, jx)
    end
    return I
end

@inline function _wrapperiodic(I, N, j, l)
    I1 = unitindex(N, j)
    # -1 because of the relation u[1] ~ u[end]
    if I[j] <= 1
        I = I + I1 * (l - 1)
    elseif I[j] > l
        I = I - I1 * (l - 1)
    end
    return I
end

"""
Allow stencils indexing over periodic boundaries. Index through this function.
"""
function wrapinterface(I, s, ::PeriodicBoundary, u, jx)
    j, x = jx
    return _wrapperiodic(I, ndims(u, s), j, length(s, x))
end

function wrapinterface(I, s, ::Val{false}, u, jx)
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
    return I1, discu2, l2
end

function _wrapinterface(I, s, b::InterfaceBoundary{Val{false}, Val{true}}, j)
    I1, discu2, l1, l2 = get_interface_vars(b, s, j)
    if I[j] <= 1
        I = I + (l2 - 1) * I1
        I = RefCartesianIndex(I, discu2)
    end
    return I
end

function _wrapinterface(I, s, b::InterfaceBoundary{Val{true},Val{false}}, j)
    I1, discu2, l1, l2 = get_interface_vars(b, s, j)
    if I[j] > l1
        I = I + (I[j] - 2l1 + 1) * I1
        I = RefCartesianIndex(I, discu2)
    end
    return I
end

function _wrapinterface(I, s, b::InterfaceBoundary{B, B}, j) where B
    throw(ArgumentError("Interface $(b.eq) joins two variables at the same boundary end, this is not supported. Please post an issue if you need this feature."))
end
