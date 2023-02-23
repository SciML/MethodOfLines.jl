
isarr(x) = symtype(x) isa AbstractArray

#TODO missing primitives for broadcasting over broadcasted objects

#* Assuming no nested ArrayMakers
#* assuming no mapreduce objects

#! Unfinished

"""
Should fold any broadcasts to inner ArrayOps, and fold inner ArrayOps
"""
function fold(term)
    if !istree(term)
        return term
    end
    if term isa ArrayMaker
        args = arguments(term)
        T = args[1]
        pairs = args[3] .=> fold.(args[4:end])
        return Construct_ArrayMaker{T}(args[2], pairs)
    end
    op = operation(term)
    if op isa typeof(broadcast)
        return broadcast_reduce(op, arguments(term)...)
    end

end

broadcast_reduce(f, x...) = reduce((a, b) -> broadcast_reduce(f, a, b), x)


function broadcast_reduce(f, a::Number, b::ArrayMaker)
    args = arguments(b)
    T = args[1]

    pairs = args[3] .=> broadcast_reduce.((f,), (a,), args[4:end])
    return Construct_ArrayMaker{T}(args[2], pairs)
end

function broadcast_reduce(f, a::ArrayMaker, b::Number)
    args = arguments(a)
    T = args[1]

    pairs = args[3] .=> broadcast_reduce.((f,), args[4:end], (b,))
    return Construct_ArrayMaker{T}(args[2], pairs)
end

function broadcast_reduce(f, a::ArrayMaker, b::ArrayMaker)
    args1 = arguments(a)
    args2 = arguments(b)
    T = promote_type(args1[1], args2[1])
    @assert args1[2] == args2[2] "Dimension mismatch: sizes of the following ArrayMakers are not equal: $a and $b"
    pairs1 = args1[3] .=> fold.(args1[4:end])
    pairs2 = args2[3] .=> fold.(args2[4:end])
    pairs = broadcast_reduce(f, pairs1, pairs2)
    return Construct_ArrayMaker{T}(args1[2], pairs)
end

function broadcast_reduce(f, a::Vector{<:Pair}, b::Vector{<:Pair})
    pairs = []
    for p1 in a
        for p2 in b
            r = _intersection(p1, p2)
            if r !== nothing
                push!(pairs, r => broadcast_reduce(f, r[2][1], r[2][2]))
            end
            r = _difference(p1, p2)
            if r !== nothing
                if f in (+, -)
                    push!(pairs, r)
                end
            end
            r = _difference(p2, p1)
            if r !== nothing
                if isequal(f, +)
                    push!(pairs, r)
                elseif isequal(f, -)
                    push!(pairs, r => -r[2])
                end
            end
        end
    end
    # ? is this necessary?
    map(enumerate(pairs)) do (i, p)
        mates = filter(enumerate(pairs)) do (j, q)
            !(i == j) && isequal(p[1], q[1])
        end
        mates = map(m -> m[2], mates)
        if length(mates) > 0
            p[1] => broadcast_reduce(f, p[2], mates...)
        else
            p
        end
    return pairs
end

function broadcast_reduce(f, a::ArrayOp, b::ArrayOp)
    args1 = arguments(a)
    args2 = arguments(b)
    is1 = args1[1]
    is2 = args2[1]
    expr = f(args1[2], args2[2])
    @assert all(isequal.(is1, is2)) "reducing indices different for $a and $b, got $is1 and $is2"
    @assert args1[3] == args2[3] "reducing ops different for $a and $b"
    ranges = _intersection.(map(i_x -> args1[6][i_x], is1), map(i_x -> args2[6][i_x], is2))

    return FillArrayOp(expr, is, ranges)
end

function broadcast_reduce(f, a::Number, b::ArrayOp)
    args = arguments(b)
    is = args[1]
    expr = f(a, args[6])
    ranges = map(i_x -> args[6][i_x], is)

    return FillArrayOp(expr, is, ranges)
end

function broadcast_reduce(f, a::ArrayOp, b::Number)
    args = arguments(a)
    is = args[1]
    expr = f(args[2], b)
    ranges = map(i_x -> args[6][i_x], is)

    return FillArrayOp(expr, is, ranges)
end

broadcast_reduce(f, a::Number, b::Number) = f(a, b)

function _intersection(r1::AbstractRange, r2::AbstractRange)
    r = max(r1[1], r2[1]):min(r1[end], r2[end])
    if length(r) == 0
        return nothing
    else
        return r
    end
end

function _intersection(p1::Pair, p2::Pair)
    r = _union(p1[1], p2[1])
    if r === nothing
        return nothing
    else
        return r => vcat(p1[2], p2[2])
    end
end

function _difference(r1::AbstractRange, r2::AbstractRange)
    r = r1[1]:r2[1]-1
    if length(r) == 0
        return nothing
    else
        return r
    end
end

function _difference(p1::Pair, p2::Pair)
    r = _difference(p1[1], p2[1])
    if r === nothing
        return nothing
    else
        return r => p1[2]
    end
end
