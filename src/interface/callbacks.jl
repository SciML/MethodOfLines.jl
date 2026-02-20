struct MOLDiscCallback
    f::Function
    p::Any
    sym::Any
end

function MOLDiscCallback(f, disc_ps)
    symh = hash(f)
    name = Symbol("MOLDiscCallback_$symh")
    sym = unwrap(Symbolics.variable(name))
    return MOLDiscCallback(f, disc_ps, sym)
end

(M::MOLDiscCallback)(s) = M.f(s, M.p)
