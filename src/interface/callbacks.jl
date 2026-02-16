struct MOLDiscCallback
    f::Function
    p::Any
    sym::Any
end

function MOLDiscCallback(f, disc_ps)
    symh = hash(f)
    symname = Symbol("MOLDiscCallback_$symh")
    sym = unwrap(only(@parameters $symname))
    return MOLDiscCallback(f, disc_ps, sym)
end

(M::MOLDiscCallback)(s) = M.f(s, M.p)
