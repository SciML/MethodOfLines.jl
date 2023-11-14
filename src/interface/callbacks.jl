struct MOLDiscCallback
    f::Function
    p
    sym
end

function MOLDiscCallback(f, disc_ps)
    symh = hash(f)
    sym = unwrap(first(@parameters(Symbol("MOLDiscCallback_$symh"))))
    return MOLDiscCallback(f, disc_ps, sym)
end

(M::MOLDiscCallback)(s) = M.f(s, M.p)