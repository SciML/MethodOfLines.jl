struct MOLDiscCallback
    f::Function
    p::Any
    sym::Any
end

function MOLDiscCallback(f, disc_ps)
    symh = hash(f)
    sym = unwrap(Symbolics.variable(Symbol("MOLDiscCallback_$symh"), T=Real))
    return MOLDiscCallback(f, disc_ps, sym)
end

(M::MOLDiscCallback)(s) = M.f(s, M.p)
