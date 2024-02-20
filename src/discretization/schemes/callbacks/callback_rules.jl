function generate_cb_rules(II, s, depvars, derivweights, bmap, indexmap, terms)
    cbs = derivweights.callbacks
    cb_rules = [cb.sym => cb(s) for cb in cbs]
    return cb_rules
end
