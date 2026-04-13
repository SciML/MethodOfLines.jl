# --- nonlinear Laplacian detection ------------------------------------------

"""
    _detect_nonlinlap_terms(pde, s, depvars, exclude_terms=Dict())

Scan PDE terms for nonlinear Laplacian patterns `Dx(expr * Dx(u))`.
Returns the set of matched terms (symbolic expressions).
Terms in `exclude_terms` (e.g., spherical-matched terms) are skipped.
"""
function _detect_nonlinlap_terms(pde, s, depvars, exclude_terms=Dict{Any, NamedTuple}())
    terms = split_terms(pde, s.x̄)
    matched = Set{Any}()
    for u in depvars
        for x in ivs(u, s)
            rules = [
                @rule(*(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => true),
                @rule($(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => true),
                @rule($(Differential(x))($(Differential(x))(u) / ~a) => true),
                @rule(*(~~b, $(Differential(x))($(Differential(x))(u) / ~a), ~~c) => true),
                @rule(/(*(~~b, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~d)), ~~c), ~e) => true),
            ]
            for t in terms
                haskey(exclude_terms, t) && continue
                for r in rules
                    if r(t) !== nothing
                        push!(matched, t)
                        break
                    end
                end
            end
        end
    end
    return matched
end

# --- spherical Laplacian detection ------------------------------------------

"""
    _detect_spherical_terms(pde, s, depvars)

Scan PDE terms for spherical Laplacian patterns `r^{-2} * Dr(r^2 * Dr(u))`.
Returns a `Dict` mapping each matched term to a `NamedTuple` with
`(u, r, innerexpr, outer_coeff)` for template building.
"""
function _detect_spherical_terms(pde, s, depvars)
    # Use split_additive_terms (NOT split_terms) to preserve the complete
    # spherical expression `1/r^2 * Dr(r^2 * Dr(u))` as a single term.
    # split_terms(pde, s.x̄) decomposes it into pieces that the patterns
    # cannot match.
    terms = split_additive_terms(pde)
    matched = Dict{Any, NamedTuple}()
    for u in depvars
        for r in ivs(u, s)
            # Pattern 1: *(~~a, 1/(r^2), Dr(*(~~c, r^2, ~~d, Dr(u), ~~e)), ~~b)
            rule1 = @rule *(
                ~~a,
                1 / (r^2),
                $(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e)),
                ~~b
            ) => (
                u = u, r = r,
                innerexpr = *(~c..., ~d..., ~e..., Num(1)),
                outer_coeff = *(~a..., ~b..., Num(1))
            )

            # Pattern 2: /(*(~~a, Dr(*(~~c, r^2, ~~d, Dr(u), ~~e)), ~~b), r^2)
            rule2 = @rule /(
                *(
                    ~~a, $(Differential(r))(
                        *(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e)
                    ), ~~b
                ),
                (r^2)
            ) => (
                u = u, r = r,
                innerexpr = *(~c..., ~d..., ~e..., Num(1)),
                outer_coeff = *(~a..., ~b..., Num(1))
            )

            # Pattern 3: /(Dr(*(~~c, r^2, ~~d, Dr(u), ~~e)), r^2)
            rule3 = @rule /(
                ($(Differential(r))(*(~~c, (r^2), ~~d, $(Differential(r))(u), ~~e))),
                (r^2)
            ) => (
                u = u, r = r,
                innerexpr = *(~c..., ~d..., ~e..., Num(1)),
                outer_coeff = Num(1)
            )

            rules = [rule1, rule2, rule3]
            for t in terms
                haskey(matched, t) && continue
                for rl in rules
                    result = rl(t)
                    if result !== nothing
                        matched[t] = result
                        break
                    end
                end
            end
        end
    end
    return matched
end

