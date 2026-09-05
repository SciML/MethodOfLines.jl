using Test, MethodOfLines, ModelingToolkitBase, Symbolics, SymbolicUtils, SciMLBase
using SciMLBase: symbolic_discretize

struct RecordingArrayObserved{A, B}
    arrays::A
    bases::B
    requests::Set{Any}
end

function (record::RecordingArrayObserved)(symbol, state, parameters, time)
    symbol = Symbolics.unwrap(symbol)
    push!(record.requests, symbol)
    for (array, basis) in zip(record.arrays, record.bases)
        if isequal(symbol, array)
            return basis * state
        elseif SymbolicUtils.iscall(symbol) && SymbolicUtils.operation(symbol) === getindex
            args = SymbolicUtils.arguments(symbol)
            isequal(first(args), array) || continue
            indices = SymbolicUtils.unwrap_const.(args[2:end])
            return (basis * state)[indices...]
        end
    end
    error("unexpected observed request: $symbol")
end

@testset "array reconstruction queries are field-based" begin
    for intervals in (5, 12)
        @independent_variables t x
        @variables v(..) w(..)
        D = Differential(t)
        @named pde = PDESystem(
            [D(v(t, x)) ~ -v(t, x), D(w(x, t)) ~ -w(x, t)],
            [v(0, x) ~ 1 + x, w(x, 0) ~ 2 - x],
            [t ∈ (0.0, 0.2), x ∈ (0.0, 1.0)], [t, x], [v(t, x), w(x, t)]
        )
        source, _ = symbolic_discretize(pde, MOLFiniteDifference([x => 1 / intervals], t))
        metadata = SymbolicUtils.getmetadata(source, ModelingToolkitBase.ProblemTypeCtx, nothing)
        arrays = map([v(t, x), w(x, t)]) do field
            scalar = Symbolics.unwrap(first(metadata.discretespace.discvars[field]))
            first(SymbolicUtils.arguments(scalar))
        end
        grid = collect(range(0.0, 1.0; length = intervals + 1))
        bases = [hcat(ones(length(grid)), grid), hcat(2ones(length(grid)), -grid)]
        @variables reduced(t)[1:2]
        @named rom = System(
            [D(reduced) ~ -reduced], t, Symbolics.unwrap.(Symbolics.scalarize(reduced)), [];
            observed = [array ~ basis * reduced for (array, basis) in zip(arrays, bases)]
        )
        record = RecordingArrayObserved(arrays, bases, Set{Any}())
        rhs! = (du, u, p, t) -> (du .= -u)
        f = ODEFunction{true}(rhs!; sys = complete(rom), observed = record)
        prob = ODEProblem(f, ones(2), (0.0, 0.2))
        times = [0.0, 0.1, 0.2]
        states = [fill(exp(-time), 2) for time in times]
        raw_solution = SciMLBase.build_solution(prob, nothing, times, states)
        solution = SciMLBase.PDETimeSeriesSolution(raw_solution, metadata)
        @test solution[v(t, x)] ≈ exp.(-times) * (1 .+ grid)'
        @test solution[w(x, t)] ≈ (2 .- grid) * exp.(-times)'
        @test length(record.requests) == 2
    end
end
