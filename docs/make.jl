using Documenter, DiffEqOperators

makedocs(
    sitename="MethodOfLines.jl",
    authors="Chris Rackauckas et al.",
    clean=true,
    doctest=false,
    modules=[MethodOfLines],

    format=Documenter.HTML(assets=["assets/favicon.ico"],
                           canonical="https://methodoflines.sciml.ai/stable/"),

    pages=[
        "MethodOfLines.jl: Automated Finite Difference for Phyiscs-Informed Learning" => "index.md",
        "Operator Tutorials" => [
            "operator_tutorials/kdv.md"
        ],
        "Operators" => [
            "operators/operator_overview.md",
            "operators/derivative_operators.md",
            "operators/vector_calculus_operators.md",
            "operators/vector_jacobian_product.md",
            "operators/jacobian_vector_product.md",
            "operators/matrix_free_operators.md"
        ],
        "Nonlinear Derivatives" => [
            "nonlinear_derivatives/nonlinear_diffusion.md"
        ]
     ]
)

deploydocs(
    repo="github.com/SciML/DiffEqOperators.jl";
    push_preview=true
)
