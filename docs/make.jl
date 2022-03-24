using Documenter, MethodOfLines

makedocs(
    sitename="MethodOfLines.jl",
    authors="Chris Rackauckas, Alex Jones et al.",
    clean=true,
    doctest=false,
    modules=[MethodOfLines],

    format=Documenter.HTML(assets=["assets/favicon.ico"],
                           canonical="https://methodoflines.sciml.ai/stable/"),

    pages=[
        "MethodOfLines.jl: Automated Finite Difference for Phyiscs-Informed Learning" => "index.md",
        "Tutorials" => ["tutorials/brusselator.md", "tutorials/icbc_sampled.md"],
        "MOLFiniteDifference" => "MOLFiniteDifference.md",
        "Boundary Conditions" => "boundary_conditions.md",
        "How it works" => "howitworks.md",
        "Notes for developers: Implement a scheme" => "devnotes.md",
        "Generated Examples" => ["generated/bruss_sys.md", "generated/bruss_ode_eqs.md"]
        # "Tutorial: Burgers" => "tutorials/burgers.md",
        # "Tutorial: 1D Linear Diffusion" => "tutorials/1d_linear_diffusion.md",
        # "Tutorial: 1D Non-Linear Diffusion" => "tutorials/1d_nonlinear_diffusion.md",
        # "Tutorial: 2D Diffusion" => "tutorials/2d_diffusion.md",
        # "Tutorial: 1D Linear Convection" => "tutorials/1d_linear_convection.md",
        # "Tutorial: 1D Higher Order" => "tutorials/1d_higher_order.md",
        # "Tutorial: Stationary Nonlinear Problems" => "tutorials/stationary_nonlinear_problems.md",
        # "Tutorial: 1D Partial DAE" => "tutorials/1d_partial_DAE.md",

     ]
)

deploydocs(
    repo="github.com/SciML/MethodOfLines.jl";
    push_preview=true
)
