using Documenter, MethodOfLines

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
     ]
)

deploydocs(
    repo="github.com/SciML/MethodOfLines.jl";
    push_preview=true
)
