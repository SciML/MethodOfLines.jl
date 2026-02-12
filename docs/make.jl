using Documenter, MethodOfLines

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
ENV["JULIA_DEBUG"] = Documenter

include("pages.jl")

makedocs(
    sitename = "MethodOfLines.jl",
    authors = "Chris Rackauckas, Alex Jones et al.",
    clean = true, doctest = false, linkcheck = true,
    modules = [MethodOfLines],
    warnonly = [:docs_block, :missing_docs, :cross_references],
    linkcheck_ignore = [
        # Returns 403 from CI; see https://github.com/SciML/MethodOfLines.jl/issues/518
        "https://docs.sciml.ai/ModelingToolkit/stable/systems/ODESystem/#Standard-Problem-Constructors",
    ],
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/MethodOfLines/stable/"
    ),
    pages = pages
)

deploydocs(repo = "github.com/SciML/MethodOfLines.jl"; push_preview = true)
