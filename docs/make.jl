using Documenter, MethodOfLines

cp("./docs/Manifest.toml", "./docs/src/assets/Manifest.toml", force = true)
cp("./docs/Project.toml", "./docs/src/assets/Project.toml", force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"

include("pages.jl")

makedocs(sitename = "MethodOfLines.jl",
         strict = [
             :doctest,
             :linkcheck,
             :parse_error,
             :example_block,
             # Other available options are
             # :autodocs_block, :cross_references, :docs_block, :eval_block, :example_block, :footnote, :meta_block, :missing_docs, :setup_block
         ],
         authors = "Chris Rackauckas, Alex Jones et al.",
         clean = true, doctest = false, linkcheck = true,
         modules = [MethodOfLines],
         warnonly = [:docs_block, :missing_docs, :cross_references],
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/MethodOfLines/stable/"),
         pages = pages)

deploydocs(repo = "github.com/SciML/MethodOfLines.jl"; push_preview = true)
