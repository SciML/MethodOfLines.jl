using Documenter, MethodOfLines

# Determine the correct path based on where the script is run from
docs_dir = @__DIR__
root_dir = dirname(docs_dir)

# If we're already in the docs directory, adjust paths accordingly
if basename(pwd()) == "docs"
    manifest_src = "Manifest.toml"
    project_src = "Project.toml"
else
    manifest_src = joinpath("docs", "Manifest.toml")
    project_src = joinpath("docs", "Project.toml")
end

assets_dir = joinpath(docs_dir, "src", "assets")
mkpath(assets_dir)  # Ensure the assets directory exists

cp(manifest_src, joinpath(assets_dir, "Manifest.toml"), force = true)
cp(project_src, joinpath(assets_dir, "Project.toml"), force = true)

# Make sure that plots don't throw a bunch of warnings / errors!
ENV["GKSwstype"] = "100"
ENV["JULIA_DEBUG"] = Documenter

include("pages.jl")

makedocs(sitename = "MethodOfLines.jl",
    authors = "Chris Rackauckas, Alex Jones et al.",
    clean = true, doctest = false, linkcheck = true,
    modules = [MethodOfLines],
    warnonly = [:docs_block, :missing_docs, :cross_references],
    format = Documenter.HTML(assets = ["assets/favicon.ico"],
        canonical = "https://docs.sciml.ai/MethodOfLines/stable/"),
    pages = pages)

deploydocs(repo = "github.com/SciML/MethodOfLines.jl"; push_preview = true)
