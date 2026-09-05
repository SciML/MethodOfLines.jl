module MethodOfLinesModelingToolkitNeuralNetsExt

using MethodOfLines: MethodOfLines
using ModelingToolkitNeuralNets: ModelingToolkitNeuralNets
using SymbolicUtils: BasicSymbolic

# Lux networks take one column per sample, so a whole field slice is one call.
MethodOfLines.batched_callable(f::BasicSymbolic) = ModelingToolkitNeuralNets.isneuralnetwork(f)

end
