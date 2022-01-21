
using DiffEqOperators

"""
Inner function for `get_half_offset_weights_and_stencil` (see below).
"""
function _get_weights_and_stencil(D, I)
    # k => i of itap - 
    # offset is important due to boundary proximity
    # The low boundary coeffs has a heirarchy of coefficients following: number of indices from boundary -> which half offset point does it correspond to -> weights
    if I <= (D.boundary_point_count)
        println("low boundary")
        weights = D.low_boundary_coefs[I]    
        offset = 1 - I
        Itap = [I + (i+offset) for i in 0:(D.boundary_stencil_length-1)]
    elseif I > (20 - D.boundary_point_count)
        println("high boundary")

        weights = D.high_boundary_coefs[20-I+1]
        offset = 20 - I
        Itap = [I + (i+offset) for i in (-D.boundary_stencil_length+1):1:0]
    else
        println("interior")

        weights = D.stencil_coefs
        Itap = [I + i for i in (1-div(D.stencil_length,2)):(div(D.stencil_length,2))]
    end    

    return (weights, Itap)
end


function clip(I, n)
    return I>(n/2) ? I-1 : I+1
    
end

for I in [2,19, 3, 18, 10]
    
    D =  CompleteHalfCenteredDifference(1, 6, 1.0)
    
     
    
    outerweights, outerstencil = _get_weights_and_stencil(D, clip(I,20))
    
    # Get the correct weights and stencils for this II
    
    @show I
    @show outerweights
    @show outerstencil
    
    
    for II in outerstencil
        println()
        println("II: ", II)
        println()
        weights, stencil = _get_weights_and_stencil(D, II)
        println(weights)
        println(stencil)
    end

end