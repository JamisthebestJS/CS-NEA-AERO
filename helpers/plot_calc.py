#calaulates forces and stuff for the graph plotting.
#This may be temporary and be moved to LBM since there is where all the stuff is, but may like push the info across to here, not sure yet.

"""
Need to calculate lift and drag.


Lift could be done 2ways, or maybe as both and sum:
- calc pressure differential between top and bottom of aerofoil multiplied by area
- calc shear forces on surface of aerofoil and sum

drag could be done as:
- calc pressure differential between font and back of aerofoil mult by area
- calc shear forces on front and back and sum


DRAG:

Fd = 0.5(ρ v² Cd A)
Fd = drag force - calculating
ρ = fluid density - known
v = flow velocity - known
Cd = drag coefficient - unknown
A = reference area - known (projected frontal area)

but, how to calculate drag coefficient without ^^ this?






post-process results to extract pressure and shear stress data.
Integrate these data to find lift and drag forces.
Divide forces by reference area and dynamic pressure to get lift and drag coefficients.


still researching this
"""