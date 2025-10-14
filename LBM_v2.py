# LBM attempt 2
r"""
------

Lattice Boltzmann Computations

Density:

ρ = ∑ᵢ fᵢ


Velocities:

u = 1/ρ ∑ᵢ fᵢ cᵢ


Equilibrium:

fᵢᵉ = ρ Wᵢ (1 + 3 cᵢ ⋅ u + 9/2 (cᵢ ⋅ u)² − 3/2 ||u||₂²)


BGK Collision:

fᵢ ← fᵢ − ω (fᵢ − fᵢᵉ)


with the following quantities:

fᵢ  : Discrete velocities
fᵢᵉ : Equilibrium discrete velocities
ρ   : Density
∑ᵢ  : Summation over all discrete velocities
cᵢ  : Lattice Velocities
Wᵢ  : Lattice Weights
ω   : Relaxation factor

------

The flow configuration is defined using the Reynolds Number

Re = (U R) / ν

with:

Re : Reynolds Number
U  : Inflow Velocity
R  : Cylinder Radius
ν  : Kinematic Viscosity

Can be re-arranged in terms of the kinematic viscosity

ν = (U R) / Re

Then the relaxation factor is computed according to

ω = 1 / (3 ν + 0.5)

------

Note that this scheme can become unstable for Reynoldsnumbers >~ 350 ²

"""

import pygame
import numpy as np

global count
count = 0


#constants
REYNOLDS = 100

N_POINTS_X = 600
N_POINTS_Y = 300

SCREEN_COEF = 4

SCREEN_HEIGHT = N_POINTS_Y * SCREEN_COEF
SCREEN_WIDTH = N_POINTS_X * SCREEN_COEF

MAX_RENDERED_VAL = 500

"""
need to define inner boundary for aerofoil
being done in point_drawing, saved as a True/False field, and then loaded here at beginning of simulation

"""

MAX_HORIZONTAL_INFLOW_VEL = 500

PLOT_N_STEPS = 25
SKIP_FIRST_N_STEPS = 0

N_DISCRETE_VELOCITIES = 9

LATTICE_VELS = np.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1,],
    [0, 0, 1, 0, -1, 1, 1, -1, -1,]])

LATTICE_INDICES = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
OPPOSITE_LATTICE_INDICES = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

LATTICE_WEIGHTS = np.array([ #these seem arbitrary, but look at writeup for explanation (need to write it first though)
    4/9, #centre
    1/9, 1/9, 1/9, 1/9, #cardinals
    1/36, 1/36, 1/36, 1/36]) #diagonals

RIGHT_VELS = np.array([1, 5, 8])
LEFT_VELS = np.array([3, 6, 7])
TOP_VELS = np.array([2, 5, 6])
BOTTOM_VELS = np.array([4, 7, 8])
VERTICAL_VELS = np.array([0, 2, 4])
HORIZONTAL_VELS = np.array([0, 1, 3])

global density


class Helpers(staticmethod):
        
    @staticmethod
    def get_density(discrete_vels):
        # ρ = ∑ᵢ fᵢ
        density = np.sum(discrete_vels, axis=-1)
        return density

    @staticmethod
    def get_macro_velocities(discrete_vels, density):
        # u = 1/ρ ∑ᵢ fᵢ cᵢ
        
        #debugging
        #print("a", (np.einsum('NMQ,dQ->NMd', discrete_vels, LATTICE_VELS)).shape)
        #print("b", (density[..., np.newaxis]).shape)
        
        #∑ᵢ fᵢ cᵢ
        temp = np.einsum('NMQ,dQ->NMd', discrete_vels, LATTICE_VELS)
        # u = 1/ρ ∑ᵢ fᵢ cᵢ
        macro_vels = temp[:, :, :] / density[..., np.newaxis]
        """
        NM means an array that is M x N (rows x cols)
        Q means 9 discrete (mesocopic) velocities
        d means 2 macroscopic velocities (x and y)
        
        So converting from 9 discrete velocities to 2 macroscopic velocities to get a velocity in terms of x and y
        
        The weird stuff in the density is to sort of force density to be a 3rd rank tensor, so it can be broadcasted properly (i.e. cant divide 3d by 2d so make 2d 3d then divide)
        """
        
        
        return macro_vels

    @staticmethod
    def get_equilibrium_velocities(macro_vels, density):
        # fᵢᵉ = ρ Wᵢ (1 + 3 cᵢ ⋅ u + 9/2 (cᵢ ⋅ u)² − 3/2 ||u||₂²)

        #proj_discrete_vels = np.einsum('dQ,NMd->NMQ', macro_vels, LATTICE_VELS)
        #cᵢ
        
        #debugging
        #global count
        #count+=1
        #print("iter", count)
        #print("a", LATTICE_VELS.shape)
        #print("b", macro_vels.shape)
        
        #print("p_d_v s", np.einsum("dQ,NMd->NMQ", LATTICE_VELS, macro_vels).shape)
        proj_discrete_vels = np.einsum("dQ,NMd->NMQ", LATTICE_VELS, macro_vels)

        #||u||₂
        #print("m_v_m s", np.linalg.norm(macro_vels, axis=-1, ord=2).shape)
        macro_vels_mag = np.linalg.norm(macro_vels, axis=-1, ord=2) #macro_vels over the last axis, and euclidian norm
        
        #debugging
        #print("a", (density[..., np.newaxis]).shape)
        #print("b", (LATTICE_WEIGHTS[np.newaxis, np.newaxis, :]).shape)
        #print("c", (1 + 3 * proj_discrete_vels + 9/2 * proj_discrete_vels**2 - 3/2 * macro_vels_mag[..., np.newaxis]**2).shape)
        #print("d", (density[..., np.newaxis] * LATTICE_WEIGHTS[np.newaxis, np.newaxis, :]).shape)
        
        #maths according to the equation above
        equilibrium_discrete_vels = (
            density[..., np.newaxis] * 
            LATTICE_WEIGHTS[np.newaxis, np.newaxis, :] * (
                1 + 3 * proj_discrete_vels + 9/2 * proj_discrete_vels**2 - 3/2 * macro_vels_mag[..., np.newaxis]**2 
                ) 
            )
        

        return equilibrium_discrete_vels

    @staticmethod
    def load_aerofoil():
        mask = []
        with open("Aerofoils\\aerofoil 1.txt", "r")as file:
            for line in file:
                mask.append(line)

        return mask

def render(density, screen):
    ob_mask = []
    #each sim square is a 2x2 rendered square.
    #each square, if not boundary, is rendered as a colour depending on density of the square
    count = 0
    for i in range(len(density)):
        for j in range(len(density[i])):
            count+=1
            local_dens = density[i, j]
            if local_dens > 0:
                #want to know the percentage out of some max value and then multiply the percentage by 255 for the colour shade
                val = min(255, local_dens)
                multiplier = val / MAX_RENDERED_VAL
                
                pygame.draw.rect(screen, (255*multiplier, 0, 0), pygame.Rect(i*SCREEN_COEF, j*SCREEN_COEF, 2, 2))

            
            #elif ob_mask[i][j] == True: #if aerofoil:
                #pygame.draw.rect(screen, (255,255,255), pygame.Rect(i*SCREEN_COEF, j*SCREEN_COEF, 2, 2))

    pygame.display.update()
    pygame.time.delay(2000)


def update():
    #temporary
    ob_mask = Helpers.load_aerofoil() #I think this doesnt work quite right? requires a list of indices I think rather than True/False
    # 1. Apply outflow boundary condition on the right boundary
    discrete_vels_prev[-1, :, LEFT_VELS] = discrete_vels_prev[-2, :, LEFT_VELS] #(bounday stuff has same value as stuff one cell further left)

    # 2. Compute Macroscopic Quantities (density and velocities)
    density_prev = Helpers.get_density(discrete_vels_prev)
    macro_vels_prev = Helpers.get_macro_velocities(discrete_vels_prev, density_prev)

    # 3. Apply inflow stuff by Zou/He  (Dirichlet BC)
    macro_vels_prev[0, 1:-1, :] = velocity_profile[0, 1:-1, :] #at all points but very top and very bottom
    
    
    #maths according to Zou/He
    density_prev[0, :] = (Helpers.get_density(discrete_vels_prev[0, :, VERTICAL_VELS].T) + 2 * Helpers.get_density(discrete_vels_prev[0, :, LEFT_VELS].T))\
        / (1 - macro_vels_prev[0, :, 0])
    
    # 4. calc the discrete equilibria velocities
    equilibrium_discrete_vels = Helpers.get_equilibrium_velocities(macro_vels_prev, density_prev)
    
    #more Zou/He
    discrete_vels_prev[0, :, RIGHT_VELS] = (equilibrium_discrete_vels[0, :, RIGHT_VELS])
    
    # 5. BGK collisions
    # fᵢ ← fᵢ − ω (fᵢ − fᵢᵉ)
    discrete_vels_post_collision = discrete_vels_prev - relaxation_factor * (discrete_vels_prev - equilibrium_discrete_vels)
    ob_mask = []
    # 6. bounce-back (for no-slip on interior boundary)
    for i in range(N_DISCRETE_VELOCITIES):
        #basically, anything that is now inside the aerfoil is reversed so it "bounces back" from the boundary
        discrete_vels_post_collision[ob_mask, LATTICE_INDICES[i]] = discrete_vels_prev[ob_mask, OPPOSITE_LATTICE_INDICES[i]]
        
    # 7. streaming along lattice vels
    discrete_vels_streamed = discrete_vels_post_collision
    for i in range(N_DISCRETE_VELOCITIES):
        #e.g. velocity going diagonally down and right goes to the cell which is diagonally down and right from where it was
        #etc. for all 9 (8 going outwards) velocities
        discrete_vels_streamed[:, :, i] = np.roll(
            np.roll(
                discrete_vels_post_collision[:, :, i],
                LATTICE_VELS[0, i], axis = 0),
                
            LATTICE_VELS[1, i], axis = 0
            )
    
    return discrete_vels_streamed


def LBM_main():
    #SETUP
    kinematic_viscosity = (MAX_HORIZONTAL_INFLOW_VEL * 10) / REYNOLDS   #NOTE: 10 is radius, but an aerofoil doesnt have a radius, so not sure how that works
    
    global relaxation_factor
    relaxation_factor = 1 / (3 * kinematic_viscosity + 0.5)

    #def mesh
    x = np.arange(N_POINTS_X)
    y = np.arange(N_POINTS_Y)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    #velocity profile
    global velocity_profile
    velocity_profile = np.zeros((N_POINTS_X, N_POINTS_Y, 2))
    velocity_profile[:, :, 0] = MAX_HORIZONTAL_INFLOW_VEL #horizontal velocity profile

    global discrete_vels_prev
    discrete_vels_prev = Helpers.get_equilibrium_velocities(velocity_profile, np.ones((N_POINTS_X, N_POINTS_Y)))
    
    print("finished setup")
    return True



def LBM_main_loop(screen):
    screen.fill("black")

    global discrete_vels_prev
    discrete_vels_next = update()
    
    discrete_vels_prev = discrete_vels_next
    density = Helpers.get_density(discrete_vels_next)
    render(density, screen)


"""
#TODO:
- potentially implement jax since its a lot faster.


ERRORS:
-


BUGS:
- nothing renders (I think due to no injected density?)






#DONE


FEATURES:
-



ERRORS:
- currently lots of array stuff isnt working due to dimension stuff



BUGS:







"""