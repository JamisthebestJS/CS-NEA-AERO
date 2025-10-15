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
from pathlib import Path
import jax.numpy as jnp
import jax

global aerofoil
aerofoil = []

#constants
REYNOLDS = 100

N_POINTS_X = 600
N_POINTS_Y = 300


SCREEN_HEIGHT = N_POINTS_Y
SCREEN_WIDTH = N_POINTS_X 

MAX_RENDERED_VAL = 0.04

"""
need to define inner boundary for aerofoil
being done in point_drawing, saved as a True/False field, and then loaded here at beginning of simulation

"""

MAX_HORIZONTAL_INFLOW_VEL = 500

PLOT_N_STEPS = 25
SKIP_FIRST_N_STEPS = 0

N_DISCRETE_VELOCITIES = 9

LATTICE_VELS = jnp.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1,],
    [0, 0, 1, 0, -1, 1, 1, -1, -1,]])

LATTICE_INDICES = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
OPPOSITE_LATTICE_INDICES = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

LATTICE_WEIGHTS = jnp.array([ #these seem arbitrary, but look at writeup for explanation (need to write it first though)
    4/9, #centre
    1/9, 1/9, 1/9, 1/9, #cardinals
    1/36, 1/36, 1/36, 1/36]) #diagonals

RIGHT_VELS = jnp.array([1, 5, 8])
LEFT_VELS = jnp.array([3, 6, 7])
TOP_VELS = jnp.array([2, 5, 6])
BOTTOM_VELS = jnp.array([4, 7, 8])
VERTICAL_VELS = jnp.array([0, 2, 4])
HORIZONTAL_VELS = jnp.array([0, 1, 3])

global density


class Helpers(staticmethod):
        
    @staticmethod
    def get_density(discrete_vels):
        # ρ = ∑ᵢ fᵢ
        density = jnp.sum(discrete_vels, axis=-1)
        return density

    @staticmethod
    def get_macro_velocities(discrete_vels, density):
        # u = 1/ρ ∑ᵢ fᵢ cᵢ
        
        #debugging
        #print("a", (np.einsum('NMQ,dQ->NMd', discrete_vels, LATTICE_VELS)).shape)
        #print("b", (density[..., np.newaxis]).shape)
        
        #∑ᵢ fᵢ cᵢ
        temp = jnp.einsum('NMQ,dQ->NMd', discrete_vels, LATTICE_VELS)
        # u = 1/ρ ∑ᵢ fᵢ cᵢ
        macro_vels = temp[:, :, :] / density[..., jnp.newaxis]
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

        #cᵢ
        proj_discrete_vels = jnp.einsum("dQ,NMd->NMQ", LATTICE_VELS, macro_vels)

        #||u||₂
        macro_vels_mag = jnp.linalg.norm(macro_vels, axis=-1, ord=2) #macro_vels over the last axis, and euclidian norm
        
        #maths according to the equation above
        equilibrium_discrete_vels = (
            density[..., jnp.newaxis] * 
            LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :] * (
                1 + 3 * proj_discrete_vels + 9/2 * proj_discrete_vels**2 - 3/2 * macro_vels_mag[..., jnp.newaxis]**2 
                ) 
            )
        

        return equilibrium_discrete_vels

    @staticmethod
    def load_aerofoil():
        mask = []        
        #temporarily always uses aerofoil 1
        aerofoil_file = Path("Aerofoils\\aerofoil 1.txt")
        if aerofoil_file.is_file() == False:
            print("No aerofoil to load. Please draw an aerofoil before attempting to use the wind tunnel simulator")
            return -1
        else:
            with open("Aerofoils\\aerofoil 1.txt", "r")as file:
                for line in file:
                    mask.append(line)
            return mask
        

def render(screen, vel_mag):
    global aerofoil
    
    width, height = vel_mag.shape
    
    colours = jnp.zeros((width, height, 3), dtype=jnp.uint8)
    values = jnp.clip(vel_mag, 0, MAX_RENDERED_VAL)
    col = (values/MAX_RENDERED_VAL*255).astype(jnp.uint8)
    colours = colours.at[..., 0].set(col)


    if isinstance(aerofoil, (list, tuple)):
        for x, y in aerofoil:
            if 0 <= x < width and 0 <= y < height:
                colours = colours.at[x, y, :].set(255)
    
    elif isinstance(aerofoil, jnp.ndarray):
        mask = aerofoil.astype(bool)
        colours[mask, :] = 255
    
    
    surf = pygame.surfarray.make_surface(colours)
    surf = pygame.transform.scale(surf, (width, height))
    screen.blit(surf, (0,0))
    
    pygame.display.flip()

@jax.jit
def update(discrete_vels_prev):
    global aerofoil
    
    # 1. Apply outflow boundary condition on the right boundary
    discrete_vels_prev = discrete_vels_prev.at[-1, :, LEFT_VELS].set(discrete_vels_prev[-2, :, LEFT_VELS]) #(bounday stuff has same value as stuff one cell further left)

    # 2. Compute Macroscopic Quantities (density and velocities)
    density_prev = Helpers.get_density(discrete_vels_prev)
    macro_vels_prev = Helpers.get_macro_velocities(discrete_vels_prev, density_prev)

    # 3. Apply inflow stuff by Zou/He  (Dirichlet BC)
    macro_vels_prev = macro_vels_prev.at[0, 1:-1, :].set(velocity_profile[0, 1:-1, :]) #at all points but very top and very bottom
    
    
    #maths according to Zou/He
    density_prev = density_prev.at[0, :].set((Helpers.get_density(discrete_vels_prev[0, :, VERTICAL_VELS].T) + 2 * Helpers.get_density(discrete_vels_prev[0, :, LEFT_VELS].T))\
        / (1 - macro_vels_prev[0, :, 0]))
    
    # 4. calc the discrete equilibria velocities
    equilibrium_discrete_vels = Helpers.get_equilibrium_velocities(macro_vels_prev, density_prev)
    
    #more Zou/He
    discrete_vels_prev = discrete_vels_prev.at[0, :, RIGHT_VELS].set((equilibrium_discrete_vels[0, :, RIGHT_VELS]))
    
    # 5. BGK collisions
    # fᵢ ← fᵢ − ω (fᵢ − fᵢᵉ)
    discrete_vels_post_collision = discrete_vels_prev - relaxation_factor * (discrete_vels_prev - equilibrium_discrete_vels)
    # 6. bounce-back (for no-slip on interior boundary)
    for i in range(N_DISCRETE_VELOCITIES):
        #basically, anything that is now inside the aerfoil is reversed so it "bounces back" from the boundary
        discrete_vels_post_collision = discrete_vels_post_collision.at[aerofoil, LATTICE_INDICES[i]].set(discrete_vels_prev[aerofoil, OPPOSITE_LATTICE_INDICES[i]])
        
    # 7. streaming along lattice vels
    discrete_vels_streamed = discrete_vels_post_collision
    for i in range(N_DISCRETE_VELOCITIES):
        #e.g. velocity going diagonally down and right goes to the cell which is diagonally down and right from where it was
        #etc. for all 9 (8 going outwards) velocities
        discrete_vels_streamed = discrete_vels_streamed.at[:, :, i].set(jnp.roll(
            jnp.roll(
                discrete_vels_post_collision[:, :, i], LATTICE_VELS[0, i], axis = 0),
            
            LATTICE_VELS[1, i], axis = 0
            ))
    
    return discrete_vels_streamed


def LBM_setup():    
    #SETUP
    kinematic_viscosity = (MAX_HORIZONTAL_INFLOW_VEL * 10) / REYNOLDS   #NOTE: 10 is radius, but an aerofoil doesnt have a radius, so not sure how that works
    
    global relaxation_factor
    relaxation_factor = 1 / (3 * kinematic_viscosity + 0.5)

    #def mesh
    x = jnp.arange(N_POINTS_X)
    y = jnp.arange(N_POINTS_Y)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    #velocity profile
    global velocity_profile
    velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    velocity_profile = velocity_profile.at[:, :, 0].set(MAX_HORIZONTAL_INFLOW_VEL) #horizontal velocity profile

    global discrete_vels_prev
    discrete_vels_prev = Helpers.get_equilibrium_velocities(velocity_profile, jnp.ones((N_POINTS_X, N_POINTS_Y)))
    
    global aerofoil
    ob_mask = Helpers.load_aerofoil() #I think this doesnt work quite right? requires a list of indices I think rather than True/False
    #need to get list of coords from this.
    for i in range(len(ob_mask)):
        for j in range(len(ob_mask[i])):
            if ob_mask[i][j] == str(1): #part of aerofoil
                aerofoil.append([i + SCREEN_WIDTH//8 , j])
    #i think ^^ not working: unrelated error showed that aerofoil in update loop was [] (i.e. empty which would be why no render)
    
    print("finished setup")
    return True



def LBM_main_loop(screen): 
    screen.fill("black")

    global discrete_vels_prev
    discrete_vels_next = update(discrete_vels_prev)
    
    discrete_vels_prev = discrete_vels_next
    density = Helpers.get_density(discrete_vels_next)
    macro_vels = Helpers.get_macro_velocities(discrete_vels_next, density)
    vel_magnitude = jnp.linalg.norm(macro_vels, axis = -1, ord = 2)
    render(screen, vel_magnitude)


"""
#TODO:


ERRORS:
-

 
BUGS:
- rendering flashes quite a bit
- aerofoil not loading (doesnt render or affect anything)
- really inconsistent framerate (probably whats causing flashing)
- velocity rendering is weird (and the velocity kinda just disappears)



"""