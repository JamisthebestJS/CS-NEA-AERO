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
import numpy as np
from helpers.graphs import update_graphs

global aerofoil, render_aerofoil
aerofoil = []
render_aerofoil = False


global frame_count
frame_count = 0

#constants
#changed from 100



REYNOLDS = 100

N_POINTS_X = 600
N_POINTS_Y = 300


SCREEN_HEIGHT = N_POINTS_Y
SCREEN_WIDTH = N_POINTS_X 

AEROFOIL_OFFSET = SCREEN_WIDTH//8

MAX_RENDERED_VAL = 0.16
MAX_RENDERED_DENSITY = 4
MAX_RENDERED_VORT = 0.00833

"""
need to define inner boundary for aerofoil
being done in point_drawing, saved as a True/False field, and then loaded here at beginning of simulation

"""

MAX_HORIZONTAL_INFLOW_VEL = 0.04

DIAG_COEF = 0.70711


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

#these are causing some instability potentially, will have to check properly if the case
KINEMATIC_VISCOSITY = 1
reynolds_number = (MAX_HORIZONTAL_INFLOW_VEL) / KINEMATIC_VISCOSITY
speed_of_sound = 1/jnp.sqrt(3)
mach_number = MAX_HORIZONTAL_INFLOW_VEL / (1/3) #(= MAX_HORIZONTAL_INFLOW_VEL / speed_of_sound_L**2)
RELAXATION_OMEGA = (1.0 / (KINEMATIC_VISCOSITY/(1/3)) + 2/3)

MOMENTUM_EXCHANGE_MASK_IN = jnp.zeros((N_POINTS_X, N_POINTS_Y, 9)) > 0
momentum_exchange_mask_in_per_iter = jnp.zeros((N_POINTS_X, N_POINTS_Y, 9)) > 0
MOMENTUM_EXCHANGE_MASK_OUT = jnp.zeros((N_POINTS_X, N_POINTS_Y, 9)) > 0
momentum_exchange_mask_out_per_iter = jnp.zeros((N_POINTS_X, N_POINTS_Y, 9)) > 0


# Enable 64bit
jax.config.update("jax_enable_x64", True)



class Helpers(staticmethod):

    #not working?
    @staticmethod
    def load_aerofoil(aerofoil_name):
        mask = jnp.zeros((N_POINTS_X, N_POINTS_Y))
        #temporarily always uses aerofoil 1
        aerofoil_file = Path(f"Aerofoils\\{aerofoil_name}")
        print(f"using aerofoil file: {aerofoil_file}")
        if aerofoil_file.is_file() == False:
            print("No aerofoil to load. Please draw an aerofoil before attempting to use the wind tunnel simulator")
        else:
            with open(aerofoil_file, "r")as file:
                for i, line in enumerate(file):
                    line = line.rstrip('\n')
                    line = list(map(int, line))
                    mask = mask.at[i + AEROFOIL_OFFSET].set(line)       
        return mask

    @staticmethod
    def init_colours(screen):
        global colours
        colours = jnp.zeros((screen.get_width(), screen.get_height(), 3), dtype=jnp.uint8)




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
    
    
    @jax.jit
    def get_force(discrete_velocities):
        force = jnp.sum(
            (LATTICE_VELS.T[jnp.newaxis, jnp.newaxis, ...] * discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_IN] + 
            (LATTICE_VELS.T[OPPOSITE_LATTICE_INDICES][jnp.newaxis, jnp.newaxis, ...] * discrete_velocities[..., jnp.newaxis])[MOMENTUM_EXCHANGE_MASK_OUT]
        )

        return force

    @staticmethod
    def get_cαcβ(alpha, beta):
        c_alpha = LATTICE_VELS[alpha, :]
        c_beta  = LATTICE_VELS[beta,  :]
        return c_alpha * c_beta
    
    @staticmethod
    def get_non_equilibrium_velocities(discrete_vels, macro_vels, density):
        equilibrium_velocities = Helpers.get_equilibrium_velocities(macro_vels, density)
        return discrete_vels - equilibrium_velocities

    
    @staticmethod
    def get_stress_tensor(discrete_vels, macro_vels, density):
        non_equilibrium_vels = Helpers.get_non_equilibrium_velocities(discrete_vels, macro_vels, density)

        non_equilibrium_stress_tensor = ((1 - RELAXATION_OMEGA / 2) * jnp.sum(cαcβ * non_equilibrium_vels, axis = -1))
        return non_equilibrium_stress_tensor

    @staticmethod
    def get_strain_rate_tensor_LB(discrete_vels, macro_vels, density):
        stress_tensor = Helpers.get_stress_tensor(discrete_vels, macro_vels, density)
        strain_rate_tensor = (stress_tensor / (2 * density * KINEMATIC_VISCOSITY))
        return strain_rate_tensor
    
    @staticmethod
    def get_strain_rate_tensor_FD(macro_vels):
        disalligned_gradients = jnp.array([jnp.gradient(macro_vels[..., i]) for i in range(2)])

        gradients = jnp.einsum('...ij -> ...ji', disalligned_gradients)
        return - (gradients + jnp.einsum('...ij -> ...ij', gradients))/2
        
    

def render(screen, vel_mag, density, vorticity, render_type):
    global aerofoil, colours, frame_count, render_aerofoil
    frame_count+=1
    if render_type == "density":
        colours = colours.at[..., 0].set((density[:]/MAX_RENDERED_DENSITY)*255).astype(jnp.uint8)
        colours = colours.at[..., 1].set(0)
        
    elif render_type == "velocity":
        colours = colours.at[..., 0].set((vel_mag[:]/MAX_RENDERED_VAL)*255).astype(jnp.uint8)
        colours = colours.at[..., 1].set(0)
        
    elif render_type == "vorticity":
        colours = colours.at[..., 0].set((vorticity[:]/MAX_RENDERED_VORT)*255).astype(jnp.uint8)
        colours = colours.at[..., 1].set((-vorticity[:]/MAX_RENDERED_VORT)*255).astype(jnp.uint8)
        
    
    #values = jnp.clip(vel_mag, 0, MAX_RENDERED_VAL)

    
    #this is causing MASSIVE performance issues. Bigger aerofoil shapes take wayyy longer to render (like quarters frame rate for big ones)
    """
    if aerofoil != [] and render_aerofoil == True:
        coords = jnp.array(aerofoil)
        xs = coords[:, 0]
        ys = coords[:, 1]
        colours = colours.at[xs, ys, :].set((255)).astype(jnp.uint8)
    """
    surf = pygame.surfarray.make_surface(colours)
    #surf = pygame.transform.scale(surf, (width, height))
    screen.blit(surf, (0,0))


    if frame_count % 100 == 0:
        print(f"frame {frame_count} rendered")
    
    pygame.display.flip()
    return render_type

@jax.jit
def update(discrete_vels_prev):
    global aerofoil
       
    
    # 1. Apply outflow boundary condition on the right boundary
    discrete_vels_prev = discrete_vels_prev.at[-1, :, LEFT_VELS].set(discrete_vels_prev[-2, :, LEFT_VELS]) #(bounday stuff has same value as stuff one cell further left)

    # 2. Compute Macroscopic Quantities (density and velocities)
    density_prev = Helpers.get_density(discrete_vels_prev)
    macro_vels_prev = Helpers.get_macro_velocities(discrete_vels_prev, density_prev)
    # 3. Apply inflow stuff by Zou/He  (Dirichlet BC)
    macro_vels_prev = macro_vels_prev.at[0, 1:-1, :].set(velocity_profile[0, 1:-1, :])
    #maths according to Zou/He
    density_prev = density_prev.at[0, :].set((Helpers.get_density(discrete_vels_prev[0, :, VERTICAL_VELS].T) + 2 * Helpers.get_density(discrete_vels_prev[0, :, LEFT_VELS].T)) / (1 - macro_vels_prev[0, :, 0]))
    
    # 4. calc the discrete equilibria velocities
    equilibrium_discrete_vels = Helpers.get_equilibrium_velocities(macro_vels_prev, density_prev)
    
    #more Zou/He
    discrete_vels_prev = discrete_vels_prev.at[0, :, RIGHT_VELS].set(equilibrium_discrete_vels[0, :, RIGHT_VELS])
    
    # 5. BGK collisions
    # fᵢ ← fᵢ − ω (fᵢ − fᵢᵉ)
    discrete_vels_post_collisions = discrete_vels_prev - RELAXATION_OMEGA * (discrete_vels_prev - equilibrium_discrete_vels)
    print("done 5")
    
    # 6. bounce-back (for no-slip on interior boundary)
    print("done coords")
    xs = aerofoil[:, 0]
    print("done  xs")
    ys = aerofoil[:, 1]
    print("done ys")
    for i in range(N_DISCRETE_VELOCITIES):
        discrete_vels_post_collisions = discrete_vels_post_collisions.at[xs, ys, LATTICE_INDICES[i]].set(discrete_vels_prev[xs, ys, OPPOSITE_LATTICE_INDICES[i]])
    print("done 6")
    #slip on top and bottom walls (i.e. any vertical velocity is converted to horizontal.)
    #need some paper for this.
    # 8->1, 4->0, 7->3 => bottom wall
    # 2->0, 6->3, 5->1 => top wall
    #^^doesnt work :(
        #^^ causes MAJOR instabilities, basically instantly does the 'cold blu-tac stretching' thing
    
    # 7. streaming along lattice vels
    discrete_vels_streamed = discrete_vels_post_collisions
    for i in range(N_DISCRETE_VELOCITIES):
        #e.g. velocity going diagonally down and right goes to the cell which is diagonally down and right from where it was
        #etc. for all 9 (8 going outwards) velocities
        discrete_vels_streamed = discrete_vels_streamed.at[:, :, i].set(
            jnp.roll(
            jnp.roll(
                discrete_vels_post_collisions[:, :, i], LATTICE_VELS[0, i], axis = 0), LATTICE_VELS[1, i], axis = 1
                    )
                    )
    #discrete_vels_streamed = discrete_vels_streamed.at[:, 0, BOTTOM_VELS].set((discrete_vels_prev[:, 0, ])*DIAG_COEF)
    #discrete_vels_streamed = discrete_vels_streamed.at[:, -1, TOP_VELS].set(0)
    return discrete_vels_streamed


def LBM_setup(screen, aerofoil_name):
    """
    #SETUP
    kinematic_viscosity = (MAX_HORIZONTAL_INFLOW_VEL) / REYNOLDS   #NOTE: 10 is radius, but an aerofoil doesnt have a radius, so not sure how that works
    
    global relaxation_factor
    #changed from + 0.5
    relaxation_factor = 1 / (3 * kinematic_viscosity + 2/3+1/16) #looks like strange number to add. Its been tested as good for stability
    """
    #def mesh
    x = jnp.arange(N_POINTS_X)
    y = jnp.arange(N_POINTS_Y)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    #velocity profile
    global velocity_profile
    velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    velocity_profile = velocity_profile.at[0, :, 0].set(MAX_HORIZONTAL_INFLOW_VEL) #horizontal velocity profile

    global discrete_vels_prev
    discrete_vels_prev = Helpers.get_equilibrium_velocities(velocity_profile, jnp.ones((N_POINTS_X, N_POINTS_Y)))
    
    global aerofoil, ob_mask
    ob_mask = Helpers.load_aerofoil(aerofoil_name)
    #need to get list of coords from this.
    if  not jnp.array_equal(ob_mask, jnp.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))):
        indexes = jnp.where(ob_mask == 1)
        aerofoil = jnp.array((indexes[0], indexes[1]))
        #i think ^^ not working: unrelated error showed that aerofoil in update loop was [] (i.e. empty which would be why no render)
    else:
        aerofoil = []
        print("aerofoil is None")

    
    #so dont have to initialise colours every frame
    Helpers.init_colours(screen)
    
    global MOMENTUM_EXCHANGE_MASK_IN, MOMENTUM_EXCHANGE_MASK_OUT
    for i, (x,y) in enumerate(LATTICE_VELS.T):
        location_in = jnp.logical_and(jnp.roll(
                                            jnp.roll(
                                                jnp.logical_not(ob_mask),
                                            x, axis=0), 
                                        y, axis=1), ob_mask)
        MOMENTUM_EXCHANGE_MASK_IN = MOMENTUM_EXCHANGE_MASK_IN.at[location_in, i].set(True)
        
        
        location_out = jnp.logical_and(jnp.roll(
                                            jnp.roll(
                                                ob_mask,
                                            -x, axis=0),
                                        -y, axis=1), jnp.logical_not(ob_mask))
        
        MOMENTUM_EXCHANGE_MASK_OUT = MOMENTUM_EXCHANGE_MASK_OUT.at[location_out, OPPOSITE_LATTICE_INDICES[i]].set(True)
    
    global cαcβ
    alpha, beta = jnp.meshgrid(jnp.arange(N_POINTS_Y), jnp.arange(N_POINTS_X))
    cαcβ = Helpers.get_cαcβ(alpha, beta)
    

    
    
    
    
    print("finished setup")
    return True



def LBM_main_loop(screen, iteration, render_type):
    if screen !=None:
        screen.fill("black")

    global discrete_vels_prev, frame_count
    discrete_vels_next = update(discrete_vels_prev)
    discrete_vels_prev = discrete_vels_next
    density = Helpers.get_density(discrete_vels_next)
    macro_vels = Helpers.get_macro_velocities(discrete_vels_next, density)
    vel_magnitude = jnp.linalg.norm(macro_vels, axis = -1, ord = 2)
    
    #tests
    horizontal_force = Helpers.get_force(discrete_vels_next)
    shear_rate_tensor = Helpers.get_strain_rate_tensor_FD(macro_vels)
    strain_rate_FD = shear_rate_tensor[..., 0]

    shear_stress = Helpers.get_strain_rate_tensor_LB(discrete_vels_next, macro_vels, density)
    strain_rate_LB = shear_stress[..., 0]
    #need to calc total strain rates FD and LB
    #then find difference
    lift = 0

    if iteration % 25 == 0 and iteration > 500:
        update_graphs(it_count = iteration, lift_item = lift, drag_item=horizontal_force)
    
    
    
    d_u__d_x, d_u__d_y = jnp.gradient(macro_vels[..., 0])
    d_v__d_x, d_v__d_y = jnp.gradient(macro_vels[..., 1])
    curl = d_u__d_y - d_v__d_x
    render(screen, vel_magnitude, density, curl, render_type)
    iteration += 1
    return iteration

"""
#TODO:
- change boundary conditions to make top and bottom be bounce back (no slip)
- smagorinsky turbulence model? Might be too much for the scope

ERRORS:
-

 
BUGS:



"""