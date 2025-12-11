# LBM attempt 2
r"""
some notey bits. Remove at some point
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

Note that this scheme can become unstable for Reynoldsnumbers >~ 350

"""

import pygame
from pathlib import Path
import jax.numpy as jnp
import jax
from helpers.graphs import update_graphs
from helpers.saving_settings import load_settings
from helpers.unit_conversions import Conversions

render_aerofoil = False


global frame_count
frame_count = 0

#constants
#changed from 100

N_POINTS_X = 600
N_POINTS_Y = 300

SCREEN_HEIGHT = N_POINTS_Y
SCREEN_WIDTH = N_POINTS_X 
AEROFOIL_OFFSET = SCREEN_WIDTH//8

#changes how vibrant different vels/dens/curls are rendered
MAX_RENDERED_VAL = 0.16
MAX_RENDERED_DENSITY = 4
MAX_RENDERED_VORT = 0.00833

MAX_HORIZONTAL_INFLOW_VEL = 0.05

DIAG_COEF = 0.70711
KINEMATIC_VISCOSITY = 1
RELAXATION_NUMBER = (1.0 / (KINEMATIC_VISCOSITY/(1/3)) + 1/2) #check stability of 1/2 over 2/3 (technically should be 1/2)



N_DISCRETE_VELOCITIES = 9
LATTICE_VELS = jnp.array([
    [0, 1, 0, -1, 0, 1, -1, -1, 1,],
    [0, 0, 1, 0, -1, 1, 1, -1, -1,]])

LATTICE_INDICES = jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
OPPOSITE_LATTICE_INDICES = jnp.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

LATTICE_WEIGHTS = jnp.array([
    4/9, #centre
    1/9, 1/9, 1/9, 1/9, #cardinals
    1/36, 1/36, 1/36, 1/36]) #diagonals
RIGHT_VELS = jnp.array([1, 5, 8])
LEFT_VELS = jnp.array([3, 6, 7])
TOP_VELS = jnp.array([2, 5, 6])
BOTTOM_VELS = jnp.array([4, 7, 8])
VERTICAL_VELS = jnp.array([0, 2, 4])
HORIZONTAL_VELS = jnp.array([0, 1, 3])

# Enable 64bit
jax.config.update("jax_enable_x64", True)


class Aerofoil(object):

    def __init__(self, aerofoil_name):
        self.aerofoil_name = aerofoil_name
    
    def load_aerofoil(self, ):
        mask = jnp.zeros((N_POINTS_X, N_POINTS_Y))
        aerofoil_file = Path(f"src\helpers\\txt_files\Aerofoils\{self.aerofoil_name}")
        print(f"using aerofoil file: {aerofoil_file}")
        if aerofoil_file.is_file() == False:
            print("No aerofoil to load. Please draw an aerofoil before attempting to use the wind tunnel simulator")
        else:
            with open(aerofoil_file, "r")as file:
                for j, line in enumerate(file):
                    line = line.rstrip('\n')
                    line = list(map(int, line))
                    mask = mask.at[AEROFOIL_OFFSET:AEROFOIL_OFFSET+len(line), j].set(line[:])       
        return mask
    
        
    def get_aerofoil_coords(self):
            if self.aerofoil_name == "jet":
            #***********************
            pass
        elif self.aerofoil_name == "prop":
            #***********************
            pass
        else:
            #loading the aerofoil shape from file
            aerofoil = jnp.array([])
            ob_mask = jnp.array([])
            ob_mask = self.load_aerofoil(self.aerofoil_name)
            #if no aerofoil with that name exists - done for performance reasons
            if  not jnp.array_equal(ob_mask, jnp.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))):
                aerofoil = jnp.array(jnp.where(ob_mask == 1))
            else:
                aerofoil = []
                print("aerofoil is None")
        return aerofoil


class Calculators(staticmethod):

    @staticmethod
    def get_density(discrete_vels):
        # ρ = ∑ᵢ fᵢ
        density = jnp.sum(discrete_vels, axis=-1)
        return density

    @staticmethod
    def get_macro_velocities(discrete_vels, density):
        # u = 1/ρ ∑ᵢ fᵢ cᵢ
        
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
            density[..., jnp.newaxis] * LATTICE_WEIGHTS[jnp.newaxis, jnp.newaxis, :] * (
                1 + 3 * proj_discrete_vels + 9/2 * proj_discrete_vels**2 - 3/2 * macro_vels_mag[..., jnp.newaxis]**2 
                ) 
            )
        return equilibrium_discrete_vels
    
    @jax.jit
    def new_force(delta_vel, delta_density):
        #should probably be done better since diving by density then multiplying
        momentum_density = jnp.einsum('NMQ,dQ->NMd', delta_vel, LATTICE_VELS)
        
        delta_x_momentum = jnp.sum(momentum_density[:, :, 0]*delta_density[:, :]*delta_x**3)
        delta_y_momentum = jnp.sum(momentum_density[:, :, 1]*delta_density[:, :]*delta_x**3)
        #force = delta_momentum / 1 (as across 1 timestep)        
        return delta_x_momentum, delta_y_momentum



def render(screen, vel_mag, density, vorticity, render_type):
    global aerofoil_coords, colours, frame_count, render_aerofoil
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

    #this is causing MASSIVE performance issues. Bigger aerofoil shapes take wayyy longer to render (like quarters frame rate for big ones)
    """
    if aerofoil_coords != [] and render_aerofoil == True:
        coords = jnp.array(aerofoil)
        xs = coords[:, 0]
        ys = coords[:, 1]
        colours = colours.at[xs, ys, :].set((255)).astype(jnp.uint8)
    """
    surface = pygame.surfarray.make_surface(colours)
    #surf = pygame.transform.scale(surf, (width, height))
    screen.blit(surface, (0,0))


    if frame_count % 100 == 0:
        print(f"frame {frame_count} rendered")
    
    pygame.display.flip()
    return render_type

@jax.jit
def update(discrete_vels_prev):
    global aerofoil_coords
       
    
    # 1. Apply outflow boundary condition on the right boundary
    discrete_vels_prev = discrete_vels_prev.at[-1, :, LEFT_VELS].set(discrete_vels_prev[-2, :, LEFT_VELS]) #(bounday stuff has same value as stuff one cell further left)

    # 2. Compute Macroscopic Quantities (density and velocities)
    density_prev = Calculators.get_density(discrete_vels_prev)
    macro_vels_prev = Calculators.get_macro_velocities(discrete_vels_prev, density_prev)
    
    # 3. Apply inflow stuff by Zou/He  (Dirichlet BC)
    macro_vels_prev = macro_vels_prev.at[0, 1:-1, :].set(velocity_profile[0, 1:-1, :])
    #maths according to Zou/He
    density_prev = density_prev.at[0, :].set((Calculators.get_density(discrete_vels_prev[0, :, VERTICAL_VELS].T) + 2 * Calculators.get_density(discrete_vels_prev[0, :, LEFT_VELS].T)) / (1 - macro_vels_prev[0, :, 0]))
    
    # 4. calc the discrete equilibria velocities
    equilibrium_discrete_vels = Calculators.get_equilibrium_velocities(macro_vels_prev, density_prev)
    #more Zou/He
    discrete_vels_prev = discrete_vels_prev.at[0, :, RIGHT_VELS].set(equilibrium_discrete_vels[0, :, RIGHT_VELS])
    
    # 5. BGK collisions
    # fᵢ ← fᵢ − ω (fᵢ − fᵢᵉ)
    discrete_vels_post_collisions = discrete_vels_prev - RELAXATION_NUMBER * (discrete_vels_prev - equilibrium_discrete_vels)
    discrete_vels_post_bounceback = discrete_vels_post_collisions
    
    # 6. bounce-back (for no-slip on interior boundary)
    pre_bounceback_density = Calculators.get_density(discrete_vels_post_collisions)
    for i in range(N_DISCRETE_VELOCITIES):
        discrete_vels_post_bounceback = discrete_vels_post_bounceback.at[aerofoil_coords[0, :], aerofoil_coords[1, :], LATTICE_INDICES[i]].set(discrete_vels_prev[aerofoil_coords[0, :], aerofoil_coords[1, :], OPPOSITE_LATTICE_INDICES[i]])
    post_bounceback_density = Calculators.get_density(discrete_vels_post_bounceback)
    #delta v due to momentum exchange
    delta_vel = discrete_vels_post_bounceback - discrete_vels_post_collisions
    #delta density due to momentum exchange
    delta_density = post_bounceback_density - pre_bounceback_density
    
    
    # 7. streaming along lattice vels
    discrete_vels_streamed = discrete_vels_post_bounceback
    for i in range(N_DISCRETE_VELOCITIES):
        discrete_vels_streamed = discrete_vels_streamed.at[:, :, i].set(
            jnp.roll(
            jnp.roll(
                discrete_vels_post_bounceback[:, :, i], LATTICE_VELS[0, i], axis = 0), LATTICE_VELS[1, i], axis = 1
                    )
                    )
    return discrete_vels_streamed, delta_density, delta_vel


def LBM_setup(aerofoil_name):
    """
    SETUP
    """
    #velocity profile
    global velocity_profile
    velocity_profile = jnp.zeros((N_POINTS_X, N_POINTS_Y, 2))
    velocity_profile = velocity_profile.at[0, :, 0].set(MAX_HORIZONTAL_INFLOW_VEL) #horizontal velocity profile

    global discrete_vels_prev
    discrete_vels_prev = Calculators.get_equilibrium_velocities(velocity_profile, jnp.ones((N_POINTS_X, N_POINTS_Y)))
    
    global aerofoil_coords
    aerofoil = Aerofoil(aerofoil_name)
    aerofoil_coords = Aerofoil.get_aerofoil_coords()

    #initialises an array which holds RGB val for every pixel in window
    global colours
    colours = jnp.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=jnp.uint8)
    
    #loading settings from file
    setting_tags = ["density", "temperature", "altitude", "sim_scale", "inflow_velocity"]
    global setting_values
    setting_values = load_settings(setting_tags)
    global converters
    converters = Conversions(
        SI_velocity = float(setting_values[setting_tags.index("inflow_velocity")]),
        SI_density = float(setting_values[setting_tags.index("density")]),
        average_sim_density = Calculators.get_density(jnp.full((N_POINTS_X, N_POINTS_Y, N_DISCRETE_VELOCITIES),MAX_HORIZONTAL_INFLOW_VEL)).mean()

   )
    global delta_x
    delta_x = converters.SI_to_sim_length(float(setting_values[setting_tags.index("sim_scale")])) / N_POINTS_X
    print("finished setup")
    return True



def LBM_main_loop(screen, iteration, render_type):
    if screen !=None:
        screen.fill("black")

    #moving to next time-step
    global discrete_vels_prev, frame_count
    discrete_vels_next, delta_density, delta_vel = update(discrete_vels_prev)
    

    #force plotting
    if iteration % 25 == 0 and iteration > 500:
        hori_force, vert_force = Calculators.new_force(delta_vel, delta_density)
        hori_force_SI = converters.sim_to_SI_force(hori_force)
        vert_force_SI = converters.sim_to_SI_force(vert_force)
        print(f"iteration {iteration} - horizontal force: {hori_force_SI}, vertical force: {vert_force_SI}")
        update_graphs(it_count = iteration, lift_item=vert_force_SI, drag_item=hori_force_SI)
    
    
    #drawing the simulation
    discrete_vels_prev = discrete_vels_next
    density = Calculators.get_density(discrete_vels_next)
    macro_vels = Calculators.get_macro_velocities(discrete_vels_next, density)
    vel_magnitude = jnp.linalg.norm(macro_vels, axis = -1, ord = 2)
    d_u__d_x, d_u__d_y = jnp.gradient(macro_vels[..., 0])
    d_v__d_x, d_v__d_y = jnp.gradient(macro_vels[..., 1])
    curl = d_u__d_y - d_v__d_x
    render(screen, vel_magnitude, density, curl, render_type)
    
    iteration += 1
    return iteration

"""
#TODO:
- 

ERRORS:
-

 
BUGS:



"""
