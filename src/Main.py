
import pygame
from point_drawing_v2 import p_main
from LBM_v2 import LBM_setup, LBM_main_loop
from helpers.menus import construct_menus

 
#SCREEN_HEIGHT = 800
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

SIM_WIDTH = 600
SIM_HEIGHT = 300

M_ITERATIONS = 1000000


#fucntion to start the simulations and to loop for propulsion devices and etc..
#*******************************************

def sim(screen, render_type, iteration):
    iteration = LBM_main_loop(screen, iteration, render_type)
    return iteration



#resets variables to go back to the main menu
def go_to_main_menu():
    running = True
    iteration = 0
    menu_type = "main_menu"
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    print("back to main menu")
    
    
    return running, iteration, menu_type, screen


def main(screen):
    running = True
    setup = False
    iteration = 0
    menu_type = "main_menu"
    render_type = "vorticity" #options: "density", "velocity", "vorticity"


    #instantiating the menus and adding them to the dictionary
    menus_dict = construct_menus(screen, big_font, menu_font, small_font)
    doings = {
    "draw": p_main,
    "prop": sim,
    "jet": sim,
    "sim": sim,
    
    }
    DISPLAYS = menus_dict | doings
        
    
    while running:
        
        if menu_type == "sim" or menu_type == "prop" or menu_type == "jet":
            iteration = sim(screen, render_type, iteration)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                quit()
            

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running, iteration, menu_type, screen = go_to_main_menu()
                
                elif event.key == pygame.K_v:
                    render_type = "velocity"
                elif event.key == pygame.K_c:
                    render_type = "vorticity"
                elif event.key == pygame.K_d:
                    render_type = "density"
            
            #*************************
            if menu_type == "draw":
                p_main(screen, event, menu_font)

                
            elif menu_type in menus_dict:
                #else do stuff for when its a class (strictly menu)
                menu_type, screen = DISPLAYS[menu_type].controller(event, )
                #if an aerofoil has been returned by 
                if menu_type not in DISPLAYS:
                    aerofoil_name = menu_type
                    menu_type = 'sim'
                    screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
                    LBM_setup(aerofoil_name)

#pygame boilerplate
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Virtual Wind Tunnel")
clock = pygame.time.Clock()
small_font = pygame.font.SysFont("arial", 20)
menu_font = pygame.font.SysFont("arial", 30)
big_font = pygame.font.SysFont("arial", 40)
pygame.font.init()

#driver code
if __name__ == "__main__":
    main(screen)
    

"""
#TODO


BUGS:



ERRORS:
- in rendering (LBM) casting error from float32 to uint8
- mega crash caused by something to do with something to do with the plotting stuff (although never happened since the first time??)
^^ just got it again. no idea the cause. will run without touching and see if same thing.
^^ its when I try to move the matplotlib graph. no idea why
^^ probably like a library thing, so just going to accept that that happens.

DONE#:
ADDED:


IMPROVED:
- 

FIXED:
- clearing does not visually clear vertices (DRAW)
    ^^can move, but not delete. Removed from CM.points, but not from Vertex.vertices


SUCCINCT TODO:
- try to fix the periodic top/bottom boundary condition - problem is with using jnp.roll. Not sure how to fix, other than maybe
    ^^ maybe applying a no-stick condition after rolling (streaming)
- implement new flood-fill starting point finding algo
- implement changing sim parameters
- unit conversions
- thrust simulation - boundary conditions for interior boundary, as well as preset interior boundaries
- menu rejig - WIP


"""