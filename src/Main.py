import pygame
from point_drawing_v2 import p_main
from LBM_v2 import LBM_setup, LBM_main_loop
from helpers.menus import construct_menus

 
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
SIM_WIDTH = 600
SIM_HEIGHT = 300
M_ITERATIONS = 1000000


#resets variables to go back to the main menu
def go_to_main_menu():
    running = True
    iteration = 0
    menu_type = "main_menu"
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    print("back to main menu")
    return running, iteration, menu_type, screen

#main loop
def main(screen):
    running = True
    iteration = 0
    menu_type = "main_menu"
    render_type = "vorticity" #options: "density", "velocity", "vorticity"

    #instantiating the menus and constructing dictionary
    menus_dict = construct_menus(screen, big_font, menu_font, small_font)
    sims_dict = {
    "prop": LBM_main_loop, #the simulations are dealth with 
    "jet": LBM_main_loop,
    "sim": LBM_main_loop,
    }
    draw_dict = {
        "draw": p_main
    }
    DISPLAYS = menus_dict | sims_dict | draw_dict
        
    
    while running:
        #if fluid sim is running
        if menu_type in sims_dict:
            iteration = DISPLAYS[menu_type](screen, iteration, render_type)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                quit()
            
            #return to main menu
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running, iteration, menu_type, screen = go_to_main_menu()
                
                #allows for different rendering modes in fluid sim
                elif event.key == pygame.K_v:
                    render_type = "velocity"
                elif event.key == pygame.K_c:
                    render_type = "vorticity"
                elif event.key == pygame.K_d:
                    render_type = "density"
            
            if menu_type == "draw":
                DISPLAYS[menu_type](screen, event, menu_font)

                
            elif menu_type in menus_dict:
                menu_type, screen = DISPLAYS[menu_type].controller(event, )
                #if aerofoil name is returned
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