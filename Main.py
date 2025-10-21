
import pygame
from point_drawing_v2 import p_main
from LBM_v2 import LBM_setup, LBM_main_loop
from helpers.menus.LBM_menu import Menus
from helpers.menus.main_menu import the_main_menu

 
#SCREEN_HEIGHT = 800
SCREEN_WIDTH = 600

SIM_WIDTH = 600
SIM_HEIGHT = 300

M_ITERATIONS = 1000000






MENUS_DICT = {
"LBM_main": Menus.main_LBM_menu,
"sim_settings": Menus.LBM_sim_settings_menu,
"aero_menu": Menus.LBM_choose_aero_menu,
"main_menu": the_main_menu,
"draw": p_main,

}


def go_to_main_menu():
    running = True
    set_up = False
    iteration = 0
    menu_type = "main_menu"
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))
    
    return running, set_up, iteration, menu_type, screen




def main(screen):
    running = True
    set_up = False
    iteration = 0
    menu_type = "main_menu"
    render_type = "vorticity" #options: "density", "velocity", "vorticity"
    
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                quit()
            

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running, set_up, iteration, menu_type, screen = go_to_main_menu()
                
                elif event.key == pygame.K_v:
                    render_type = "velocity"
                elif event.key == pygame.K_c:
                    render_type = "vorticity"
                elif event.key == pygame.K_d:
                    render_type = "density"

            if menu_type not in MENUS_DICT and menu_type != None:
                aerofoil = menu_type
                #start sim
                if not set_up:
                    screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
                    LBM_setup(screen, aerofoil)
                    set_up = True            
            
            if menu_type == "aero_menu" or menu_type == "draw":
                font = list_font
            else:
                font = menu_font
            
            if menu_type in MENUS_DICT:
                screen.fill((0,0,0))
                
                if menu_type == "main_menu":
                    menu_type = MENUS_DICT[menu_type](screen, event, font, big_font)
                else:
                    menu_type = MENUS_DICT[menu_type](screen, event, font)

            

                
                
                
            
        if menu_type not in MENUS_DICT and set_up:
            iteration = LBM_main_loop(screen, iteration, render_type)
            
            if iteration >= M_ITERATIONS:
                running, set_up, iteration, menu_type, screen = go_to_main_menu()
                print("Simulation complete, returning to main menu")


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))
pygame.display.set_caption("Virtual Wind Tunnel")
clock = pygame.time.Clock()
menu_font = pygame.font.SysFont("arial", 30)
list_font = pygame.font.SysFont("arial", 20)
big_font = pygame.font.SysFont("arial", 40)
pygame.font.init()

if __name__ == "__main__":
    main(screen)
    
    
"""
#TODO
- 

BUGS:
- a couple issues with menus. Specifically the point drawing sidebar (still WIP)


ERRORS:
- in rendering (LBM) casting error from float32 to uint8
- if any path is off the screen, error (DRAW)
- mega crash caused by something to do with something to do with the plotting stuff (although never happened since the first time??)


DONE#:
ADDED:
- 

IMPROVED:
- 

FIXED:


"""