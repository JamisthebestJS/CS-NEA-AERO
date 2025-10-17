
import pygame
from point_drawing_v2 import p_main
from LBM_v2 import LBM_setup, LBM_main_loop
from helpers.menus.LBM_main_menu import Menus

 
SCREEN_HEIGHT = 800
SCREEN_WIDTH = 600

SIM_WIDTH = 600
SIM_HEIGHT = 300

M_ITERATIONS = 1000000




#renders main menu
#returns what button is pressed if a button is pressed
def main_menu(event):
    #should get rendering working really
    if event.type == pygame.QUIT:
        pygame.quit()
    
    if event.type == pygame.MOUSEBUTTONDOWN:
        print(f"mouse press registered at {pygame.mouse.get_pos()}")
        #currently: if click on left side of screen, starts sim, if click on right side of screen, starts draw
        if pygame.mouse.get_pos()[0] > SCREEN_WIDTH//2:
            return "draw"
        else:
            return "sim"
        
    return None

MENUS_DICT = {
"main": Menus.main_LBM_menu,
"sim_settings": Menus.LBM_sim_settings_menu,
"aero_menu": Menus.LBM_choose_aero_menu,
}


def main(screen):
    choice = None
    running = True
    set_up = False
    iteration = 0
    menu_type = "main"
    
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
            

            
            
            if choice == None:
                #if the choice is exited by pressing esc, then you can re-chose (doesnt work, so yeah, not ideal)
                choice = main_menu(event)

            elif choice == "draw":
                running = p_main(screen, event)
    
            elif choice == "sim":
                if menu_type == "aero_menu":
                    font = list_font
                else:
                    font = menu_font
                if menu_type != None and menu_type in MENUS_DICT:
                    menu_type = MENUS_DICT[menu_type](screen, event, font)
                
                elif menu_type not in MENUS_DICT:
                    aerofoil = menu_type
                    #start sim
                    if not set_up:
                        screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
                        LBM_setup(screen, aerofoil)
                        set_up = True
            
        if choice == "sim" and set_up:
            iteration = LBM_main_loop(screen, iteration)
            if iteration >= M_ITERATIONS:
                choice = None
                menu_type = "main"
                set_up = False
                iteration = 0
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                print("Simulation complete, returning to main menu")


pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))
pygame.display.set_caption("Virtual Wind Tunnel")
clock = pygame.time.Clock()
menu_font = pygame.font.SysFont("arial", 30)
list_font = pygame.font.SysFont("arial", 20)
pygame.font.init()

if __name__ == "__main__":
    main(screen)
    
    
"""
#TODO
-


BUGS:
- exiting simulation to main menu doesnt work
- velocity is always max or 0. No real variation
- some weird stuff happening


ERRORS:
- in rendering (LBM) casting error from float32 to uint8



DONE#:

ADDED:
- 



IMPROVED:
- 


FIXED:
- velocity acting strangely. Still not perfect. (LBM)
- half fixed aerofoil saving. Strangely cuts off the bottom half however (LBM/PD)
- only updates when mouse movement (LBM)
- aerofoil does not seem to affect flow (LBM)



"""