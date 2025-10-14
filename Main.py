
import pygame
from point_drawing_v2 import p_main
from LBM_v2 import LBM_main, LBM_main_loop


 
SCREEN_HEIGHT = 800
SCREEN_WIDTH = 600

SIM_WIDTH = 600
SIM_HEIGHT = 300

M_ITERATIONS = 10000




#renders main menu
#returns what button is pressed if a button is pressed
def main_menu(event):
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



def main(screen):
    choice = None
    running = True
    iteration = setup = False
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
            
            
            
            if choice == None:
                #if the choice is exited by pressing esc, then you can re-chose
                choice = main_menu(event)
            
            elif choice == "sim":
                if setup == False:
                    setup = LBM_main()
                    
                screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
                if iteration < M_ITERATIONS:
                    iteration+=1
                    LBM_main_loop(screen)
                    
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        choice == None
                        #hopefully exits only uppermost loop (the sim loop)
                        break
            
            elif choice == "draw":
                running = p_main(screen, event)



pygame.init()
global screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_WIDTH))
pygame.display.set_caption("Virtual Wind Tunnel")
clock = pygame.time.Clock()


if __name__ == "__main__":
    main(screen)
    
    
"""
#TODO
-


BUGS:
- exiting simulation to main menu doesnt work



ERRORS:
-



DONE#:

IMPROVED:
- 


FIXED:
-




"""