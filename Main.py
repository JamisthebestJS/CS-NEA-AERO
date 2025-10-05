
import pygame
from point_drawing_v2 import p_main
from LBM_v2 import LBM_main, LBM_main_loop


 
SCREEN_HEIGHT = 800
SCREEN_WIDTH = 600

SIM_WIDTH = 600
SIM_HEIGHT = 300

M_ITERATIONS = 10000





def main_menu():
    
    #renders main menu
    #returns what button is pressed if a button is pressed
    return input("Choose 'sim' or 'draw':")




def main():
    choice = main_menu()
    
    pygame.init()
    screen_width, screen_height = SCREEN_WIDTH, SCREEN_WIDTH
    global screen
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Virtual Wind Tunnel")
    clock = pygame.time.Clock()

    
    running = True
    while running:
        
        for event in pygame.event.get():
            
            if choice == None:
                #if the choice is exited by pressing esc, then you can re-chose
                choice = main_menu()
            
            if choice == "sim":
                LBM_main()
                screen = pygame.display.set_mode((SIM_WIDTH, SIM_HEIGHT))
                for i in range(M_ITERATIONS):
                    clock.tick(60)
                    LBM_main_loop(screen)
                    pygame.display.flip()
                    
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        choice == None
                        #hopefully exits only uppermost loop (the sim loop)
                        break
                
            
            elif choice == "draw":
                p_main(screen, event)


if __name__ == "__main__":
    main()
    
    
    
"""
#TODO
-





BUGS:
- for some reason, no window opens before choosing a thing
- window isnt properly working (cant click on it to make it come to front) when choosing sim
- exiting simulation to main menu doesnt work

- drawing doesnt work. Not noticing key inputs (any?)

ERRORS:
-






"""