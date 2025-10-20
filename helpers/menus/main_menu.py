#main menu stuff

import pygame

BTN_COLOUR = (60,220,80)

def the_main_menu(screen, event, small_font, big_font):
    width, height = screen.get_size()
    
    
    pygame.draw.rect(screen, BTN_COLOUR, (width/4, 3*height/8, width/2, height/10))
        
    sim_text = small_font.render("Simulations", True, (0,0,0))
    sim_text_width, sim_text_height = small_font.size("Simulations")
    screen.blit(sim_text, (width/4 + (width/2 - sim_text_width)/2, 3*height/8 + (height/10 - sim_text_height)/2))
    
    pygame.draw.rect(screen, BTN_COLOUR, (width/4, 5*height/8, width/2, height/10))
    drawing_text = small_font.render("Aerofoil Drawing", True, (0,0,0))
    drawing_text_width, drawing_text_height = small_font.size("Aerofoil Drawing")
    screen.blit(drawing_text, (width/4 + (width/2 - drawing_text_width)/2, 5*height/8 + (height/10 - drawing_text_height)/2))
    
    
    if event.type == pygame.MOUSEBUTTONDOWN:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        if width/4 <= mouse_x <= 3*width and 3*height/8 <= mouse_y <= 3*height/8 + height/10:
            print("Simulation Menu Opened")
            return "LBM_main"
            
            
        elif width/4 <= mouse_x <= 3*width/4 and 5*height/8 <= mouse_y <= 5*height/8 + height/10:
            print("Drawing Aerofoil")
            return "draw"
    
    
    
    title_text = big_font.render("Wind Tunnel Simulator", True, (255, 255, 255))
    title_text_width, title_text_height = big_font.size("Windtunnel Simulator")
    screen.blit(title_text, (width/4 + (width/2 - title_text_width)/2, height/8 + (height/10 - title_text_height)/2))
    
    
    pygame.display.flip()
    return "main_menu"
    
    
    

