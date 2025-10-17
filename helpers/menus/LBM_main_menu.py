#LBM_main_menu
import pygame
import os

BTN_COLOUR = (60,220,80)

# List items
ITEM_HEIGHT = 50

# Scroll variables
scroll_y = 0
SCROLL_SPEED = 20



class Menus():
        
    @staticmethod
    def main_LBM_menu(screen, event, font, ):
        width, height = screen.get_size()
        #need to render a main menu which has options for setting up the LBM simulation
        #should include options for setting viscosity, density, initial velocity, aerofoil shape
        #as well as render type
        #and a way to actually begin the simulation
        
        
        pygame.draw.rect(screen, BTN_COLOUR, (width/4, height/8, width/2, height/10))
        
        start_sim_text = font.render("Start Simulation", True, (0,0,0))
        start_sim_text_width, start_sim_text_height = font.size("Start Simulation")
        screen.blit(start_sim_text, (width/4 + (width/2 - start_sim_text_width)/2, height/8 + (height/10 - start_sim_text_height)/2))
        
        pygame.draw.rect(screen, BTN_COLOUR, (width/4, 3*height/8, width/2, height/10))
        sim_settings_text = font.render("Simulation Settings", True, (0,0,0))
        sim_settings_text_width, sim_settings_text_height = font.size("Simulation Settings")
        screen.blit(sim_settings_text, (width/4 + (width/2 - sim_settings_text_width)/2, 3*height/8 + (height/10 - sim_settings_text_height)/2))
        
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if width/4 <= mouse_x <= 3*width and height/8 <= mouse_y <= height/8 + height/10:
                print("Start Simulation button pressed")
                return "aero_menu"
                
                
            elif width/4 <= mouse_x <= 3*width/4 and 3*height/8 <= mouse_y <= 3*height/8 + height/10:
                print("Simulation Settings button pressed")
                return "sim_settings"
                
        
        
        pygame.display.flip()
        return "main"
        
    @staticmethod
    def LBM_sim_settings_menu(screen, event, font):
        width, height = screen.get_size()
        #need to render a menu which has options for setting up the LBM simulation
        #should include options for setting viscosity, density, initial velocity, aerofoil shape
        #as well as render type
        #and a way to actually begin the simulation
        
        screen.fill((0,0,0))
        
        pygame.draw.rect(screen, BTN_COLOUR, (width/4, height/8, width/2, height/10))
        
        set_viscosity_text = font.render("Set Viscosity", True, (0,0,0))
        set_viscosity_text_width, set_viscosity_text_height = font.size("Set Viscosity")
        screen.blit(set_viscosity_text, (width/4 + (width/2 - set_viscosity_text_width)/2, height/8 + (height/10 - set_viscosity_text_height)/2))
        
        pygame.draw.rect(screen, BTN_COLOUR, (width/4, 3*height/8, width/2, height/10))
        set_density_text = font.render("Set Density", True, (0,0,0))
        set_density_text_width, set_density_text_height = font.size("Set Density")
        screen.blit(set_density_text, (width/4 + (width/2 - set_density_text_width)/2, 3*height/8 + (height/10 - set_density_text_height)/2))
        
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if width/4 <= mouse_x <= 3*width and height/8 <= mouse_y <= height/8 + height/10:
                print("Set Viscosity button pressed")
                
                
            elif width/4 <= mouse_x <= 3*width and height/8 <= mouse_y <= height/8 + height/10:
                print("Set Density button pressed")

        return "sim_settings"


    @staticmethod
    def LBM_choose_aero_menu(screen, event, font):
        #need to get list of aerofoils in the Aerofoils/ directory
        
        directory = "Aerofoils"
        files = os.listdir(directory)

        # Filter only files
        aerofoils_list = [f for f in files if os.path.isfile(os.path.join(directory, f))]
        
        scroll_y = 0
        
        list_height = len(aerofoils_list) * ITEM_HEIGHT
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                scroll_y = min(scroll_y + SCROLL_SPEED, 0)
            elif event.button == 5:
                scroll_y = max(scroll_y - SCROLL_SPEED, -(list_height - screen.get_height()))
        
        
        #need to be able to return which one clicked on
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                index = (mouse_y - scroll_y)//ITEM_HEIGHT
                if 0 <= index < len(aerofoils_list):
                    print(f"Aerofoil {aerofoils_list[index]} selected")
                    item = aerofoils_list[index]
                    return item
                
        HelperFuncs.render_scrollable_list(screen, aerofoils_list, font)
        return "aero_menu"


class HelperFuncs():
        
    @staticmethod
    def render_scrollable_list(screen, items, font):
        screen.fill((255,255,255))

        # Draw items
        for i, item in enumerate(items):
            item_rect = pygame.Rect(50, i * ITEM_HEIGHT + scroll_y, screen.get_width() - 100, ITEM_HEIGHT)
            if 0 <= item_rect.y <= screen.get_height():  # Only draw visible items
                pygame.draw.rect(screen, (50,50,50) if i % 2 == 0 else (255,255,255), item_rect)
                text = font.render(item, True, (0,0,0))
                screen.blit(text, (item_rect.x + 10, item_rect.y + 10))

        # Update display
        pygame.display.flip()



"""
should probably make a dict in Main.py which holds the possible return values from these menus, and then calls them from Main.py


"""