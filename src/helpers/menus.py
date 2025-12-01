#LBM_main_menu
import pygame
import os
from helpers.sort import insertion_sort


BTN_COLOUR = (60,220,80)
# List items

# Scroll variables
SCROLL_SPEED = 20

global LBM_menus
LBM_menus = []

FRACTION = 1/9


#function to instantiate menu objects
def construct_menus(screen, big_font, med_font, small_font):
    
    #main LBM menu
    branches = ["aero_menu", "prop_or_jet", "sim_settings"]
    btn_texts = ["Aerofoil", "Thrust Engine", "Simulation Settings"]
    menu_type = "LBM_main"
    main_LBM_menu = MiscMenu(screen, big_font, med_font, menu_type, btn_texts, branches = branches, title = "Simulations", )

    #aerofoil list
    menu_type = "aero_menu"
    aero_list = ListMenu(screen, med_font, small_font, menu_type, vis_size = 10)
    
    #propulsion type menu
    branches = ["prop", "jet"]
    btn_texts = []
    menu_type = "prop_or_jet"
    prop_or_jet_menu = MiscMenu(screen, big_font, med_font, menu_type, btn_texts, branches = branches, title = "Thrust Simulations", )
    
    
    #sim settings menu
    setting_titles = []
    setting_defaults = []
    menu_type = "sim_settings"
    settings_menu = SettingsMenu(screen, big_font, small_font, menu_type, setting_titles, 
                                 setting_defaults, title = "Simulation Settings", )
    
    
    #main menu
    branches = ["LBM_main", "draw"]
    btn_texts = ["Simulations", "Aerofoil Drawer"]
    menu_type = "main_menu"
    the_main_menu = MiscMenu(screen, big_font, med_font, menu_type, btn_texts, branches = branches, title = "Wind Tunnel Simulator", )

    menus_dict = {
    "main_menu": the_main_menu,
    "LBM_main": main_LBM_menu,
    "aero_menu": aero_list,
    "prop_or_jet": prop_or_jet_menu,
    "sim_settings": settings_menu,
    
    }
    return menus_dict

#Superclass
class Menu(object):
    def __init__(self, screen, font, font2, type):
        self.screen = screen
        self.font = font
        self.font2 = font2
        self.type = type
        self.width, self.height = screen.get_size()
    
    def render(self, ):
        print("no render method programmed")
        pygame.display.flip()
        return self.screen

    def do_input(self, event, ):
        print("no do_input method programmed")
        return self.type
        
    def controller(self, event, ):
        self.screen = self.render()
        self.type = self.do_input(event, )
        return str(self.type), self.screen




#Subclasses
#need to think on this one a bit.
class SettingsMenu(Menu):
    def __init__(self, screen, font, font2, type, setting_titles, setting_defaults, title):
        super().__init__(screen, font, font2, type, )
        self.title = title

    #these are hardcoded as the layout isnt really generalisable 
    def render(self):
        #**************
        #temperature and altitude
        box = pygame.Rect(self.width*1/7, self.height*3/9, self.width*2/7, self.height/9)
        pygame.draw.rect(self.screen, BTN_COLOUR, box)
        temp_text =  self.font2.render()
        
        
        
        #density
        
        
        
        #inflow velocity
        
        
        #simulation width
        return self.screen
    

    def do_input(self, event, ):
        #overwriting
        pass




class ListMenu(Menu):
    def __init__(self, screen, font, font2, type, vis_size):
        super().__init__(screen, font, font2, type, )
        self.vis_size = vis_size
        self.item_height = self.height//vis_size
        self.scroll_y = 0
        self.directory = "src\helpers\\txt_files\Aerofoils"
    
    
    def render(self, items, ):
        self.screen.fill((255,255,255))
  
        # Draw items
        text_y_pad = 15
        text_x_pad = 10
        for i, item in enumerate(items):
            item_rect = pygame.Rect(50, i * self.item_height + self.scroll_y, self.screen.get_width() - 100, self.item_height)
            if 0 <= item_rect.y <= self.screen.get_height():  # Only draw visible items
                pygame.draw.rect(self.screen, (50,50,50) if i % 2 == 0 else (255,255,255), item_rect)
                text = self.font.render(item, True, (0,0,0))
                self.screen.blit(text, (item_rect.x + text_x_pad, item_rect.y + text_y_pad))
        # Update display
        pygame.display.flip()
        return self.screen
    
    
    def do_input(self, event, items, list_height):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                self.scroll_y = min(self.scroll_y + SCROLL_SPEED, 0)
            elif event.button == 5:
                self.scroll_y = max(self.scroll_y - SCROLL_SPEED, self.screen.get_height() - list_height)
        
            elif event.button == 1:
                item = None
                mouse_x, mouse_y = pygame.mouse.get_pos()
                index = (mouse_y - self.scroll_y)//self.item_height
                if 0 <= index < len(items):
                    print(f"Aerofoil {items[index]} selected")
                    item = items[index]
                    return item
    
    
    def controller(self, event, ):
        files = os.listdir(self.directory)
        # Filter only files
        aerofoils_list = [f for f in files if os.path.isfile(os.path.join(self.directory, f))]
        if aerofoils_list != None:
            items = insertion_sort(aerofoils_list)
            list_height = len(items) * self.item_height
            
            menu_type = self.do_input(event, items, list_height)
            self.screen = self.render(items, )
        if menu_type != None:
            return menu_type, self.screen
        else:
            return self.type, self.screen




class MiscMenu(Menu):
    def __init__(self, screen, font, font2, type, texts_list, title, branches):
        super().__init__(screen, font, font2, type)
        self.btn_texts = texts_list
        self.btn_count = len(self.btn_texts)
        self.btn_height = 1/10 * self.width
        self.hoz_pad = 1/4 * self.height
        self.branches = branches
        self.title = title
    
    def render(self, ):
        width = self.width
        height = self.height
        #need to do title still
        
        for i in range(self.btn_count):
            pygame.draw.rect(self.screen, BTN_COLOUR, (self.hoz_pad, (3+2*i)*FRACTION*height, (width-2*self.hoz_pad), self.btn_height))
            text = self.font2.render(self.btn_texts[i], True, (0,0,0))
            text_width, text_height = self.font2.size(self.btn_texts[i])
            self.screen.blit(text, (self.hoz_pad + (width-2*self.hoz_pad - text_width)/2, (3+2*i)*FRACTION*height + (self.btn_height - text_height)/2))
        pygame.display.flip()
        return self.screen

    def do_input(self, event, ):
        width = self.width
        height = self.height
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            for i in range(self.btn_count):
                if self.hoz_pad <= mouse_x <= (width-self.hoz_pad) and (3+2*i)*FRACTION*height <= mouse_y <= (3+2*i)*FRACTION*height + self.btn_height:
                    print(f"{self.btn_texts[i]} clicked")
                    return self.branches[i]
        
        return self.type


