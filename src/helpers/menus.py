#LBM_main_menu
import pygame
import os
from helpers.sort import aerofoil_sort
from helpers.validation import validation_dict
from helpers.saving_settings import save_settings, load_settings
from helpers.variable_calc import VariableCalculators

BTN_COLOUR = (60,220,80)
ACTIVE_BTN_COLOUR = (220,80,60)
FRACTION = 1/9
SCROLL_SPEED = 20
global LBM_menus
LBM_menus = []

#function to instantiate menu objects
def construct_menus(screen, big_font, med_font, small_font):
    
    #main LBM menu
    branches = ["aero_menu", "prop_or_jet", "settings"]
    btn_texts = ["Aerofoil", "Thrust Engine", "Simulation Settings"]
    menu_type = "LBM_main"
    main_LBM_menu = MiscMenu(screen, big_font, med_font, menu_type, btn_texts, branches = branches, title = "Simulations", )

    #aerofoil list
    menu_type = "aero_menu"
    aero_list = ListMenu(screen, med_font, small_font, menu_type, vis_size = 10)
    
    #propulsion type menu
    branches = ["prop", "jet"]
    btn_texts = ["Propeller", "Jet Engine"]
    menu_type = "prop_or_jet"
    prop_or_jet_menu = MiscMenu(screen, big_font, med_font, menu_type, btn_texts, branches = branches, title = "Thrust Simulations", )
    
    #sim settings menu
    setting_titles = ["temperature(C): ", "altitude(m): ", "density(kg/m³): ", "sim width(m): ", "max velocity(m/s): "]
    setting_tags = ["temperature", "altitude", "density", "sim_width", "inflow_velocity"]
    setting_defaults = load_settings(setting_tags)
    if setting_defaults == [0,0,0,0,0]:
        setting_defaults = ["20", "12200", "1.2", "5", "30"]
    try:
        for num in setting_defaults:
            float(num)
    except:
        print("invalid stored values. reverting to original values")
        setting_defaults = ["20", "12200", "1.2", "5", "30"]

    menu_type = "settings"
    settings_menu = SettingsMenu(screen, big_font, small_font, menu_type, setting_titles, 
                                 setting_defaults, setting_tags, title = "Simulation Settings", )
    
    
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
    "settings": settings_menu,
    
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
        font = self.font
        title_text = pygame.font.render(f"{self.type}", True, (255, 255, 255))
        title_width, title_height = font.size(f"{self.type}")
        self.screen.blit(title_text, ((self.width - title_width)/2, (FRACTION*self.height - title_height/2)))

        pygame.display.flip()
        return self.screen

    def do_input(self, event, ):
        print("no do_input method programmed")
        return self.type
            
    def controller(self, event, ):
        self.screen = self.render()
        new_type = self.do_input(event, )
        return str(new_type), self.screen



#Subclasses
#need to think on this one a bit.
class SettingsMenu(Menu):
    def __init__(self, screen, font, font2, type, setting_titles, setting_defaults, setting_tags, title):
        super().__init__(screen, font, font2, type, )
        self.title = title
        self.setting_titles = setting_titles
        self.button_inputs = setting_defaults
        self.button_inputs = [str(item) for item in self.button_inputs]
        self.setting_tags = setting_tags
        self.error_type = None
        self.input_rects, self.active_inputs = self.create_textboxes()
        self.erroring_index = -1
        self.prev_erroring_index = -1

    def render(self):
        #renders the menu layout
        #ensures error message isnt cleared every frame
        if self.error_type == None or self.prev_erroring_index != self.erroring_index:
            self.screen.fill((0, 0, 0))
        
        if self.error_type != None:
            if self.error_type == "invalidInput":
                error_message = self.font2.render(f"Invalid input for {self.setting_tags[self.erroring_index]}", True, (255, 0, 0))
            else:
                error_message = self.font2.render(f"{self.error_type}", True, (255, 0, 0))
            self.screen.blit(error_message, (self.width/14, self.height*8/9))

        title_text = self.font.render(self.title, True, (255, 255, 255))
        title_width, title_height = self.font.size(self.title)
        self.screen.blit(title_text, ((self.width - title_width)/2, FRACTION*self.height - title_height/2))

        x_pad = 5
        y_pad = (self.input_rects[0].height - self.font2.size(self.setting_titles[0])[1])/2

        for i in range(len(self.setting_titles)):
            if self.active_inputs[i]:
                colour = ACTIVE_BTN_COLOUR
            else:
                colour = BTN_COLOUR
            pygame.draw.rect(self.screen, colour, (self.input_rects[i][:]))
            textbox_text = self.font2.render(self.setting_titles[i]+self.button_inputs[i], True, (0,0,0))
            self.screen.blit(textbox_text, (self.input_rects[i].x + x_pad, self.input_rects[i].y + y_pad))

        return self.screen

    def do_input(self, event, ):
        for i in range(len(self.setting_titles)):
            #for clicks
            if event.type == pygame.MOUSEBUTTONDOWN:
                #if click on textbox, toggle active, and deactivate
                if self.input_rects[i].collidepoint(event.pos):
                    print("clicked on", self.setting_titles[i])
                    if self.active_inputs[i] == False:
                        self.active_inputs = [False]*len(self.setting_titles)
                        self.active_inputs[i] = True
                    else:
                        self.active_inputs[i] = False

        #for text entry
        index = -1
        if event.type == pygame.KEYDOWN:
            try:
                index = self.active_inputs.index(True)
            except:
                print("No active input")
                self.error_type = "noActiveInputs"
            self.error_type = None
            
            if event.key == pygame.K_BACKSPACE:
                if len(self.button_inputs[index]) > 0:
                    self.button_inputs[index] = self.button_inputs[index][:-1]
                else:
                    self.button_inputs[index] = ""
            
            elif event.key == pygame.K_RETURN:
                #validating input then saving to file
                print("validating string", str(self.button_inputs[index]))
                if validation_dict[self.setting_tags[index]](str(self.button_inputs[index])) == True:
                    self.active_inputs[index] = False
                    save_settings(self.setting_tags, self.button_inputs)
                else:
                    print("invalid input for ", self.setting_tags[index])
                    self.error_type = "invalidInput"
                    self.prev_erroring_index = self.erroring_index
                    self.erroring_index = index
                    
                #if values of temp and altitude entered and not typing density, clear density
                #should only calcualte these values if changing temp density or altitude
                if self.error_type == None:
                    list_of_tags = ["temperature", "density", "altitude"]
                    if self.setting_tags[index] in list_of_tags:
                        if self.button_inputs[self.setting_tags.index("temperature")] != "" and self.button_inputs[self.setting_tags.index("altitude")] != "" \
                            and index != self.setting_tags.index("density"):
                            temperature = self.button_inputs[self.setting_tags.index("temperature")]
                            altitude = self.button_inputs[self.setting_tags.index("altitude")]
                            self.button_inputs[self.setting_tags.index("density")] = VariableCalculators.calculate_density(temperature=float(temperature), altitude=float(altitude))
                        #vice verse
                        if self.button_inputs[self.setting_tags.index("density")] != "" and index != self.setting_tags.index("temperature")\
                            and index != self.setting_tags.index("altitude"):
                            density = self.button_inputs[self.setting_tags.index("density")]
                            self.button_inputs[self.setting_tags.index("temperature")] = VariableCalculators.calculate_temperature(float(density))
                            if float(density) > 0.75:
                                altitude = "0.1"
                            else:
                                altitude = "12200"
                            self.button_inputs[self.setting_tags.index("altitude")] = altitude
            
            else:
                self.button_inputs[index] += str(event.unicode)
            
        return self.screen
    
    def create_textboxes(self, ):
        input_rects = []
        for i in range(len(self.setting_titles)):
            input_rects.append(pygame.Rect(self.width/14, (3+2*i)*self.height/9, self.width*5/14, self.height/9))
        active_inputs = [False]*len(self.setting_titles)
        #shifts density textbox to the right
        input_rects[2].x = self.screen.get_width()*8/14
        input_rects[2].y = self.screen.get_height()*3/9
        #shifts sim scale textbox up
        input_rects[3].y = self.screen.get_height()*7/9
        #shifts inflow velocity textbox back into screen and onto the right to fit
        input_rects[4].x = self.screen.get_width()*8/14
        input_rects[4].y = self.screen.get_height()*7/9
        return input_rects, active_inputs

    def controller(self, event, ):
        self.screen = self.render()
        self.do_input(event, )
        pygame.display.flip()
        return str(self.type), self.screen


class ListMenu(Menu):
    def __init__(self, screen, font, font2, type, vis_size):
        super().__init__(screen, font, font2, type, )
        self.vis_size = vis_size
        self.item_height = self.height//vis_size
        self.scroll_y = 0
        self.directory = r"src\helpers\txt_files\Aerofoils"
    
    
    def render(self, items, ):
        self.screen.fill((255,255,255))
        #draw items
        text_y_pad = 15
        text_x_pad = 10
        title_height = self.screen.get_height()/9
        for i, item in enumerate(items):
            item_rect = pygame.Rect(50, i * self.item_height + self.scroll_y + title_height, self.screen.get_width() - 100, self.item_height)
            if 0 <= item_rect.y <= self.screen.get_height():  # Only draw visible items
                pygame.draw.rect(self.screen, (50,50,50) if i % 2 == 0 else (255,255,255), item_rect)
                text = self.font2.render(item, True, (0,0,0) if i%2 != 0 else (255, 255, 255))
                self.screen.blit(text, (item_rect.x + text_x_pad, item_rect.y + text_y_pad))
        
        title_text = self.font.render("Aerofoils", True, (0,0,0))    
        self.screen.blit(title_text, (self.screen.get_width()/2 - self.font.size("Aerofoils")[0]/2, text_y_pad))    
        
        # Update display
        pygame.display.flip()
        return self.screen
    
    
    def do_input(self, event, items, list_height):
        #if scrolling
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                self.scroll_y = min(self.scroll_y + SCROLL_SPEED, 0)
            elif event.button == 5:
                self.scroll_y = max(self.scroll_y - SCROLL_SPEED, self.screen.get_height() - list_height)
            #if left click, find clicked aerofoil
            elif event.button == 1:
                item = None
                mouse_x, mouse_y = pygame.mouse.get_pos()
                index = int((mouse_y - self.scroll_y - self.screen.get_height()/9)//self.item_height) #put "- self.screen.get_height()/9" after "self.scroll_y"
                if 0 <= index < len(items) and 50 <= mouse_x <= self.screen.get_width() - 100:
                    print(f"Aerofoil {items[index]} selected")
                    item = items[index]
                    return item
    
    
    def controller(self, event, ):
        files = os.listdir(self.directory)
        # Filter only files and then sort numerically
        aerofoils_list = [f for f in files if os.path.isfile(os.path.join(self.directory, f))]
        if aerofoils_list != None:
            items = aerofoil_sort(aerofoils_list)
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
        font = self.font
        title_text = font.render(f"{self.title}", True, (255, 255, 255))
        title_width, title_height = font.size(f"{self.title}")
        self.screen.blit(title_text, ((width - title_width)/2, FRACTION*height - title_height/2))

        #renders buttons with button text
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
        #checks if a button has been clicked
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            
            for i in range(self.btn_count):
                if self.hoz_pad <= mouse_x <= (width-self.hoz_pad) and (3+2*i)*FRACTION*height <= mouse_y <= (3+2*i)*FRACTION*height + self.btn_height:
                    print(f"{self.btn_texts[i]} clicked")
                    return self.branches[i]
        
        return self.type