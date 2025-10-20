#menu stuff for saving aerofoils

import pygame


BTN_COLOUR = (60,220,80)
SB_BTN_COLOUR = (220, 60, 80)

def s_menu(aerofoil, screen, font):
    width, height = screen.get_size()
    
    choosing = True
    while choosing:
        screen.fill((0,0,0))
        
        pygame.draw.rect(screen, BTN_COLOUR, (width/4, height/8, width/2, height/10))
        confirm_text = font.render("Confirm Save", True, (0,0,0))
        confirm_text_width, confirm_text_height = font.size("Confirm Save")
        screen.blit(confirm_text, (width/4 + (width/2 - confirm_text_width)/2, height/8 + (height/10 - confirm_text_height)/2))

        pygame.draw.rect(screen, BTN_COLOUR, (width/4, 3*height/8, width/2, height/10))
        cancel_text = font.render("Cancel Save", True, (0,0,0))
        cancel_text_width, cancel_text_height = font.size("Cancel Save")
        screen.blit(cancel_text, (width/4 + (width/2 - cancel_text_width)/2, 3*height/8 + (height/10 - cancel_text_height)/2))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    choosing = False
        

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if width/4 <= mouse_x <= 3*width and height/8 <= mouse_y <= height/8 + height/10:
                    save(aerofoil)
                    choosing = False
                
                elif width/4 <= mouse_x <= 3*width and 3*height/8 <= mouse_y <= 3*height/8 + height/10:
                    choosing = False
                    
                    
        pygame.display.flip()
    

def save(aerofoil):
    #open stats file (to see how many aerofoils exist)
    stats_file = open("stats.txt", "r")
    stats_content = []
    all_content = []
    
    for line in stats_file:
        #remove non-number characters, then append the remaining number to stats_content
        result = ''.join([char for char in line if char.isdigit()])
        stats_content.append(result)
        all_content.append(line)
    aerofoil_count = stats_content[0]
    stats_file.close()
    
    new_aerofoil_count = str(int(aerofoil_count) + 1) #keeps a record of how many aerofoils so has a default name for the aerofoils as they are saved.
    #writes new aerofoil count into the file
    all_content[0] = "aerofoil_count = " + str(new_aerofoil_count)
    with open("stats.txt", "w") as file:
        for line in all_content:
            file.write(line)
            
    name = None
    #creating the file which stores the object mask
    if name == None:
        name = f"aerofoil {new_aerofoil_count}"
    
    with open(f"Aerofoils\{name}.txt", "w") as file:
        for line in aerofoil:
            for node in line:
                if node == True:
                    file.write(str(1))
                elif node == False: #can change to else if works
                    file.write(str(0))
            file.write("\n")
    file.close()




def sidebar_btns(screen, mode, font):
    
    width, height = screen.get_size()
    
    pygame.draw.rect(screen, (SB_BTN_COLOUR), (9*width/10, 0, width/10, height/10))
    new_text = font.render("N", True, (0,0,0))
    new_text_width, new_text_height = font.size("N")
    screen.blit(new_text, (9*width/10 + (width/10 - new_text_width)/2, 0 + (height/10 - new_text_height)/2))
        
    
    return mode, screen