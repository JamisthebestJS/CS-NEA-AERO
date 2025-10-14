import pygame
from enum import Enum, auto
import numpy as np
from helpers.introsort import introsort




SCREEN_SIZE = 700
#pythag shows cat_rom_def should be sqrt(2)*side_length which is what is done below
CAT_ROM_DEF =  int(1.41*(SCREEN_SIZE))#number of points between each pair of vertices in catmull rom spline (wants to be num of pixels diagonally)

SIM_HEIGHT = 300
MAX_VERTEX_COUNT = 30


class CatmullRom(staticmethod):
    points = []
    path = np.array([])
    
    #do CatmullRom.update_vertices() when vertex moved/deleted
    #do CatmullRom.new_vertex() when vertex added
    
    @staticmethod
    def catmull_rom(p0, p1, p2, p3, t):
        a0 = []
        a1 = []
        a2 = []
        for i in range(2):
            a0.append(-0.5 * p0[i] + 1.5 * p1[i] - 1.5 * p2[i] + 0.5 * p3[i])
            a1.append(p0[i] - 2.5 * p1[i] + 2 * p2[i] - 0.5 * p3[i])
            a2.append(-0.5 * p0[i] + 0.5 * p2[i])
        a3 = p1
        point = []
        for i in range(2):
            point.append(((a0[i] * t + a1[i]) * t + a2[i]) * t + a3[i])
        return point
    
    @staticmethod
    def catmull_rom_path(resolution):
        n = len(CatmullRom.points)
        for i in range(n-1):
            p0 = CatmullRom.points[i - 1] if i > 0 else CatmullRom.points[i]
            p1 = CatmullRom.points[i]
            p2 = CatmullRom.points[i + 1]
            p3 = CatmullRom.points[i + 2] if i + 2 < n else CatmullRom.points[i + 1]
            for j in range(resolution):
                t = j / resolution
                #if i < n - 1 or j == 0:
                CatmullRom.path.append(CatmullRom.catmull_rom(p0, p1, p2, p3, t))
            
            #trying to get first to join to last point
        for j in range(resolution):
            t = j / resolution
            CatmullRom.path.append(CatmullRom.catmull_rom(CatmullRom.points[-2], CatmullRom.points[-1], CatmullRom.points[0], CatmullRom.points[1], t))
    
    @staticmethod
    def render_path(screen):
        for point in CatmullRom.path:
            pygame.draw.circle(screen, (255, 255, 255), (int((point[0] + 1)), int((point[1] + 1))), 1)
    
    def get_path():
        return CatmullRom.path

    @staticmethod
    def del_point(vertex_pos):
        CatmullRom.points.remove(vertex_pos)
        CatmullRom.new_path(CAT_ROM_DEF)
    
    @staticmethod
    def move_points(old_pos, new_pos):
        #find old pos in points list and replace with new pos
        for i in range(len(CatmullRom.points)):
            if CatmullRom.points[i] == old_pos:
                CatmullRom.points[i] = new_pos

        CatmullRom.new_path(CAT_ROM_DEF)
    
    @staticmethod
    def new_point(vertex_pos):
        CatmullRom.points.append(vertex_pos)
        #clear path and recalculate
        CatmullRom.new_path(CAT_ROM_DEF)
    
    
    @staticmethod
    def new_path(definition):
        CatmullRom.path = []
        if len(CatmullRom.points) > 1:
            CatmullRom.catmull_rom_path(definition)

    @staticmethod
    def clear_path():
        CatmullRom.path = []
        CatmullRom.points = []

class Modes(staticmethod):
    current_mode = ""
        
    @staticmethod
    def clear_all():
        Vertex.vertices = []
        CatmullRom.clear_path()

    @staticmethod
    def place_vertex(x, y):
        if VertexListOperations.get_vertices_list_length() < MAX_VERTEX_COUNT:
            print(f"Placing vertex at ({x}, {y})")
            Vertex(x, y)
            CatmullRom.new_point((x, y))
        else:
            print(f"reached limit of {MAX_VERTEX_COUNT}")

    @staticmethod
    def delete_vertex(vertex_x, vertex_y):
        ver_id = VertexListOperations.get_vertex_ID(vertex_x, vertex_y)
        #prints whether vertex found or not (and so deleted or not)
        if ver_id != -1:
            vertex = VertexListOperations.get_vertex(ver_id)
            CatmullRom.del_point(vertex.get_pos())
            VertexListOperations.del_from_vertices_list(vertex_x, vertex_y)
        else:
            print(f"No vertex found at ({vertex_x}, {vertex_y}) to delete")
        #need to then update lines
    
    @staticmethod
    def move_vertex(vert_id, dx, dy):
        vertex = VertexListOperations.get_vertex(vert_id)
        start_pos = vertex.get_pos()
        new_pos = (start_pos[0] + dx, start_pos[1] + dy)
        vertex.update_pos(new_pos[0], new_pos[1])
        CatmullRom.move_points(start_pos, new_pos)
    
    
    @staticmethod
    def save_to_file():
        """
        Should maybe change this a little so it calls a Helper method which does the masking stuff. 
        And then maybe another method from elsewhere which renders the save menu etc. bc the other Modes methods are quite small and more or less just call other functions.
        
        """
        
        #why make rendered aerofoil smaller? its like I'm applyin the scalar multiplication to the wrong version of path, but I'm most assuredly not?
        
        save_path = CatmullRom.get_path()
        
        #discretisation
        for i in save_path:
            i[0] = int(i[0]//(SCREEN_SIZE//SIM_HEIGHT))
            i[1] = int(i[1]//(SCREEN_SIZE//SIM_HEIGHT))
        
        #removes duplicates
        new = []
        seen_keys = set()
        for sublist in save_path:
            # Convert the mutable list to an immutable tuple to make it hashable
            key = tuple(sublist)
            if key not in seen_keys:
                seen_keys.add(key)
                new.append(sublist)
        
        #here the save_path is a list of unique coordinates that fuly defines the boundary of the aerofoil
        save_path = new
        
        #make a list of sublists with all points with the same x pos, and a list of sublists with points w same y pos
        #these lists also need to be sorted ASC according to the pas-meme attribute
        hori_sublist = [[] for i in range(SIM_HEIGHT)] #creates a list of (what will be lists). 1 for each possible y pos
        vert_sublist = [[] for i in range(SIM_HEIGHT)] #creates a list of (what will be lists). 1 for each possible x pos
        for i in range(len(save_path)):
            vert_sublist[save_path[i][0]].append(save_path[i])
            hori_sublist[save_path[i][1]].append(save_path[i])
        
        # remove empty sublists
        vert_sublist = [sublist for sublist in vert_sublist if sublist]
        hori_sublist = [sublist for sublist in hori_sublist if sublist]
        
                
        #now we need to sort the sublists in terms of the changing element, ASC
        #note this doesnt work if shape is tooo small (like a couple pixels or smt idrk)
        #also bit of an issue as its trying to sort 2-el arrays. For quick and insertion its simple enough but really unsure about heap sort
        for i in vert_sublist:
            introsort(i, 1)
        for i in hori_sublist:
            introsort(i, 1)
        
        #create seperate True/False masks for hori and vert (between 2 most extreme of each sublist)
        #AND the masks (to ensure any weird wave-like aerofoil shapes are correctly masked)
        vert_mask = np.zeros((SIM_HEIGHT, SIM_HEIGHT))
        hori_mask = np.zeros((SIM_HEIGHT, SIM_HEIGHT))
        
        #iterate through each sublist.
        #do some algo for swapping to 1s in each mask
        #if odd number, ignore for now
        #if even number, for each one, swap, starting at false for 0, going to false for 2.
        #this could do with being explained in NEA with some pseudocode or something like that
        
        
        #for each column which contains some aerofoil
        for sublist in vert_sublist:
            #if odd number of aero nodes, ignore (temporarily)
            if len(sublist) % 2 != 0:
                continue
            else: 
                #even number of points in that column
                swaps = 0
                y_start = 0
                #for each node in the sublist
                for node in sublist:
                    #make True btwn the first and second, False btwn second and third, etc... in the vertical mask
                    y_end = node[1]
                    vert_mask[node[0], y_start:y_end] = swaps
                    y_start = y_end
                    swaps = (1 if (swaps == 0) else 0)
        
        #something going wrong here or in hori_sublist stuff
        for sublist in hori_sublist:
            print(f"sublist {sublist}")
            #if odd number of aero nodes, ignore (temporarily)
            if len(sublist) % 2 != 0:
                continue
            else: 
                #even number of points in that column
                swaps = 0
                x_start = 0
                #for each node in the sublist
                for node in sublist:
                    #make True btwn the first and second, False btwn second and third, etc... in the vertical mask
                    x_end = node[0]
                    hori_mask[x_start:x_end, node[1]] = swaps
                    x_start = x_end
                    swaps = (1 if (swaps == 0) else 1)

        

        
        #HORI_MASK IS ALL FALSE
        
        
        #AND the masks and save to file      
        object_mask = np.array
        object_mask = np.logical_and(hori_mask,vert_mask)
        for i in object_mask:
            for j in i:
                if j == 1:
                    print("true")
        Toolbox.aerofoil_save_to_file(object_mask)  
            
        
    @staticmethod
    def set_mode(new_mode):
        Modes.current_mode = new_mode

    @staticmethod
    def get_mode():
        return Modes.current_mode

class Menus(staticmethod):
    current_menu = ""
    
    def get_open_menu():
        return Menus.current_menu
    
    def open_menu(menu):
        Menus.current_menu = menu
    
    def render_aerofoil_menu():
        pass
        #renders the aerofoil selection/save/edit/etc. menu
    
    def render_sidebar():
        pass
        #renders sidebar buttons which can be clicked
    
    def menu_selection(x, y):
        pass
        #takes where the mouse has clicked and determines whether you have clicked a button and if so, does what it should do
    
    
MODE_DICT = {
    "new": Modes.place_vertex,
    "delete": Modes.delete_vertex,
    "move": Modes.move_vertex,
    "clear": Modes.clear_all,
    "save": Modes.save_to_file
    }


class VertexListOperations(object):
    vertices = []
    
    @staticmethod
    def render_all_vertices(screen):
        for vertex in VertexListOperations.vertices:
            vertex.render(screen)
    
    @staticmethod
    def get_vertex(index):
        for i in VertexListOperations.vertices:
            if i.ID == index:
                return i
    
    @staticmethod
    def get_vertices_list_length():
        return len(VertexListOperations.vertices)
    
    @staticmethod
    def del_from_vertices_list(x, y):
        id = VertexListOperations.get_vertex_ID(x, y)
        if id != -1:
            if len(VertexListOperations.vertices) < 1:
                print("no vertex to delete from VertexListOperations.vertices")
                return
            
            vertex = VertexListOperations.get_vertex(id)
            if vertex == None:
                return
            
            print(f"deleting vertex {id} at ({x}, {y}) from VertexListOperations.vertices")
            VertexListOperations.vertices.remove(vertex)
            del(vertex)
    
    @staticmethod
    def append_vertices_list(vertex):
        VertexListOperations.vertices.append(vertex)
        
    @staticmethod
    def get_vertex_ID(check_x, check_y):
        for i in range(VertexListOperations.get_vertices_list_length()):
            vertex = VertexListOperations.vertices[i]
            
            #if within radius of click (5 pixels) (using circle stuff)
            if (vertex.x - check_x) ** 2 + (vertex.y - check_y) ** 2 <= 5 ** 2:
                return vertex.ID
        
        return -1 #if not in vertex list

#do i need this class? idk rn, we'll see
class Toolbox(staticmethod):
    @staticmethod
    def snap_to_grid(coord, grid_size):
        snapped_x = round(coord[0])
        snapped_y = round(coord[1])
        return snapped_x, snapped_y
            
    @staticmethod
    def aerofoil_save_to_file(ob_mask):
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
            for line in ob_mask:
                for node in line:
                    file.write(str(node))
                file.write("\n")
        file.close()
    
#Vertex class
#may need to split into stuff about the list (static) and actual vertices (nonstatic)
class Vertex(object):
    id_counter = 0
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.ID = Vertex.id_counter
        Vertex.id_counter += 1
        VertexListOperations.append_vertices_list(self)
    
    def get_pos(self):
        return (self.x, self.y)
    
    def get_old_pos(self):
        return (self.old_x, self.old_y)
    
    def update_pos(self, new_x, new_y):
        self.x = new_x
        self.y = new_y
    
    def render(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), 2)

class Aerofoil(object):
    #for rendering save/delete file menu (&edit?? could save the vertices in a seperate file which allows it to be edited later?)
    #reason for class is the scrollable menu listing the aerofoils would work quite nicely as instances of the class.
    
    def __init__(self):
        pass
    
    def render(self, screen):
        pass
    
    def get_vertices(self):
        #gets aerofoils vertices from file
        #the rendered button in the menu is then highlighted, so you can save over the old aerofoil, or save as new
        #sets VertexListOperations.vertices to the vertices from file (and calls a CRS path update)
        pass

    def new_aerofoil(self):
        pass
        #adds button to menu when/after saving an aerofoil
    
#mouse dragging vars:
dragging = False
drag_start_pos = None
drag_end_pos = None
dragged_vertex_ID = None

def p_main(screen, event):
    global dragging, drag_start_pos, drag_end_pos, dragged_vertex_ID
    #so can quit while drawing
    running = True
    if event.type == pygame.QUIT:
        running = False
        pygame.quit()
    
    #CHANGING MODES
    if event.type == pygame.KEYDOWN:
            #for testing, set modes with keyboard keys
            
            old_mode = Modes.get_mode()
            if event.key == pygame.K_n:
                Modes.set_mode("new")
            elif event.key == pygame.K_d:
                Modes.set_mode("delete")
            elif event.key == pygame.K_m:
                Modes.set_mode("move")
            elif event.key == pygame.K_c:
                Modes.set_mode("clear")
            elif event.key == pygame.K_s:
                Modes.set_mode("save")
            
            #debug
            if Modes.get_mode != old_mode:
                print(f"Mode set to {Modes.get_mode()}")
    
    
    

    
    #IF MOUSE CLICK
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Mouse button 1 (left click) has been pressed
            mouse_x, mouse_y = Toolbox.snap_to_grid(pygame.mouse.get_pos(), SCREEN_SIZE)
            print(f"Left click at ({mouse_x}, {mouse_y})")
            
            mode = Modes.get_mode()
            if mode == "delete" or mode == "new":
                MODE_DICT[mode](mouse_x, mouse_y)
            
            elif mode == "clear" or mode == "save":
                #move is dealt with separately below
                MODE_DICT[mode]()
    
    
        # Track mouse drag: store position on press, compare on release
            if Modes.get_mode() == "move":
                dragged_vertex_ID = VertexListOperations.get_vertex_ID(mouse_x, mouse_y)
                if dragged_vertex_ID != -1:
                    drag_start_pos = event.pos
                    dragging = True
        
    #IF MOUSE RELEASE
    #dy, dx fed into Modes.move_vert
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        if dragging:
            drag_end_pos = pygame.mouse.get_pos()
            dx = drag_end_pos[0] - drag_start_pos[0]
            dy = drag_end_pos[1] - drag_start_pos[1]
            
            if dx+dy != 0 and dragged_vertex_ID != -1: #if vertex has been moved
                print(f"vertex {dragged_vertex_ID} dragged from {drag_start_pos} to {drag_end_pos}, delta: ({dx}, {dy})")
                Modes.move_vertex(dragged_vertex_ID, dx, dy)
                
            dragged_vertex_ID = 0
            dragging = False
                
    screen.fill((0, 0, 0, 0))
    CatmullRom.render_path(screen)
    VertexListOperations.render_all_vertices(screen)
    pygame.display.flip()

    return running


"""
TODO:
- some sort of file saving. The formatting of this is dependent on how I want to load it in the aero simulator


- GUI stuff. Buttons for modes, maybe a sidebar for info
(smaller things)
- dont allow to place a vertex on top of another vertex (or too close)


ERRORS (causes crash):
-


BUG:
- saving the aerofoil for some reason visually shrinks it to simualation size, which shouldnt happen
    I am really lost as to why. Does NOT seem like it

- fix aerofoil mask generator


DONE (since last git push):

added:
-


fixed:
-


removed:
- 





"""