import pygame
import numpy as np
from helpers.queue import Queue
from helpers.menus.aero_menu import s_menu, sidebar_btns
import math



SCREEN_SIZE = 700
#pythag shows cat_rom_def should be sqrt(2)*side_length which is what is done below
CAT_ROM_DEF =  int(1.41*(SCREEN_SIZE))#number of points between each pair of vertices in catmull rom spline (wants to be num of pixels diagonally)

SIM_HEIGHT = 300
MAX_VERTEX_COUNT = 30


class CatmullRom(staticmethod):
    points = []
    path = np.array([])

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
            if 0 <= x < 9*SCREEN_SIZE/10 and 0 <= y <= SCREEN_SIZE:
                print(f"Placing vertex at ({x}, {y})")
                Vertex(x, y)
                CatmullRom.new_point((x, y))
            else:
                print(f"vertex out of bounds at {(x, y)}")
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
    def save_to_file(screen, font):        
        #so saving doesnt affect rendered shape
        raw_path = CatmullRom.get_path()
        save_path = [ [float(p[0]), float(p[1])] if not isinstance(p, (list, tuple)) else [p[0], p[1]] for p in raw_path ]
        #discretisation and resizing to fit simulation size
        for i in save_path:
            i[0] = int(i[0]//(SCREEN_SIZE/SIM_HEIGHT))
            i[1] = int(i[1]//(SCREEN_SIZE/SIM_HEIGHT))
        
        #removes duplicates
        new = []
        seen_keys = set()
        for sublist in save_path:
            sub = tuple(sublist)
            key = sub
            if key not in seen_keys:
                seen_keys.add(key)
                new.append(sub)
        #here the save_path is a list of the minimal number of unique coordinates that still fully defines the boundary of the aerofoil
        save_path = new
        
        #make a true/false mask of outline
        
        object_mask = np.zeros((SIM_HEIGHT, SIM_HEIGHT))
        old_object_mask = np.zeros((SIM_HEIGHT, SIM_HEIGHT))
        for i in save_path:
            object_mask[i[0], i[1]] = True
            old_object_mask[i[0], i[1]] = True
        
        #need to somehow choose a random point in the aerofoil to begin BFS flood-fill
        #Take top and bottom nodes
        #get vector AB btwn them
        """
        get unit vector of AB
        make the smallest component =1 by multiplying unitAB by a coefficient
        round the other component
        add this repeatedly to A until a non-node is found, or you go past B
        If you go past B, repeat with left and right most nodes
        if this also fails, then no flood-fill algo
        """
        #first: finding the top and bottom, left and right -most extreme nodes.
        right = (1000, 0)
        left = (-1, 0)
        bottom = (0, -1)
        top = (0, 1000)
        for point in (save_path):
            if point[0] < right[0]:
                right = point
            elif point[0] > left[0]:
                left = point
            
            if point[1] < top[1]:
                top = point
            elif point[1] > bottom[1]:
                bottom = point
        
        #vector maths to get discretised direction vector
        TB_vec = tuple(x-y for x, y in zip(bottom, top))
        TB_vec_mag = math.sqrt(TB_vec[0]**2 + TB_vec[1]**2)
        
        n_TB_vec = (TB_vec[0] / TB_vec_mag, TB_vec[1] / TB_vec_mag)
        discrete_dir_vec = n_TB_vec
        
        small_comp = min(discrete_dir_vec[0], discrete_dir_vec[1])
        coef = 1/small_comp
        
        discrete_dir_vec = (int(discrete_dir_vec[0]*coef), int(discrete_dir_vec[1]*coef))
        
        discrete_dir_vec_mag = math.sqrt(discrete_dir_vec[0]**2 + discrete_dir_vec[1]**2)
        
        #repeatedly adds dir vec to bottom and checks if it lands on a non-boundary node (i.e. is inside)
        x, y = -1, -1
        for i in range(int(TB_vec_mag//discrete_dir_vec_mag)):
            check_point = tuple(x+y for x, y in zip(bottom, discrete_dir_vec))
            
            if object_mask[check_point[0]][check_point[1]] == False:
                x = check_point[0]
                y = check_point[1]
                break
        
        #do left to right check if cant find valid start point in the bottom to top check
        if x == -1:
            LR_vec = tuple(x-y for x, y in zip(right, left))
            LR_vec_mag = math.sqrt(LR_vec[0]**2 + LR_vec[1]**2)
            
            n_LR_vec = (LR_vec[0] / LR_vec_mag, LR_vec[1] / LR_vec_mag)
            discrete_dir_vec = n_LR_vec
            
            small_comp = min(LR_vec[0], LR_vec[1])
            coef = 1/small_comp
            
            discrete_dir_vec = (int(discrete_dir_vec[0]*coef), int(discrete_dir_vec[1]*coef))
            
            discrete_dir_vec_mag = math.sqrt(LR_vec[0]**2 + LR_vec[1]**2)

            for i in range(int(LR_vec_mag//discrete_dir_vec_mag)):
                check_point = tuple(x+y for x, y in zip(left, discrete_dir_vec))
                
                if object_mask[check_point[0]][check_point[1]] == False:
                    x = check_point[0]
                    y = check_point[1]
                    break
        
        #if valid start pos found
        if x != -1 and y != -1:
            print(f"starting flood-fill at {x, y}")
            Toolbox.flood_fill(x=x, y=y, image=object_mask, new_colour=1, queue=Queue())
        
        
        s_menu(object_mask, screen, font)
        s_menu(old_object_mask, screen, font)
        
        
        
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
    def snap_to_grid(coord):
        snapped_x = round(coord[0])
        snapped_y = round(coord[1])
        return snapped_x, snapped_y
    
    @staticmethod 
    def flood_fill(x, y, image, new_colour, queue):
        old_colour = image[x, y]
        print("olf colour", old_colour)
        
        if old_colour == new_colour:
            return image
        
        queue.enqueue((x, y))
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        image[x, y] = new_colour
        
        queue_count = 0
        while queue.is_empty() == False:
            x, y = queue.dequeue()
            
            for dx, dy in directions:
                queue_count+=1
                nx = x+dx
                ny = y+dy
                
                if 0 <= nx < len(image) and 0 <= ny < len(image[0])\
                    and image[nx, ny] == old_colour:
                    image[nx, ny] = new_colour
                    queue.enqueue((nx, ny))
        print("flood-fill finished after filling", queue_count, "nodes")
    
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


#mouse dragging vars:
dragging = False
drag_start_pos = None
drag_end_pos = None
dragged_vertex_ID = None

def p_main(screen, event, font):
    global dragging, drag_start_pos, drag_end_pos, dragged_vertex_ID
    #so can quit while drawing
    running = True
    if event.type == pygame.QUIT:
        running = False
        pygame.quit()
    
    #SIDEBAR WAY OF CHANGING MODES
    mode = Modes.get_mode()
    mode, screen = sidebar_btns(screen, mode, font)
    
    
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
            mouse_x, mouse_y = Toolbox.snap_to_grid(pygame.mouse.get_pos())
            print(f"Left click at ({mouse_x}, {mouse_y})")
            
            if mode == "delete" or mode == "new":
                MODE_DICT[mode](mouse_x, mouse_y)
            
            elif mode == "clear":
                MODE_DICT[mode]()
            
            elif mode == "save":
                MODE_DICT[mode](screen, font)
    
    
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

    return "draw"


"""
TODO:
- GUI stuff. Buttons for modes, maybe a sidebar for info
(smaller things)
- dont allow to place a vertex on top of another vertex (or too close)

ERRORS (causes crash):
-

BUG:
- saving the aerofoil for some reason visually shrinks it to simualation size, which shouldnt happen
    I am really lost as to why. Does NOT seem like it should do that at all




"""