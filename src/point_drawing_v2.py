import pygame
import jax.numpy as jnp
from helpers.queue import Queue
from helpers.aerofoil_save_funcs import save_menu



SCREEN_SIZE = 600
#pythag shows CAT_ROM_DEFINITION should be sqrt(2)*side_length which is what is done below
CAT_ROM_DEFINITION =  int(1.41*(SCREEN_SIZE))#number of points between each pair of vertices in catmull rom spline (wants to be num of pixels diagonally)

SIM_HEIGHT = 300
MAX_VERTEX_COUNT = 30


class CatmullRom(object):

    def __init__(self, control_points, path):
        self.__control_points = control_points
        self.__path = path

    #calculates q(t) from t for both x and y directions and returns (q_x(t), q_y(t))
    def catmull_rom_point_calc(self, p0, p1, p2, p3, t):
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
    
    #creates a list of points (each one a pixel) between P0 and Pn which creates the spline
    def catmull_rom_path_calc(self, ):
        n = len(self.__control_points)
        for i in range(n-1):
            p0 = self.__control_points[i - 1] if i > 0 else self.__control_points[i]
            p1 = self.__control_points[i]
            p2 = self.__control_points[i + 1]
            p3 = self.__control_points[i + 2] if i + 2 < n else self.__control_points[i + 1]
            for j in range(CAT_ROM_DEFINITION):
                t = j / CAT_ROM_DEFINITION
                self.__path.append(self.catmull_rom_point_calc(p0, p1, p2, p3, t))
            
        #trying to get first to join to last point
        for j in range(CAT_ROM_DEFINITION):
            t = j / CAT_ROM_DEFINITION
            self.__path.append(self.catmull_rom_point_calc(self.__control_points[-2], self.__control_points[-1], self.__control_points[0], self.__control_points[1], t))
    
    def render_path(self, screen):
        for point in self.__path:
            pygame.draw.circle(screen, (255, 255, 255), (int((point[0] + 1)), int((point[1] + 1))), 1)
    
    def get_path(self):
        return self.__path

    def del_point(self, vertex_position):
        self.__control_points.remove(vertex_position)
        self.new_path()
    
    def move_points(self, old_position, new_position):
        #find old pos in points list and replace with new pos
        for i in range(len(self.__control_points)):
            if self.__control_points[i] == old_position:
                self.__control_points[i] = new_position
        self.new_path()
    
    def new_point(self, vertex_position):
        self.__control_points.append(vertex_position)
        #clear path and recalculate
        self.new_path()
    
    def new_path(self):
        self.__path = []
        if len(self.__control_points) > 1:
            self.catmull_rom_path_calc()

    def clear_path(self):
        self.__path = []
        self.__control_points = []

spline = CatmullRom(control_points = [], path = [])


class Modes(staticmethod):
    current_mode = ""
        
    @staticmethod
    def clear_all(screen):
        screen.fill((0,0,0))
        VertexListOperations.vertices = []
        spline.clear_path()
        return screen

    @staticmethod
    def place_vertex(position):
        if VertexListOperations.get_vertices_list_length() < MAX_VERTEX_COUNT:
            if 0 <= position[0] < 9*SCREEN_SIZE/10 and 0 <= position[1] <= SCREEN_SIZE:
                print(f"Placing vertex at ({position[0]}, {position[1]})")
                Vertex(position)
                spline.new_point(position)
            else:
                print(f"vertex out of bounds at {position}")
        else:
            print(f"reached limit of {MAX_VERTEX_COUNT}")

    @staticmethod
    def delete_vertex(vertex_position):
        vertex_id = VertexListOperations.get_vertex_ID_at_position(vertex_position)
        #prints whether vertex found or not (and so deleted or not)
        if vertex_id != -1:
            vertex = VertexListOperations.get_vertex(vertex_id)
            spline.del_point(vertex.get_position())
            VertexListOperations.del_from_vertices_list(vertex_position)
        else:
            print(f"No vertex found at ({vertex_position[0]}, {vertex_position[1]}) to delete")
        #need to then update lines
    
    @staticmethod
    def move_vertex(vertex_id, move_vector):
        vertex = VertexListOperations.get_vertex(vertex_id)
        start_position = vertex.get_position()
        new_position = (start_position[0] + move_vector[0], start_position[1] + move_vector[1])
        vertex.update_position(new_position)
        spline.move_points(start_position, new_position)
    
    
    @staticmethod
    def save_to_file(screen, font):        
        #so saving doesnt affect rendered shape
        raw_path = spline.get_path()
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
        
        object_mask = jnp.zeros((SIM_HEIGHT, SIM_HEIGHT))
        for i in save_path:
            object_mask.at[i[1], i[0]].set(True) #aerofoil needs to be rotated because of the way saving works, each file row is each aerofoil column, but reading, it is 'correct
        
        x, y = Toolbox.find_flood_start(save_path)
        print("flood start", x, y)
        aerofoil_mask = Toolbox.flood_fill(x, y, object_mask, 1, Queue())
        
        #small error check - if has filled all 4 corners, likely picked wrong spot for flood fill start - NOT el-wise ob_mask
        print("corner vals:", aerofoil_mask[0,0], aerofoil_mask[0,-1], aerofoil_mask[-1,0], aerofoil_mask[-1,-1])
        if aerofoil_mask[0,0] == 1 and aerofoil_mask[0,-1] == 1 and aerofoil_mask[-1, 0] == 1 and aerofoil_mask[-1,-1] == 1:
            aerofoil_mask = jnp.logical_not(aerofoil_mask)
        
        save_menu(aerofoil_mask, screen, font)
        
    @staticmethod
    def set_mode(new_mode):
        Modes.current_mode = new_mode

    @staticmethod
    def get_mode():
        return Modes.current_mode



class Toolbox(staticmethod):
    @staticmethod
    def snap_to_grid(coord):
        snapped_x = round(coord[0])
        snapped_y = round(coord[1])
        return (snapped_x, snapped_y)
    
    @staticmethod 
    def flood_fill(x, y, array, new_colour, queue):
        array = array.tolist()

        old_colour = array[x][y]
        if old_colour == new_colour:
            return array
        
        queue.enqueue((x, y))
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        array[x][y] = new_colour
        
        queue_count = 0
        print(f"im width ={len(array)}, im height={len(array[0])}")
        
        while queue.is_empty() == False:
            x, y = queue.dequeue()
            
            for dx, dy in directions:
                queue_count+=1
                new_x = x+dx
                new_y = y+dy
                
                if 0 <= new_x < SIM_HEIGHT and 0 <= new_y < SIM_HEIGHT:
                    if array[new_x][new_y] == old_colour:
                        array[new_x][new_y] = new_colour
                        queue.enqueue((new_x, new_y))
        
        print("flood-fill finished after filling", queue_count, "nodes")
        
        return jnp.array(array)
    
    @staticmethod
    def find_flood_start(save_path):
        
    #1	Find the bottom-most point in the aerofoil perimeter
    #2	Find the 2 adjacent point to this point
    #3	Find the vector which goes from one to the other adjacent point
    #4	Find the perpendicular bisector of this vector
    #5	Clamp this vector to the 8 cardinal directions to limit vector length
    #6	Add the clamped vector to the bottom-most node to get the start-point

        #1.
        lowest = (1000,1000)
        for point in save_path:
            if point[1] < lowest[1]:
                lowest = point
        
        #2.
        lowest_index = save_path.index(lowest)
        left_adj_point = save_path[lowest_index-1]
        right_adj_point = save_path[lowest_index+1]
        
        #3.
        vector = (left_adj_point[0]-right_adj_point[0], left_adj_point[1]-right_adj_point[1])
        
        #4.
        normal_vector = (-vector[1], vector[0])
        
        #5.
        card_dir_vec = (round(Toolbox.clamp(normal_vector[0], -1, 1)), round(Toolbox.clamp(normal_vector[1], 0, 1)))
        print("card dir vec", card_dir_vec)
        print("loweest", lowest)
        
        x = lowest[0] + card_dir_vec[0]
        y = lowest[1] + card_dir_vec[1]
        
        print("x, y", x, y)
        
        return x,y
    
    @staticmethod
    def clamp(value, min, max):
        return max if value > max else min if value < min else value

    
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
            if i.get_id() == index:
                return i
    
    @staticmethod
    def get_vertices_list_length():
        print(VertexListOperations.vertices)
        return len(VertexListOperations.vertices)
    
    @staticmethod
    def del_from_vertices_list(pos):######
        id = VertexListOperations.get_vertex_ID_at_position(pos)
        if id != -1:
            if len(VertexListOperations.vertices) < 1:
                print("no vertex to delete from VertexListOperations.vertices")
                return
            
            vertex = VertexListOperations.get_vertex(id)
            if vertex == None:
                return
            
            print(f"deleting vertex {id} at ({pos[0]}, {pos[1]}) from VertexListOperations.vertices")
            VertexListOperations.vertices.remove(vertex)
            del(vertex)
    
    @staticmethod
    def append_vertices_list(vertex):
        VertexListOperations.vertices.append(vertex)
        
    @staticmethod
    def get_vertex_ID_at_position(pos):
        for i in range(VertexListOperations.get_vertices_list_length()):
            vertex = VertexListOperations.vertices[i]
            
            #if within radius of click (5 pixels) (using circle stuff)
            if (vertex.x - pos[0]) ** 2 + (vertex.y - pos[1]) ** 2 <= 5 ** 2:
                return vertex.get_id()
        
        return -1 #if not in vertex list
    

#Vertex class
#may need to split into stuff about the list (static) and actual vertices (nonstatic)
class Vertex(object):
    id_counter = 0
    def __init__(self, position):
        self.position = position
        self.x = position[0]
        self.y = position[1]
        self._ID = Vertex.id_counter
        Vertex.id_counter += 1
        VertexListOperations.append_vertices_list(self)
    
    def get_position(self):
        return (self.position[0], self.position[1])
    
    def get_id(self):
        return self._ID
    
    def update_position(self, new_position):
        self.position = new_position
        self.x = new_position[0]
        self.y = new_position[1]
    
    def render(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), self.position, 2)


#mouse dragging vars:
dragging = False
drag_start_position = None
drag_end_position = None
dragged_vertex_ID = None

def p_main(screen, event, font):
    global dragging, drag_start_position, drag_end_position, dragged_vertex_ID
    #so can quit while drawing
    if event.type == pygame.QUIT:
        pygame.quit()
    
    mode = Modes.get_mode()
    
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
            if mode != old_mode:
                print(f"Mode set to {Modes.get_mode()}")

    #IF MOUSE CLICK
    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Mouse button 1 (left click) has been pressed
            mouse_position = Toolbox.snap_to_grid(pygame.mouse.get_pos())
            print(f"Left click at ({mouse_position[0]}, {mouse_position[1]})")
            
            if mode == "delete" or mode == "new":
                MODE_DICT[mode](mouse_position)
            
            elif mode == "clear":
                screen = MODE_DICT[mode](screen)
            
            elif mode == "save":
                MODE_DICT[mode](screen, font)
    
    
        # Track mouse drag: store position on press, compare on release
            if Modes.get_mode() == "move":
                dragged_vertex_ID = VertexListOperations.get_vertex_ID_at_position(mouse_position)
                if dragged_vertex_ID != -1:
                    drag_start_position = event.pos
                    dragging = True
        
    #IF MOUSE RELEASE
    #dy, dx fed into Modes.move_vert
    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
        if dragging:
            drag_end_position = pygame.mouse.get_pos()
            dx = drag_end_position[0] - drag_start_position[0]
            dy = drag_end_position[1] - drag_start_position[1]
            
            if dx+dy != 0 and dragged_vertex_ID != -1: #if vertex has been moved
                print(f"vertex {dragged_vertex_ID} dragged from {drag_start_position} to {drag_end_position}, delta: ({dx}, {dy})")
                Modes.move_vertex(dragged_vertex_ID, (dx, dy))
                
            dragged_vertex_ID = 0
            dragging = False
    
    screen.fill((0, 0, 0, 0))
    spline.render_path(screen)
    VertexListOperations.render_all_vertices(screen)
    pygame.display.flip()

    return "draw"

