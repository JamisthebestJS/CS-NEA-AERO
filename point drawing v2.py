import pygame
from abc import ABC, abstractmethod
from enum import Enum, auto



CAT_ROM_DEF = 1000 #number of points between each pair of vertices in catmull rom spline


class CatmullRom(staticmethod):
    points = []
    path = []
    
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
            point.append(round(((a0[i] * t + a1[i]) * t + a2[i]) * t + a3[i], 8))
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
    def render_path():
        for point in CatmullRom.path:
            pygame.draw.circle(screen, (255, 255, 255), (int((point[0] + 1)), int((point[1] + 1))), 1)

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


class Modes(object):
    current_mode = ""
        
    @staticmethod
    def clear_all():
        Vertex.vertices.clear()

    @staticmethod
    def place_vertex(x, y):
        print(f"Placing vertex at ({x}, {y})")
        Vertex(x, y)
        CatmullRom.new_point((x, y))
        #need to then update lines

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
    def move_vertex(start_pos, dx, dy):
        vert_id = VertexListOperations.get_vertex_ID(start_pos[0], start_pos[1])
        vertex = VertexListOperations.get_vertex(vert_id)
        old_pos = vertex.get_pos()
        new_pos = (old_pos[0] + dx, old_pos[1] + dy)
        print(f"Moving vertex at {old_pos} by ({dx}, {dy}) to ({new_pos})")
        vertex.update_pos(new_pos[0], new_pos[1])
        CatmullRom.move_points(old_pos, new_pos)
    
    
    @staticmethod
    def save_to_file():
        pass
        #not sure exactly what format I want for this rn
    
    @staticmethod
    def set_mode(new_mode):
        Modes.current_mode = new_mode

    @staticmethod
    def get_mode():
        return Modes.current_mode


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
    def render_all_vertices():
        for vertex in VertexListOperations.vertices:
            vertex.render()
    
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
                print("Vertex not found in VertexListOperations.get_vertex")
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
    
    def render(self):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), 2)

    
    
pygame.init()
SCREEN_SIZE = 720
screen_width, screen_height = SCREEN_SIZE, SCREEN_SIZE
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Testing point drawing")
clock = pygame.time.Clock()

def __main__():
        
    running = True
    while running:
        
        
        for event in pygame.event.get():
            
            if event.type == pygame.QUIT:
                running = False
            
            #CHANGING MODES
            if event.type == pygame.KEYDOWN:
                    #for testing, set modes with keyboard keys
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
                    
                    print(f"Mode set to {Modes.get_mode()}")
            
            #IF MOUSE CLICK
            if event.type == pygame.MOUSEBUTTONDOWN:
                
                if event.button == 1:
                    # Mouse button 1 (left click) has been pressed
                    mouse_x, mouse_y = Toolbox.snap_to_grid(pygame.mouse.get_pos(), SCREEN_SIZE)
                    print(f"Left click at ({mouse_x}, {mouse_y})")
                    if Modes.get_mode() != "" and Modes.get_mode() != "move":
                        #move is dealt with separately below
                        MODE_DICT[Modes.get_mode()](mouse_x, mouse_y)
            
            
            # Track mouse drag: store position on press, compare on release
            #dy, dx fed into Modes.move_vert
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if Modes.get_mode() == "move":
                    dragged_vertex_ID = VertexListOperations.get_vertex_ID(mouse_x, mouse_y)
                    if dragged_vertex_ID != -1:
                        drag_start_pos = pygame.mouse.get_pos()
                        dragging = True
            
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if 'drag_start_pos' in locals() and dragging:
                    drag_end_pos = pygame.mouse.get_pos()
                    dx = drag_end_pos[0] - drag_start_pos[0]
                    dy = drag_end_pos[1] - drag_start_pos[1]
                    
                    if (dx != 0 or dy != 0) and dragged_vertex_ID != -1: #if vertex has been moved
                        print(f"Cursor dragged from {drag_start_pos} to {drag_end_pos}, delta: ({dx}, {dy})")
                        Modes.move_vertex(drag_start_pos, dx, dy)
                        
                    dragged_vertex_ID = 0
                    dragging = False
                    
        screen.fill((0, 0, 0, 0))
        CatmullRom.render_path()
        VertexListOperations.render_all_vertices()
        pygame.display.flip()
        clock.tick(30)



    pygame.quit()
    exit()
    
    
if __name__ == "__main__":
    __main__()


"""
TODO:
- some sort of file saving. The formatting of this is dependent on how I want to load it in the aero simulator
- GUI stuff. Buttons for modes, maybe a sidebar for info
- merge with aero simulator
- make catmull rom path update only the parts that need updating, not the whole thing (maybe? not sure if worth the effort)
(smaller things)
- add limit for number of vertices
- dont allow to place a vertex on top of another vertex (or too close)



ERRORS (causes crash):
-


BUG:
-


DONE (since last git push):

added:
- catmull rom spline drawing
- point moving
- point deleting

fixed:
- deleting vertices doesnt change catmullrom path
- adding new vertices after deleting some causes the new vertices to be undeletable
- clicking on a vertex which has been deleted (but is still rendering) deletes a vertex not clicked on (in catmull rom path, not sure if its still in VertexListOperations.vertices)
- vertices only stop rendering after you click on a vertex for the second time (which causes an error in VertexListOperations.get_vertex because the index is out of range (although no crash))
- if you try to delete a vertex and dont exactly click on it, it crashes as it cannot find the position in CatmullRom.points. Need to ensure that CatmullRom.points stores same pos as VertexListOperations.vertices
- if you try to delete the only vertex, it crashes. (index out of range in catmullrompath)
- Not clicking exactly on a vertex when trynig to move it causes it to not be updated in catmullrom.points
- moving a vertex doesnt change catmullrom path (even if it changes in catmullrom.points). (seems to not change catmullrom.points even though a print statement shows it gets to that point)
- moving a vertex moves the position from its original position to where you clicked rather than where you dragged it to? (in catmullrom.points)
- moving a point doesnt always trigger an update of catmullrom.points


removed:
- 





"""