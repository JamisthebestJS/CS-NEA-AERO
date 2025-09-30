import pygame
from abc import ABC, abstractmethod
from enum import Enum, auto




"""
Modes:
- new vertex
- delete vertex
- move vertex
- clear all
"""
class Modes(object):
    current_mode = ""
    
    @staticmethod
    def get_vertex(vertex_x, vertex_y):
        id = VertexListOperations.get_vertex_id(vertex_x, vertex_y)
        return VertexListOperations.get_vertex(id)
        
    @staticmethod
    def clear_all():
        Vertex.vertices.clear()

    @staticmethod
    def place_vertex(x, y):
        print(f"Placing vertex at ({x}, {y})")
        new_vertex = Vertex(x, y)
        VertexListOperations.append_vertices_list(new_vertex)
        if VertexListOperations.get_vertices_list_length() > 1:
            last_vertex = VertexListOperations.get_vertex(VertexListOperations.get_vertices_list_length() - 2)
            new_line = Line(last_vertex, new_vertex)
            LineListOperations.new_line(new_line)
        #need to then update lines

    @staticmethod
    def delete_vertex(vertex_x, vertex_y):
        print(f"Deleting vertex at ({vertex_x}, {vertex_y})")
        VertexListOperations.del_from_vertices_list(vertex_x, vertex_y)
        #need to then update lines
    
    @staticmethod
    def move_vertex(start_pos, dx, dy):
        print(f"Moving vertex at {start_pos} by ({dx}, {dy}) to ({start_pos[0] + dx}, {start_pos[1] + dy})")
        id = VertexListOperations.get_vertex_ID(start_pos[0], start_pos[1])
        vertex = VertexListOperations.get_vertex(id)
        vertex.update_pos(start_pos[0] + dx, start_pos[1] + dy)
        LineListOperations.update_lines(deleted_vertex=None, moved_vertex = vertex) #the issue is, the line has old vertex coords, so wont find connected lines
        #could save the old position in the Vertex class, and use it 
    
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
        return VertexListOperations.vertices[index]
    
    @staticmethod
    def get_vertices_list_length():
        return len(VertexListOperations.vertices)
    
    @staticmethod
    def del_from_vertices_list(x, y):
        id = VertexListOperations.get_vertex_ID(x, y)
        VertexListOperations.vertices.pop(id)
        
    @staticmethod
    def append_vertices_list(vertex):
        VertexListOperations.vertices.append(vertex)
        
    @staticmethod
    def get_vertex_ID(x, y, in_rad=False):
        for i in range(VertexListOperations.get_vertices_list_length()):
            vertex = VertexListOperations.vertices[i]
            #if same pos as click
            if vertex.x == x:
                if vertex.y == y:
                    return vertex.ID
            
            #if within radius of click (5 pixels) (using circle stuff)
            if in_rad:
                if (vertex.x - x) ** 2 + (vertex.y - y) ** 2 <= 5 ** 2:
                    return vertex.ID
        
        return -1 #if not in vertex list
    
class LineListOperations(object):
    lines = []
    
    @staticmethod
    def render_all_lines():
        for line in LineListOperations.lines:
            """line.render()"""
            pass
    
    @staticmethod
    def get_line(index):
        return LineListOperations.lines[index]
    
    @staticmethod
    def get_lines_list_length():
        return len(LineListOperations.lines)
    
    @staticmethod
    def del_from_lines_list(line):
        LineListOperations.lines.remove(line)
        
    @staticmethod
    def new_line(line):
        LineListOperations.lines.append(line)
        
    @staticmethod
    def find_lines_with_vertex(vertex):
        connected_lines = []
        for line in LineListOperations.lines:
            print(f"Checking line from {line.start_vertex.get_pos()} to {line.end_vertex.get_pos()} against vertex at {vertex.get_pos()} (old pos {vertex.get_old_pos()})")
            if line.start_vertex.get_pos() == vertex.get_old_pos() or line.end_vertex.get_pos() == vertex.get_old_pos():
                connected_lines.append(line)
        return connected_lines
    
    @staticmethod
    def update_lines(deleted_vertex, moved_vertex):
        #every time a line needs updating, this is called. called from move_vertex, del_vertex, new_vertex. This will then do the updating stuff
        #when a vertex is moved, the new vertex is fed into the line instance with the old vertex (2 lines should have this)
        #when a vertex is deleted, the 2 lines connected to it are deleted, and those 2 lines' other vertices are used to create new vertex
        #^^ calls the method new_line
        if deleted_vertex is not None:
            deleted_vertex_id = deleted_vertex.ID
            connected_lines = LineListOperations.find_lines_with_vertex(deleted_vertex)
            if len(connected_lines) == 2:
                #if 2 lines are connected to the deleted vertex, make a new line between the other 2 vertices of these lines
                other_vertex_1 = connected_lines[0].start_vertex if connected_lines[0].end_vertex == deleted_vertex else connected_lines[0].end_vertex #gets other vertex of one line
                other_vertex_2 = connected_lines[1].start_vertex if connected_lines[1].end_vertex == deleted_vertex else connected_lines[1].end_vertex #gets other vertex of other line
                new_line = Line(other_vertex_1, other_vertex_2)
                LineListOperations.new_line(new_line)
                #then delete the 2 old lines
                LineListOperations.del_from_lines_list(connected_lines[0])
                LineListOperations.del_from_lines_list(connected_lines[1])
                
            elif len(connected_lines) == 1:
                #if only 1 line is connected to the deleted vertex, just delete that line
                LineListOperations.del_from_lines_list(connected_lines[0])
            else:
                #if no lines are connected to the deleted vertex, do nothing
                pass
    
        if moved_vertex is not None:
            #need to find the 2 lines connected to moved vertex, and update start/end vertex to the new vertex position
            connected_lines = LineListOperations.find_lines_with_vertex(moved_vertex)
            #problem is with start/end, how to kow which one to change
            connected_lines[0]

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
        self.old_x = None
        self.old_y = None
    
    def get_pos(self):
        return (self.x, self.y)
    
    def get_old_pos(self):
        return (self.old_x, self.old_y)
    
    def update_pos(self, new_x, new_y):
        self.old_x = self.x
        self.old_y = self.y
        self.x = new_x
        self.y = new_y
    
    def render(self):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), 2)

#Line class
class Line(object):
    def __init__(self, start_vertex, end_vertex):
        #takes instances of Vertex class as vertices (NOT COORDS)
        self.start_vertex = start_vertex
        self.end_vertex = end_vertex
    """    
    #this will at some point do spline implementation
    def render_start_to_end(self):
        first_vertex = VertexListOperations.get_vertex()
        last_vertex = VertexListOperations.get_vertex()
        pygame.draw.line(screen, (0, 255, 0), (first_vertex.x, first_vertex.y), (last_vertex.x, last_vertex.y), 2)
        
    #this will at some point do spline implementation
    def render(self):
        pygame.draw.line(screen, (0, 255, 0), (self.start_vertex.x, self.start_vertex.y), (self.end_vertex.x, self.end_vertex.y), 2)
        self.render_start_to_end()
    """
    
    
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
                    dragged_vertex_ID = VertexListOperations.get_vertex_ID(mouse_x, mouse_y, in_rad=True)
                    if dragged_vertex_ID != -1:
                        drag_start_pos = pygame.mouse.get_pos()
                        dragging = True
            
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if 'drag_start_pos' in locals() and dragging:
                    drag_end_pos = pygame.mouse.get_pos()
                    dx = drag_end_pos[0] - drag_start_pos[0]
                    dy = drag_end_pos[1] - drag_start_pos[1]
                    print("dragging vertex", dragged_vertex_ID)
                    
                    if (dx != 0 or dy != 0) and dragged_vertex_ID != -1: #if vertex has been moved
                        print(f"Cursor dragged from {drag_start_pos} to {drag_end_pos}, delta: ({dx}, {dy})")
                        Modes.move_vertex(drag_start_pos, dx, dy)
                        
                    dragged_vertex_ID = 0
                    dragging = False
                    
        
        LineListOperations.render_all_lines()
        VertexListOperations.render_all_vertices()
        pygame.display.flip()
        clock.tick(30)



    pygame.quit()
    exit()
    
    
if __name__ == "__main__":
    __main__()