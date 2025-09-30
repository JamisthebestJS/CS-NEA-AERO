#testing some catmull stuff
import pygame

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Catmull-Rom Spline Test")


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


def catmull_rom_path(points, resolution):
    path = []
    n = len(points)
    for i in range(n-1):
        p0 = points[i - 1] if i > 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 < n else points[i + 1]
        for j in range(resolution):
            t = j / resolution
            #if i < n - 1 or j == 0:
            path.append(catmull_rom(p0, p1, p2, p3, t))
        
        #trying to get first to join to last point
    for j in range(resolution):
        t = j / resolution
        path.append(catmull_rom(points[-2], points[-1], points[0], points[1], t))
            
    return path


example_points = [[100.72, 300.3], [155, 95], [47, 73.8], [70.1, 30.5], [109.7, 57.2], [406.4, 13.27]]



for i in example_points:
    pygame.draw.circle(screen, (255, 0, 0), (int((i[0] + 1)), int((i[1] + 1))), 5)

path = catmull_rom_path(example_points, 1000)
for i in path:
    print(f"{i[0]} {i[1]}")
    pygame.draw.circle(screen, (255, 255, 255), (int((i[0] + 1)), int((i[1] + 1))), 2)
    

running = True
clock = pygame.time.Clock()
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()
    clock.tick(30)
pygame.quit()
