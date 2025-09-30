#testings
import pygame
import time
import numpy as np

pygame.init()
window_size = (700, 700)
screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
pygame.display.set_caption("Diffusion Simulation")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)


size = 100

next_grid = np.zeros((size, size)).tolist()


now_grid = next_grid.copy() #(for setup)


time_step = 1/30
diff = 0.1
N = size-2 #(N = size of grid - 2, to allow for adjacent squares etc. will be grid-1 when adjacent squares stuff works)

a = time_step * diff * N * N
iteration = 0

def diffuse():
    global now_grid, next_grid
    for k in range(0, 4): #number of iterations of diffusion per frame
        for i in range(0, N+2):
            for j in range(0, N+2): #for every grid cell

                # if an adjacent square is off the grid, dont try to add, and change the 4 to however many squares are being summed
                summed_square_count = 4

                #can also check for if adjacent square is boundary square here
                if i <= 0:
                    left_square_value = 0
                    summed_square_count-=1
                else:
                    left_square_value = next_grid[i-1][j]
                if i == N+1:
                    right_square_value = 0
                    summed_square_count-=1
                else:
                    right_square_value = next_grid[i+1][j]

                if j <= 0:
                    top_square_value = 0
                    summed_square_count-=1
                else:
                    top_square_value = next_grid[i][j-1]
                if j == N+1:
                    bottom_square_value = 0
                    summed_square_count-=1
                else:
                    bottom_square_value = next_grid[i][j+1]

                next_grid_sum = left_square_value + right_square_value + bottom_square_value + top_square_value
                next_grid[i][j] = round((now_grid[i][j] + a * next_grid_sum) / (1 + summed_square_count * a), 5) #rounds to 5dp
        
        now_grid = next_grid

def advect():
    pass
    #need to do
    
    """
    f = (x,y) - V(x,y) * dt
    i = floor(f)
    j = fract(f)
    lerp(a, b, k) = a + k * (b - a)
    z1 = lerp(d(i.x, i.y), d(i.x+1, i.y), j.x)
    z2 = lerp(d(i.x, i.y+1), d(i.x+1, i.y+1), j.x)
    d'(x,y) = lerp(z1, z2, j.y)
    d' = new density

    Then need to clear divergence of velocity field:
    divergence(x,y) = 0.5*(v.x(x+1,y) - v.x(x-1,y) + v.y(x,y+1) - v.y(x,y-1))
    p(x,y) = 0.25*(p(x-1,y) + p(x+1,y) + p(x,y-1) + p(x,y+1) - divergence(x,y))
    get into matrix form
    use SOR to solve
    grad(p) = (0.5 * (p(x+1,y) - p(x-1,y)), 0.5 * (p(x,y+1) - p(x,y-1)))

    v' = v - grad(p)
    v' = new velocity

    """



def square_drawing():
    global square_size
    square_size = window_size[0] // len(now_grid)
    for i in range(len(now_grid)):
        for j in range(len(now_grid[i])):
            if now_grid[i][j] > 0:
                value = now_grid[i][j]
                colour_intensity = min(255, int(value * 200))
                pygame.draw.rect(screen, (0, colour_intensity, 0), (j * square_size, i * square_size, square_size, square_size))



running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        #if click, add pressure to that square
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            now_grid[mouse_y//(square_size)][mouse_x//(square_size)] += 200  # Add pressure on click


    screen.fill((0, 0, 0))
    diffuse()
    square_drawing()

    fps = clock.get_fps()
    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))

    clock.tick(30)
    pygame.display.flip()