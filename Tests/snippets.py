#snippets


import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Scrollable List")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)

# Font
FONT = pygame.font.Font(None, 36)








"""
Cheeky little scrollable menu. Nice for aerofoil designs so that I don't need to implement save slots. Still should be able to delete and name them though.
"""
# List items
items = [f"Item {i}" for i in range(1, 51)]  # 50 items
item_height = 50
list_height = len(items) * item_height

# Scroll variables
scroll_y = 0
scroll_speed = 20

# Main loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                scroll_y = min(scroll_y + scroll_speed, 0)
            elif event.button == 5:  # Scroll down
                scroll_y = max(scroll_y - scroll_speed, -(list_height - HEIGHT))

    # Clear screen
    screen.fill(WHITE)

    # Draw items
    for i, item in enumerate(items):
        item_rect = pygame.Rect(50, i * item_height + scroll_y, WIDTH - 100, item_height)
        if 0 <= item_rect.y <= HEIGHT:  # Only draw visible items
            pygame.draw.rect(screen, GRAY if i % 2 == 0 else WHITE, item_rect)
            text = FONT.render(item, True, BLACK)
            screen.blit(text, (item_rect.x + 10, item_rect.y + 10))

    # Update display
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()




"""
Introsort

"""



import math
from heapq import heappush, heappop

def heapsort():
    global arr
    h = []

    # build heap
    for value in arr:
        heappush(h, value)
    arr = []

    # extract sorted els individually
    arr = arr + [heappop(h) for i in range(len(h))]


def insertion_sort(start, end):
    left = start
    right = end

    for i in range(left + 1, right + 1):
        key = arr[i]

        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j = j - 1
        arr[j + 1] = key


# last element as pivot.
# takes pivot  at correct pos in sorted array
# places all smaller to left of pivot
# all greater els to right
def partition(low, high):
    global arr
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i+=1
            (arr[i], arr[j]) = (arr[j], arr[i])
    (arr[i + 1], arr[high]) = (arr[high], arr[i + 1])
    return i + 1


# f(x) to find the median of 3 els
def three_median(a, b, c):
    global arr
    A = arr[a]
    B = arr[b]
    C = arr[c]
    
    #branchless 3-way comparison
    A1 = A
    B1 = B
    if (A > B):
        A = B1
        B = A1
    if (B > C):
        B = C
    if (A > B):
        B = A
        

def intro_sort_util(start, end, depth_lim):
    #if small list insertion
    if end - start < 16:
        insertion_sort(start, end)
        return

    #if recursion limit, call heap sort
    if depth_lim == 0:
        heapsort()
        return

    pivot = three_median(start, end//2, end)
    (arr[pivot], arr[end]) = (arr[end], arr[pivot])

    partition_point = depth_lim(start, end)

    #Separately sort bits before and after partition
    introsort(start, partition_point - 1, depth_lim - 1)
    introsort(partition_point + 1, end, depth_lim - 1)


def introsort(start, end):
    # init depth_lim as 2 * log(length(list))
    depth_lim = 2 * math.floor(math.log2(end - start))
    intro_sort_util(start, end, depth_lim)
    

def do_introsort(arr):     
    n = len(arr)
    introsort(0, n - 1)
    print ('Arr: ', arr)

"""
Rearrange to form Max Heap
WHILE size(heap) > 1
    swap root element with last element
    remove the element which was swapped with
    heapify remaining elements
END WHILE




FUNCTION Quicksort(Array, low, high):
   IF (low < high):
       pivot = median(Array, low, high);
       Quicksort(Array, low, pivot – 1);  
       Quicksort(Array, pivot + 1, high); 
    ENDIF
END FUNC

#
FUNCTION median(first, middle, last):
    first_1 = first
    middle_1 = middle
    
    IF (first > middle):
        first = middle_1
        middle = first_1
    END IF
        
    IF (middle > last):
        middle = last
    END IF
        
    IF (first > middle):
        middle = first
    END IF
    RETURN middle
END FUNC


FUNCTION partition(array):
    pivot = array[0]
    
    WHILE partitioning = True:

        WHILE looking = True:
            i = i + 1
            if array[i] >= pivot:
                END WHILE
        
        WHILE looking = True:
            j = j - 1
            if array[j] <= pivot:
                END WHILE
        
        IF i > j:
            END WHILE
        END IF
        
        array[i], array[j] = array[j], array[i]

"""