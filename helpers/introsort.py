#introsort

# Python implementation of Introsort algorithm

import math

#sorts consistently sized arrys/tuples based on whichever el you want
def heapify(arr, n, i, pos):
    #find largest among root and children
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i][pos] < arr[left][pos]:
        largest = left

    if right < n and arr[largest][pos] < arr[right][pos]:
        largest = right

    #if root is not largest, swap w/ largest and continue heapifying
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest, pos)


def heapsort(arr, pos):
    n = len(arr)

    #max heaping
    for i in range(n//2, -1, -1):
        heapify(arr, n, i, pos)

    for i in range(n-1, 0, -1):
        #swap
        arr[i], arr[0] = arr[0], arr[i]

        #heapify root el
        heapify(arr, i, 0, pos)



def insertion_sort(arr, begin, end, pos):
    # Traverse through 1 to len(arr)
    for i in range(begin+1, end + 1):
        inner_arr = arr[i]
        j = i-1
        while j >= begin and arr[j][pos] > inner_arr[pos]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j + 1] = inner_arr



def partition(arr, low, high, pos):
  # pivot
    pivot = arr[high]
  # index of smaller el
    i = low - 1
    for j in range(low, high):
        # If the current el is smaller than or
        # equal to the pivot
        if arr[j][pos] <= pivot[pos]:
            # increment index of smaller el
            i +=1
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
    
    temp = arr[i+1]
    arr[i+1] = arr[high]
    arr[high] = temp
    
    return i+1


def three_median(A, B, C, pos):
    A1 = A
    B1 = B
    if (A[pos] > B[pos]):
        A = B1
        B = A1
    if (B[pos] > C[pos]):
        B = C
    if (A[pos] > B[pos]):
        B = A
    return B


def introsort_util(arr, begin, end, depthLimit, pos):
    size = end - begin
    if size < 16:
        # if data set small, do insertion
        insertion_sort(arr, begin, end, pos)
        return

    if depthLimit == 0:
        # if recursion limit occurred do heap
        heapsort(arr, pos)
        return
    
    pivot = three_median(arr[begin], arr[begin+size//2], arr[end], pos)
    temp = arr[end]
    arr[end] = arr[arr.index(pivot)]
    arr[arr.index(pivot)] = temp
    # partitionPoint partitioning index
    # arr[partitionPoint] now at right place
    partitionPoint = partition(arr, begin, end, pos)
    # Separately sort els before and after partition
    introsort_util(arr, begin, partitionPoint - 1, depthLimit - 1, pos)
    introsort_util(arr, partitionPoint + 1, end, depthLimit - 1, pos)


#begin introsort
def start_introsort(arr, begin, end, pos):
    depthLimit = 2 * math.floor(math.log2(end - begin))
    introsort_util(arr, begin, end, depthLimit, pos)

def introsort(arr, pos):
    if len(arr) > 1:
        start_introsort(arr, 0, len(arr) - 1, pos)
    
    



#sorts items, not arrays by an el
def heapify_items(arr, n, i):
    #find largest among root and children
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[i] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    #if root is not largest, swap w/ largest and continue heapifying
    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify_items(arr, n, largest)


def heapsort_items(arr):
    n = len(arr)

    #max heaping
    for i in range(n//2, -1, -1):
        heapify_items(arr, n, i)

    for i in range(n-1, 0, -1):
        #swap
        arr[i], arr[0] = arr[0], arr[i]

        #heapify root el
        heapify_items(arr, i, 0)



def insertion_sort_items(arr, begin, end):
    # Traverse through 1 to len(arr)
    for i in range(begin+1, end + 1):
        item = arr[i]
        j = i-1
        while j >= begin and arr[j] > item:
            arr[j+1] = arr[j]
            j -= 1
        arr[j + 1] = item



def partition_items_items(arr, low, high):
  # pivot
    pivot = arr[high]
  # index of smaller el
    i = low - 1
    for j in range(low, high):
        # If the current el is smaller than or
        # equal to the pivot
        if arr[j] <= pivot:
            # increment index of smaller el
            i +=1
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
    
    temp = arr[i+1]
    arr[i+1] = arr[high]
    arr[high] = temp
    return i+1


def three_median_items(A, B, C):
    A1 = A
    B1 = B
    if (A > B):
        A = B1
        B = A1
    if (B > C):
        B = C
    if (A > B):
        B = A
    return B


def introsort_util_items(arr, begin, end, depthLimit):
    size = end - begin
    if size < 16:
        # if data set small, do insertion
        insertion_sort_items(arr, begin, end)
        return

    if depthLimit == 0:
        # if recursion limit occurred do heap
        heapsort_items(arr)
        return
    
    pivot = three_median_items(arr[begin], arr[begin+size//2], arr[end])
    temp = arr[end]
    arr[end] = pivot
    pivot = temp
    # partitionPoint partitioning index
    # arr[partitionPoint] now at right place
    partitionPoint = partition_items_items(arr, begin, end)
    # Separately sort els before and after partition
    introsort_util_items(arr, begin, partitionPoint - 1, depthLimit - 1)
    introsort_util_items(arr, partitionPoint + 1, end, depthLimit - 1)


#begin introsort
def start_introsort_items(arr, begin, end):
    depthLimit = 2 * math.floor(math.log2(end - begin))
    introsort_util_items(arr, begin, end, depthLimit)

def introsort_items(arr):
    if len(arr) > 1:
        start_introsort_items(arr, 0, len(arr) - 1)
    
    return arr