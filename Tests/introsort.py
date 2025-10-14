#introsort

# Python implementation of Introsort algorithm

import math
from heapq import heappush, heappop

arr = []


def heapsort(arr):
    h = []
    # building the heap
    for value in arr:
        heappush(h, value)
    arr = []
    # extracting the sorted elements one by one
    arr = arr + [heappop(h) for _ in range(len(h))]


def InsertionSort(arr, begin, end):
    # Traverse through 1 to len(arr)
    for i in range(begin+1, end + 1):
        key = arr[i]
        j = i-1
        while j >= begin and arr[j] > key:
            arr[j+1] = arr[j]
            j -= 1
        arr[j + 1] = key

def Partition(arr, low, high):
  # pivot
    pivot = arr[high]
  # index of smaller element
    i = low - 1
    for j in range(low, high):
        # If the current element is smaller than or
        # equal to the pivot
        if arr[j] <= pivot:
            # increment index of smaller element
            i +=1
            temp = arr[j]
            arr[j] = arr[i]
            arr[i] = temp
            
    (arr[i + 1], arr[high]) = (arr[high], arr[i + 1])
    temp = arr[i+1]
    arr[i+1] = arr[high]
    arr[high] = temp
    
    return i+1


def MedianOfThree(A, B, C):
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


def IntrosortUtil(arr, begin, end, depthLimit):
    size = end - begin
    if size < 16:
        # if the data set is small, call insertion sort
        InsertionSort(arr, begin, end)
        return

    if depthLimit == 0:
        # if the recursion limit is occurred call heap sort
        heapsort(arr)
        return
    
    pivot = MedianOfThree(arr[begin], arr[begin+size//2], arr[end])
    (arr[pivot], arr[end]) = (arr[end], arr[pivot])
    # partitionPoint is partitioning index,
    # arr[partitionPoint] is now at right place
    partitionPoint = Partition(arr, begin, end)
    # Separately sort elements before partition and after partition
    IntrosortUtil(arr, begin, partitionPoint - 1, depthLimit - 1)
    IntrosortUtil(arr, partitionPoint + 1, end, depthLimit - 1)


# A utility function to begin the Introsort module
def Introsort(arr, begin, end):
    depthLimit = 2 * math.floor(math.log2(end - begin))
    IntrosortUtil(arr, begin, end, depthLimit)

def main(arr):
    Introsort(arr, 0, len(arr) - 1)
    print ('Arr: ', arr)


if __name__ == '__main__':
    
    #test data
    array = [
        2, 10, 24, 2, 10, 11, 27,
        4, 2, 4, 28, 16, 9, 8,
        28, 10, 13, 24, 22, 28,
        0, 13, 27, 13, 3, 23,
        18, 22, 8, 8 ]

    
    
    main(array)