#insertion sort implementation

def insertion_sort(list):
    for i in range(len(list)):
        item = list[i]
        j = i-1
        
        while j >= 0 and item < list[j]:
            list[j+1] = list[j]
            j-=1
        list[j+1] = item

    return list