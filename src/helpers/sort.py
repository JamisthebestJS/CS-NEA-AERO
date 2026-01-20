#insertion sort implementation

def insertion_sort(list):
    for i in range(len(list)):
        item = float(list[i])
        j = i-1
        
        while j >= 0 and item < float(list[j]):
            list[j+1] = float(list[j])
            j-=1
        list[j+1] = item

    return list


def aerofoil_sort(aerofoil_list):
    nums = []
    for aerofoil in aerofoil_list:
        nums.append(''.join(char for char in aerofoil if char.isdigit()))
    
    sorted_nums = insertion_sort(nums)

    sorted_aerofoils = []
    for num in sorted_nums:
        num_to_name = f"aerofoil {str(int(num))}.txt"
        for aerofoil in aerofoil_list:
            if num_to_name == aerofoil and aerofoil not in sorted_aerofoils:
                    sorted_aerofoils.append(aerofoil)
                    aerofoil_list.remove(aerofoil)
    
    return sorted_aerofoils