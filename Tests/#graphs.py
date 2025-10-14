#graphs

#matplotlib stuff showing how forces are changing over time.







"""Pseudocode explaining the way of getting the 2 masks

#for each column which contains some aerofoil
for sublist in vert_sublist:
    #if odd number of aero nodes, ignore (temporarily)
    if len(sublist) % 2 != 0:
        continue
    else: 
        #even number of points in that column
        swaps = False
        y_start = 0
        #for each node in the sublist
        for node in sublist:
            #make True btwn the first and second, False btwn second and third, etc... in the vertical mask
            y_end = node[1]
            vert_mask[node[0], y_start:y_end] = swaps
            y_start = y_end
            #swaps beween True and False each time a node is reached
            swaps = (True if (swaps == False) else False)

#something going wrong here or in hori_sublist stuff
for sublist in hori_sublist:
    #if odd number of aero nodes, ignore (temporarily)
    if len(sublist) % 2 != 0:
        continue
    else: 
        #even number of points in that column
        swaps = False
        x_start = 0
        #for each node in the sublist
        for node in sublist:
            #make True btwn the first and second, False btwn second and third, etc... in the vertical mask
            x_end = node[0]
            vert_mask[x_start:x_end, node[1]] = swaps
            x_start = x_end
            #swaps beween True and False each time a node is reached
            swaps = (True if (swaps == False) else False)


FOR i=0 TO i=len(list_of_sublists):
    sublist = list_of_sublists[i]
    
    IF len(sublist) % 2 == 0:
        swaps = False
        start = 0
        
        FOR j=0 TO j=len(sublist):
            node = sublist[j]
            end = node[0]
            mask[start:end][node[1]] = swaps
            start = end
            
            IF swaps = True:
                swaps = False
            ELSE:
                swaps  = True:
            END IF
            
        END LOOP
        
    END IF
    
END LOOP
    

















ABADNDONED


#direction is as a vector: (0,1) - vertical or (1,0) - horizontal
FUNCTION make_mask(direction, list_of_sublists):

    for sublist in list_of_sublists:
        if len(sublist) % 2 != 0:
            continue
        else:
            swaps = False
            start = 0
            for node in sublist:
                end = node (elementwise multiplication, and then keep non-zero el) direction
                layer = node (elementwise multiplication, then keep non-zero el)  (direction (row interchanged))
                mask[start:end, layer] = swaps (ooh, but start:end and layer would need to be swapped for diff dirs)
                start = end
                swaps = (True if (swaps == False) else False)


FUNCTION MakeMask(direction, list_of_sublists):
    FOR i=0 TO len(list_of_sublists):
        sublist = list_of_sublists[i]
        IF len(sublist) % 2 == 0:
            swaps = False
            start = 0
            FOR i = 0 TO len(sublist):
                node = sublist[i]
                directed_node = ElementWiseMultiplication(node, direction)
                
                IF directed_node[0] == 0:
                    end = directed_node[1]
                ELSE:
                    end = directed_node[0]
                END IF

                not_direction = RowInterchange(direction)
                layer = ElementWiseMultiplication(node, not_direction)
                
                mask[]
            
            
            
"""