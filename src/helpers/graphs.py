import matplotlib.pyplot as plt

plt.ion()  # turning interactive mode on

drag_list = []
lift_list = []
iteration_list = []
graph = None


def update_graphs(drag_item, lift_item, it_count):
    drag_list.append(drag_item)
    iteration_list.append(it_count)
    #every 5k iterations, halve the number of points to plot to ensure not too many points to plot at higher iteration levels
    if len(iteration_list) % 5000 == 0:
        for i in range(len(iteration_list)):
            if i%2 == 0:
                del(drag_list[i])
                del(lift_list[i])
                del(iteration_list[i])
        
    if lift_item is None:
        plt.plot(iteration_list, drag_list, color='b', label='Thrust')
    else:
        lift_list.append(lift_item)
        plt.plot(iteration_list, drag_list, color='b', label='Drag')
        plt.plot(iteration_list, lift_list, color='r', label='Lift')
    
    global graph
    # removing the older graph
    if graph != None:
        del(graph)
    # plotting newer graph
    plt.xlabel("Iteration")
    plt.ylabel("Force Magnitude")
    plt.legend(["Lift", "Drag"])
    plt.pause(0.25)




def clear_graph():
    global drag_list, lift_list, iteration_list, graph
    drag_list = []
    lift_list = []
    iteration_list = []
    graph = None