import matplotlib.pyplot as plt

plt.ion()  # turning interactive mode on

drag_list = []
lift_list = []
iteration_list = []
graph = None


def update_graphs(drag_item, lift_item, it_count):
    global drag_list, lift_list, iteration_list, graph
    drag_list.append(drag_item)
    iteration_list.append(it_count)
    #every 5k iterations, halve the number of points to plot to ensure not too many points to plot at higher iteration levels
    if len(drag_list) % 5000 == 0:
        drag_list = drag_list[::2]
        iteration_list = iteration_list[::2]
        if lift_item is not None:
            lift_list = lift_list[::2]
    
        
    if lift_item is None:
        plt.plot(iteration_list, drag_list, color='b', label='Thrust')
        plt.legend(["Thrust"])
    else:
        lift_list.append(lift_item)
        plt.plot(iteration_list, drag_list, color='b', label='Drag')
        plt.plot(iteration_list, lift_list, color='r', label='Lift')
        plt.legend(["Lift", "Drag"])
    
    # removing the older graph
    if graph != None:
        del(graph)
    # plotting newer graph
    plt.xlabel("Iteration")
    plt.ylabel("Force Magnitude")
    plt.pause(0.25)




def clear_graph():
    global drag_list, lift_list, iteration_list, graph
    drag_list = []
    lift_list = []
    iteration_list = []
    graph = None