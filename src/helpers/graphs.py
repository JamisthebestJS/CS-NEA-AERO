#graphs

#matplotlib stuff showing how forces are changing over time.

import matplotlib.pyplot as plt

#currently just a random graph as a placeholder sort of thing.

plt.ion()  # turning interactive mode on

drag_list = []
lift_list = []
iteration_list = []
graph = None

# the update loop
def render_graph():
    global graph
    # removing the older graph
    if graph != None:
        del(graph)
    # plotting newer graph
    

    plt.plot(iteration_list, lift_list, color='r', label='Lift')
    plt.plot(iteration_list, drag_list, color='g', label='Drag')
    plt.xlabel("Iteration")
    plt.ylabel("Force Magnitude")
    plt.legend(["Lift", "Drag"])
    plt.pause(0.25)


def update_graphs(drag_item, lift_item, it_count):
    drag_list.append(drag_item)
    lift_list.append(lift_item)
    iteration_list.append(it_count)
    #every 5k iterations, halve the number of points to plot to ensure not too many points to plot at higher iteration levels
    if len(iteration_list) % 5000 == 0:
        for i in range(len(iteration_list)):
            if i%2 == 0:
                del(drag_list[i])
                del(lift_list[i])
                del(iteration_list[i])    
    render_graph()

def clear_graph():
    global drag_list, lift_list, iteration_list, graph
    drag_list = []
    lift_list = []
    iteration_list = []
    graph = None