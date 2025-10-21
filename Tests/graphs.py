#graphs

#matplotlib stuff showing how forces are changing over time.

import matplotlib.pyplot as plt
import random

#currently just a random graph as a placeholder sort of thing.

plt.ion()  # turning interactive mode on



    
# preparing the data
x_list = []
y_list = []

# the update loop
def render_graph(graph, x_list, y_list):
    # removing the older graph
    del(graph)
    
    # plotting newer graph
    graph = plt.plot(x_list,y_list,color = "g")[0]
    plt.xlim(x_list[0], x_list[-1])

    plt.pause(0.25)


def update_x_y(graph, x, y):
    x_list.append(x)
    y_list.append(y)
    render_graph(graph, x_list, y_list)
    
def first_graph():

    # plotting the first frame
    graph = plt.plot(x_list,y_list)[0]
    plt.ylim(0,2)
    plt.pause(1)
    
    return graph

        