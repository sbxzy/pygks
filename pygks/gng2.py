"""Growing Neural Gas main file, for developers only."""
from numpy import *
from pygraph.classes.graph import graph

winner_index = 0
winner_2nd_index = 0
setN = [] #:set of neuron weights
gr = graph() #:topology structure implemented by python-graph
T_winner = 0
T_2nd_winner = 0
accumulated = [1,1]
big_clusters = []
density = []
age_max = 0
nn_lambda = 0
alpha = 0
t = 1
numbers = [1,1]
setLabel = [0,0]
miminum_cluster = 2
delete_noise = False
eb = 0.01
en = 0.001
accumulated_error = [0,0]
ann = 1.0
bnn = 1.0


def set_parameter(age,lambda_set,ann_set,bnn_set,eb_,en_):
    """Initilization of GNG, calling this function after training will reset the
    neural network for further training. age, lambda_set, ann_set, bnn_set, eb_,
    en_ are the GNG parameters meaning max age, learning step, winner adaptation size,
    neighbor of winner adaptation size, and error reduction sizes."""
    global age_max #:max age parameter of GNG
    global nn_lambda #:training step of GNG
    global alpha
    global setN #:set of neuron weights
    global accumulated #:winning times of each neuron
    global numbers
    global density
    global big_clusters #:a dictionary stating which cluster of a neuron belong to
    global t
    global minimum_cluster
    global gr
    global delete_noise
    global eb
    global en
    global ann
    global bnn
    global accumulated_error
    ann = ann_set
    bnn = bnn_set
    eb = eb_
    en = en_
    t = 1
    setN = []
    accumulated = [1,1]
    accumulated_error = [0,0]
    numbers = [1,1]
    big_clusters = []
    density = []
    nn_lambda = lambda_set
    age_max = age
    gr = graph()
    return

def min_max_in_tresholds(neighbours,index):
    global setN
    treshold = 0.0
    if len(neighbours) == 0:
        distances = sum(pow(array(setN - setN[index]),2),axis = -1)**0.5
        distances[index] = float('inf')
        treshold = min(distances)
    else:
        distances = []
        i = 0
        for i in neighbours:
            distances.append(linalg.norm(setN[i] - setN[index]))
        treshold = max(distances)
    return treshold

def tresholds():
    global setN
    global T_winner
    global T_2nd_winner
    global gr

    winner_neighbours = gr.neighbors(winner_index)
    T_winner = min_max_in_tresholds(winner_neighbours,winner_index)

    winner_2nd_neighbours = gr.neighbors(winner_2nd_index)
    T_2nd_winner = min_max_in_tresholds(winner_2nd_neighbours,winner_2nd_index)
    return

def neighbour_count(index):
    global gr
    return len(gr.neighbors(index))

def remove_node(index):
    """Remove a neuron specified by 'index'."""
    global setN
    global gr
    bf = gr.neighbors(index)
    bfc = len(gr.neighbors(len(setN)-1))
    if (len(setN)-1) in bf:
        bfc -= 1

    last_node = len(setN) - 1
    index_neighbors = gr.neighbors(index)
    for each_node in index_neighbors:
        gr.del_edge((index,each_node))
    last_node_neighbors = gr.neighbors(last_node)
    for each_node in last_node_neighbors:
        gr.add_edge((each_node,index))
        gr.set_edge_weight((each_node,index),gr.get_edge_properties((each_node,last_node))['weight'])
    gr.del_node(last_node)
    setN[index] = setN[last_node]
    accumulated_error[index] = accumulated_error[last_node]
    accumulated[index] = accumulated[last_node]
    setN.pop(-1)
    accumulated_error.pop(-1)
    accumulated.pop(-1)
    if len(setN) != index:
        if len(gr.neighbors(index)) != bfc:
            print index,bfc,gr.neighbors(index)
            raw_input('remove error')

    return

def come_together():
    grouping2.group(setN,gr,False,minimum_cluster,alpha)
    return

def stop_and_write():
    global setN
    i = 0
    while (i !=len(setN)) & (len(setN) > 2):
        for j in gr.neighbors(i):
                if j == i:
                    print 'neuron confliction, doing nothing'
        if neighbour_count(i) < 1:
            remove_node(i)
            print 'removed neuron', i
        else:
            i += 1
    print 'End training!'
    return


def step(point,pLabel,tx):
    """The GNG training procedures in each step. 'point' is the
    input vector. 'pLabel' is the label of the input vector and
    set to 0 if unlabeled. 'tx' is the mark for end training
    (when set to -1)."""
    global winner_index
    global winner_2nd_index
    global setN
    global t
    global en
    global eb
    global accumulated_error

    if t == 1:
        setN.append(point)
        gr.add_node(0)

    elif t == 2:
        setN.append(point)
        gr.add_node(1)
        
    elif tx == -1: #-1 is the mark of finishing training
        stop_and_write()
    else:

        distances = sum((array(setN - point))**2,axis = -1)
        winner_index = argmin(distances)
        the_error = distances[winner_index]
        distances[winner_index] = float('inf')
        winner_2nd_index = argmin(distances)

        if gr.has_edge((winner_index,winner_2nd_index)) == False:
            gr.add_edge((winner_index,winner_2nd_index))
            gr.set_edge_weight((winner_index,winner_2nd_index),0)
        setN[winner_index] += eb*(point - setN[winner_index])
        accumulated_error[winner_index] += the_error
        accumulated[winner_index] += 1
        i = 0
        for i in range(len(setN)):
            if gr.has_edge((winner_index,i)):
                setN[i] += en*(point - setN[i])
                gr.set_edge_weight((winner_index,i),(gr.get_edge_properties((winner_index,i))['weight']+1))
                if (gr.get_edge_properties((winner_index,i))['weight'] > age_max):
                    gr.del_edge((winner_index,i))
                    if (len(gr.neighbors(winner_index)) == 0) and (len(setN) > 2):
                        remove_node(winner_index)

        if (t + 1) % nn_lambda == 1:
            q_node = argmax(accumulated_error)
            q_error = accumulated_error[q_node]

            max_error = float('-inf')
            max_index = q_node
            for i in gr.neighbors(q_node):
                if accumulated_error[i] > max_error:
                    max_error = accumulated_error[i]
                    max_index = i
                
            f_node = max_index
            f_error = max_error
            if q_node== f_node:
                print 'error deleting node'
                print q_node, f_node, len(setN)
            new_point = 0.5*(setN[q_node] + setN[f_node])
            new_index = len(setN)
            setN.append(new_point)
            gr.add_node(new_index)
            gr.add_edge((q_node,new_index))
            gr.set_edge_weight((q_node,new_index),0)
            if f_node != q_node:
                gr.add_edge((f_node,new_index))
                gr.set_edge_weight((f_node,new_index),0)
            else:
                print('del warning')
            if gr.has_edge((q_node,f_node)):
                gr.del_edge((q_node,f_node))
            accumulated_error[q_node] -= ann*accumulated_error[q_node]
            accumulated_error[f_node] -= ann*accumulated_error[f_node]
            accumulated_error.append(0.5*(accumulated_error[q_node] +accumulated_error[f_node]))
            accumulated_error[q_node] *= 0.5
            accumulated_error[f_node] *= 0.5
            accumulated.append((accumulated[q_node] +accumulated[f_node]))
    t += 1
    accumulated_error = list(array(accumulated_error)*(1.0 - bnn))
        
