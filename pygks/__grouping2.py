#the end is near

from copy import deepcopy
from numpy import *
from oldgraph.classes.graph import graph
from oldgraph.algorithms.accessibility import connected_components,accessibility
import itertools

def is_max(node,gr,density):
    neighbor_set = gr.neighbors(node)
    if (density[node] >= density[neighbor_set]).all():
        return True
    else:
        return False

def group(setN,gr,write_mark,minimum_cluster,alpha):
##    print gr
##    print write_mark
    N = len(setN)
##    print N
    density = zeros(N)
    for i in range(N):
        distances = 0.0
        neighbor_set = gr.neighbors(i)
        for each_node in neighbor_set:
            tmp_d = setN[i] - setN[each_node]
            distances += sqrt(inner(tmp_d,tmp_d))
        if distances == 0.0 or len(neighbor_set) == 0:
##            print i,N,'in the pool'
##            raw_input('ISB..')
            distances = 0.0
        else:
##            print distances,float(len(neighbor_set))
            distances = distances / float(len(neighbor_set))
        density[i] = 1.0 / pow(1 + distances,2.0)
    density_copy = deepcopy(density)
##    print density
    #* 1 *
    remain_set = set(range(N))
    clusters = {}
    for i in range(N):
        if is_max(i,gr,density):
            clusters[i] = i
            remain_set.remove(i)
            density_copy[i] = -1
##    print 'remaining set',len(remain_set)
    #* 2 *
    while len(remain_set) >0:
        unlabeled_max = argmax(density_copy)
        density_copy[unlabeled_max] = -1
    #* 3 *
        neighbor_set = gr.neighbors(unlabeled_max)
        tmp_density = density[neighbor_set]
        label_index = neighbor_set[argmax(tmp_density)]
        clusters[unlabeled_max] = clusters[label_index]
        remain_set.remove(unlabeled_max)
    #algorithm 2
    #find boundary edges
    cluster_centers = list(clusters.values())
##    print len(cluster_centers),cluster_centers
    borders = []
    for each_edge in gr.edges():
        if clusters[each_edge[0]] != clusters[each_edge[1]]:
            if each_edge[0] < each_edge[1]:
                borders.append(each_edge)
    #find big clusters
    connected_groups = connected_components(gr)
    group_count = len(set(connected_groups.values()))
    big_cluster_head = []
    tmp_set_appeared = set([])
    for head,group_index in list(connected_groups.items()):
        if group_index not in tmp_set_appeared:
            big_cluster_head.append(head)
            tmp_set_appeared.add(group_index)
    
    #construct edge set
    heads_tails = accessibility(gr)
    head_and_tail = {}
    for head,tail in list(heads_tails.items()):
        if head in big_cluster_head:
            head_and_tail[connected_groups[head]] = tail

    #tresholds of the super clusters
    Gc = {}
    for head,tail in list(head_and_tail.items()):
        tmp_tresh = 0.0
        count = 0.0
        for each_edge in itertools.combinations(tail,2):
            if gr.has_edge(each_edge):
                count += 1.0
                tmp_tresh += abs(density[each_edge[0]] - density[each_edge[1]])
        if count == 0.0:
##            print('sigularity')
            Gc[head] = 0.0
        else:
            Gc[head] = alpha * tmp_tresh / count
    #* 2 *
##    print 'group,tresh',borders,Gc
    while len(borders) > 0:
##        print borders,'borders'
        current_border = borders.pop()
        Dab = max(density[current_border[0]],density[current_border[1]])
        Dca = density[clusters[current_border[0]]]
        Dcb = density[clusters[current_border[1]]]
        Gtresh = Gc[connected_groups[current_border[0]]]
##        print Dca - Dab,Dcb - Dab,Gtresh,'x'
        if connected_groups[current_border[0]] != connected_groups[current_border[1]]:
            input('there be a problem')
        if ((Dca - Dab < Gtresh) | (Dcb - Dab < Gtresh)) == False:
            clusterA = clusters[current_border[0]]
            clusterB = clusters[current_border[1]]
##            print gr,'sb'
            gr.del_edge(current_border)
            tmp_borders = deepcopy(borders)
            for each_edge in tmp_borders:
                if (clusters[each_edge[0]] == clusterA and clusters[each_edge[1]] == clusterB) or (clusters[each_edge[1]] == clusterA and clusters[each_edge[0]] == clusterB):
##                    raw_input('del')
                    gr.del_edge(each_edge)
                    borders.remove(each_edge)

##    out_cast = []
##    for i in range(len(setN)):
##        if gr.neighbors(i) == 0:
##            out_cast.append(i)
##    for isolation in out_cast:
##        gr.del_node(isolation)
##        setN.remove(isolation)

##    print len(setN)
##    print write_mark
    if write_mark:
        connected_groups = connected_components(gr)
        group_count = len(set(connected_groups.values()))
        if minimum_cluster > group_count:

            grade = minimum_cluster - group_count
            
            from pygraph.algorithms.minmax import cut_tree
            yourcut = cut_tree(gr)
##            print 'cut tree',yourcut.values(),grade
            yourset = list(yourcut.values())
            for i in range(grade):
                print(min(yourset))
                yourset.remove(min(yourset))
            max_degree = min(yourset)
##            print max_degree,yourset
            
            for edge_name, cut_degree in list(yourcut.items()):
                if (cut_degree < max_degree + 1) and (gr.has_edge(edge_name)):
##                    print edge_name
                    gr.del_edge(edge_name)
                    print('cluster break x 1',edge_name)


if __name__ == '__main__':
    gr = graph()
    setN = [array([0.0,0.0]),array([1.0,0.0]),array([0.5,0.5]),array([0.5,1.0]),array([-0.1,-0.1]),array([2.0,1.0]),array([2.0,0.0]),array([1.5,0.5])]
    gr.add_nodes(list(range(8)))
    gr.add_edge((0,1))
    gr.add_edge((0,2))
    gr.add_edge((2,1))
    gr.add_edge((3,2))
    gr.add_edge((0,4))
    gr.add_edge((5,6))
    gr.add_edge((5,7))
    gr.add_edge((7,6))
    group(setN,gr,True,1,0.01)
    print(gr,connected_components(gr))

