import isoinn2
from pygraph.classes.graph import graph
from pygraph.algorithms.minmax import cut_tree
from pygraph.algorithms.accessibility import connected_components
from utils import __dict_reverse as dict_reverse
import itertools
import time
from numpy import array,sum,sqrt

class data_block:
    """This is the programming and user interface for SOINN. data is the training dataset, should be array or list, with each element numpy arrays with the same dimension.
    no_label is True or False. While set to False, the last element of each array in data will be treated as labels.
    The rest of the variables are training settings for SOINN."""
    nodes = [] #:Weight of neurons.
    gr = {} #:Topology structures.
    def __init__(self,data,no_label = True,age_max = 200,nn_lambda = 70,alpha = 2.0,del_noise = True,un_label = 0):
        isoinn2.set_parameter(age_max,nn_lambda,alpha,0,del_noise)
        timecost = time.time()
        t = 0
        gr = graph()
        if no_label:
            for n_point in data:
                t += 1
                isoinn2.step(n_point,un_label,t)
        else:
            for n_point in data:
                t += 1
                n_data = list(n_point)
                n_X = array(n_data[0:-1])
                n_Y = n_data[-1]
                isoinn2.step(n_X,n_Y,t)

        isoinn2.step(array([]),0,-1)
        print 'time cost',time.time() - timecost
        self.nodes = isoinn2.setN
        self.gr = isoinn2.gr
        print len(self.nodes)

    def output_graph(self):
        """Return the topology structure as a python-graph."""
        return self.gr

    def output_nodes(self):
        """Return the list of neuron weights."""
        return self.nodes
    
    def graph_features(self):
        """Generating topological features including vertice orders for future use."""
        gr_nodes = self.gr.nodes()
        gr_edges = self.gr.edges()
        node_count = len(gr_nodes)
        edge_count = len(gr_edges) / 2.0
        average_order = 0.0
        clustering_coefficient = 0.0
        max_order = 0
        for each_node in gr_nodes:
            #for orders
            current_node_order = self.gr.node_order(each_node)
            average_order += current_node_order
            max_order = max(max_order,current_node_order)
            #now for clustering coefficient
            direct_neighbors = self.gr.neighbors(each_node)
            tmp_v_edge_count = 0.0
            tmp_r_edge_count = 0.0
            for virtual_edge in itertools.product(direct_neighbors,direct_neighbors):
                if virtual_edge[0] != virtual_edge[1]:
                    tmp_v_edge_count += 1.0
                    if self.gr.has_edge(tuple(virtual_edge)):
                        tmp_r_edge_count += 1.0
            if tmp_v_edge_count == 0:
                clustering_coefficient += 0.0
            else:
                clustering_coefficient += (tmp_r_edge_count / tmp_v_edge_count)
        clustering_coefficient /= float(node_count)
        average_order /= float(node_count)
        #for kernel order
        cut_dict = cut_tree(self.gr)
        cut_places = set(cut_dict.values())
        how_many_kernel_orders = range(5)
        kernel_orders = []
        bloods = 0.0
        for kernel_tick in how_many_kernel_orders:
            if kernel_tick in cut_places:
                bloods += 1.0
            kernel_orders.append(bloods)
        #for redundant edges and missing edges
        redundant_edges = 0.0
        missing_edges = 0.0
        for each_edge in gr_edges:
            node0 = each_edge[0]
            node1 = each_edge[1]
            #find common set of nodes' neighbors
            common_set = set(self.gr.neighbors(node0)).intersection(set(self.gr.neighbors(node1)))
            if len(common_set) == 0:
                missing_edges += 1.0
            elif len(common_set) > 1:
                in_cell_edges = list(itertools.combinations(list(common_set),2))
                cell_judge = True
                for cell_edge in in_cell_edges:
                    if self.gr.has_edge(cell_edge):
                        cell_judge = False
                if cell_judge == False:
                    redundant_edges += 1.0
        if edge_count != 0.0:
            redundant_edges /= float(edge_count)
            missing_edges /= float(edge_count)

        #average edge lenghth
        total_length = 0.0
        for each_edge in gr_edges:
            node0 = each_edge[0]
            node1 = each_edge[1]
            total_length += sqrt(sum((self.nodes[node0] - self.nodes[node1])**2))
        if len(gr_edges) == 0:
            average_length = 0.0
        else:
            average_length = total_length / float(len(gr_edges))
            
            
        return [average_length,node_count,edge_count,average_order,max_order,redundant_edges,missing_edges] + kernel_orders

    def draw_2d(self, scale = 1, axis_ = False):
        """Draws the topology structure and neurons. scale is real number, it can be set arbitrarily to adjust the size
        of drawed neuron clusters. axis is True or False, and means weither to enable axis in the final drawings.
        In this method, MDS is used for drawing high dimensional Euclidean graphs. If you do not use this method, sklearn is
        not a prerequisite for running the pygks software."""
        groups = connected_components(self.gr)
        if len(self.nodes[0]) != 2:
            print('using MDS for none 2d drawing')
            from sklearn import manifold
            from sklearn.metrics import euclidean_distances
            similarities = euclidean_distances(self.nodes)

            for i in range(len(self.nodes)):
                for j in range(len(self.nodes)):
                    if groups[i] == groups[j]:
                        similarities[i,j] *= scale
            
            mds = manifold.MDS(n_components=2, max_iter=500, eps=1e-7,dissimilarity="precomputed", n_jobs=1)
            pos = mds.fit(similarities).embedding_
            draw_nodes = pos
        else:
            draw_nodes = self.nodes
        print('now_drawing')
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111)
        node_count = len(draw_nodes)
        for i in range(node_count):
                for j in range(i,node_count):
                        if self.gr.has_edge((i,j)):
                                 ax.plot([draw_nodes[i][0], draw_nodes[j][0]],[draw_nodes[i][1], draw_nodes[j][1]], color='k', linestyle='-', linewidth=1)
        group_counts = len(set(groups.values()))
        style_tail = ['.','o','x','^','s','+']
        style_head = ['b','r','g','k']
        style_list = []
        for each in itertools.product(style_head,style_tail):
            style_list.append(each[0]+each[1])
        i = 0
        for each in draw_nodes:
            ax.plot(each[0],each[1],style_list[groups[i]-1])
            i += 1
        if not axis_:
            plt.axis('off')
        plt.show()

    def outlier_nn(self,positive = 1,negative = -1):
        """This method finds the largest neuron cluster. If a neuron belongs to this cluster, a label specified by positive will
        be added to this neuron, else this neuron will be labeled by negative variable. The labeled results will be outputed in a
        list as labels_final."""
        groups = connected_components(self.gr)
        #find the largest group
        group_counts = dict_reverse(groups)
        max_count = 0
        for keys,values in group_counts.items():
            if len(values) > max_count:
                max_count = len(values)
                max_group = keys
        
        affines = {}
        for keys,values in groups.items():
            if values == max_group:
                affines[values] = positive
            else:
                affines[values] = negative
                
        #this is only for outlier detection
        for values in groups.values():
            if values not in affines.keys():
                affines[values] = -1      
        for keys,values in groups.items():
            groups[keys] = affines[values]
        labels_final = []
        for i in range(len(self.nodes)):
            labels_final.append(groups[i])
        print labels_final
        return self.nodes, labels_final

    def counts(self):
        """Output the winning times of each neuron and the accumulated errors of the SOINN network."""
        return isoinn2.accumulated, isoinn2.numbers

if __name__ == '__main__':
    print 'sb'
        
