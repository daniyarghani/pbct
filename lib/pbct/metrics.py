import numpy as np
from .tree_objects import Node
from .utils import clusters_to_labels
from copy import deepcopy
from collections import defaultdict
from sklearn.metrics.cluster import adjusted_rand_score as ari

def compare_trees(tree_fit, tree_sim
                  ,V=None
                  ,x=None
                  ,return_partitions=False
                  ,compare_max_depth=2
                  ,compare_full_trees=True):
    """
    Compute the ARI between clusterings at depths 1 and 2 for two clustered Markov trees.
    """
    if not isinstance(x, (list, dict)):
        raise ValueError('Input sequence x is missing or x is not of type {list/dict}.')
    if not V:
        assert tree_fit.V == tree_sim.V
        V = tree_fit.V
        
    # Ensure all nodes at depth < d have children;
    # if original node has no children, add trivial partition
    def fill_empty_children(current_node, tree):        
        d=current_node.get_depth()
        if d<compare_max_depth:
            index = current_node.get_index()
            children = current_node.get_children()
            if not children:
                trivial_node = Node(index=index+(0,), cluster=list(range(V)))
                current_node.add_child(trivial_node)
                tree.clusters[index+(0,)] = list(range(V))
                fill_empty_children(trivial_node, tree)
            else:
                for k, child_node in enumerate(children):
                    fill_empty_children(child_node, tree)
    
    def get_full_tree(tree):
        full_tree = deepcopy(tree)
        fill_empty_children(full_tree.root, full_tree)
        return full_tree
    
    if compare_full_trees:
        full_tree_fit, full_tree_sim = get_full_tree(tree_fit), get_full_tree(tree_sim)
    else:
        full_tree_fit, full_tree_sim = tree_fit, tree_sim
    
    # Get partitions at each depth from dictionary of all nodes/clusters
    def get_partitions(clusters):
        # # Deal with Nested CVBM clusters
        # if isinstance(list(clusters.keys()).pop(), int):
        #     partitions = defaultdict(lambda: defaultdict(dict))
        #     for d, clustering in clusters.items():
        #         partitions[d][()] = clustering
            
        #     return partitions

        partitions = defaultdict(lambda: defaultdict(dict))
        for index, cluster in clusters.items():
            d = len(index)
            # Store parent indices only
            partitions[d][index[:-1]].update({index[-1]:cluster})
        
        return partitions

    # Store clusterings at each depth 
    partitions_fit, partitions_sim = get_partitions(full_tree_fit.clusters), get_partitions(full_tree_sim.clusters)
          
    # Initialise ARI values
    ari_values = defaultdict(dict)
    ari_values_fit = defaultdict(dict)
    
    # Iterate over depths
    d_1,d_2 = len(partitions_fit), len(partitions_sim)
    d_max = min(compare_max_depth, max(d_1,d_2))
    for d in range(1,d_max+1):
        # Compute cross-tabulated ARIs
        for index_fit in partitions_fit[d]:
            ari_values_fit[d][index_fit] = []
            for index_sim in partitions_sim[d]:
                labels_fit = clusters_to_labels(partitions_fit[d][index_fit],V=V)
                labels_sim = clusters_to_labels(partitions_sim[d][index_sim],V=V)
                
                ari_value = ari(labels_fit, labels_sim)
                
                # Store ARI value for each compared clustering
                ari_values[d][index_fit+index_sim] = ari_value
                
                # Store ARI values as list for each fitted clustering
                ari_values_fit[d][index_fit].append(ari_value)
    
    # Get weighted average ARIs
    counts_fit = defaultdict(list) # Initialise counts to weight ARI
    ari_values_average = defaultdict(list) # Initial ARI values for averaging
    for d in range(1,d_max+1):
        for index in ari_values_fit[d]:
            # For each fitted clustering, get max ARI compared with simulated clusterings
            ari_value_max = np.max(ari_values_fit[d][index])
            ari_values_average[d].append(ari_value_max)
            
            # Get count of sequence elements attributed to the current node
            counts_fit[d].append(full_tree_fit.get_counts(x=x
                                                          ,leaf_indices=[index]
                                                          ,return_total_counts=True
                                                         ).get(index)
                                )
        
        # print(counts_fit[d])
        # print(ari_values_average[d])
        
        # Weight ARI values by node counts
        ari_values_average[d] = np.average(ari_values_average[d], weights=counts_fit[d])
    
    out = {}
    out['partitions_1'] = dict(partitions_fit)
    out['partitions_2'] = dict(partitions_sim)
    out['ari_values'] = dict(ari_values)
    out['ari_values_average'] = dict(ari_values_average)
    
    return out
