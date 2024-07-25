from .tree_objects import Node, Tree
import numpy as np
from .utils import labels_to_clusters
from collections import namedtuple
from itertools import chain

class FBM(Tree):
    """Tree subclass for generating a fixed Bayesian Markov tree.

    Args:
        Tree (object): FBM Tree object over vocabulary size V and max_depth
    """
    def __init__(self, V, max_depth):
        super().__init__(V,max_depth)

    def generate_children(self, current_node):
        if current_node.get_depth() < self.max_depth: 
            for k in range(self.V):  
                child_index = current_node.get_index() + (k,)
                child_node = Node(index=child_index, cluster=[k])  
                current_node.add_child(child_node)  
                self.generate_children(child_node)  
        else:
            # Set index of current node as a leaf index
            self.set_leaf_index(current_node.get_index())
                
    def generate_tree(self):
        self.root = Node()
        self.generate_children(self.root)

    def __str__(self):
        out = namedtuple('FBMTree', ['V','order','n_leaves'])
        return f'{out(self.V, self.max_depth, len(self.leaf_indices))}'

class CRPTree(Tree):
    """Tree subclass for simulating a PBCT using recursive Chinese restaurant (CRP) process partitioning.

    Attributes:
        V (int): Size of the vocabulary over which partitions are generated.
        max_depth (int): Maximum depth of a generated tree.
        alpha (float): Rate parameter for the CRP. If decay_rate is True, then 
            alpha is the initial rate at depth 0.
        decay_rate (bool): If True, decay the rate parameter alpha as tree depth increases.

    """
    def __init__(self, V=1, max_depth=1, alpha=1.0, decay_rate=True, gamma=1.0):
        super().__init__(V, max_depth)
        
        if isinstance(alpha, (float,int,np.int64)):
            if not alpha>0:
                raise ValueError('Rate parameter (alpha) must be a positive float.')
            else:
                self.alpha=alpha
        else:
            raise TypeError('Rate parameter (alpha) must be a positive float.')

        if not isinstance(decay_rate, bool):
            raise TypeError('decay_rate must be a boolean.')
        else:
            self.decay_rate = decay_rate
        
        self.gamma = gamma

    def __str__(self):
        out = namedtuple('CRPTree', ['V','alpha','max_depth','n_leaves'])
        return f'{out(self.V, self.alpha, self.max_depth, len(self.leaf_indices))}'
    
    def chinese_restaurant_process(self, d=0):
        """Generates a partition of the vocabulary V using the Chinese Restaurant Process (CRP) with rate parameter alpha.

        Args:
            alpha (float): Rate parameter of CRP. Defaults to 1.0.

        Returns:
            dict: Dictionary of clusters forming a partition of V.
                Example: V={0,1,2,3,4}, clusters={0:[1,2], 1:[3,5], 2:[4]}  
        """
        # Check depth parameter
        if not isinstance(d,(int,np.int64)):
            raise TypeError('Depth must be a non-negative integer.')
        else:
            if d<0:
                raise ValueError('Depth must be a non-negative integer.')
            
        # Decay CRP rate parameter alpha with depth if specified
        alpha = self.alpha*np.exp(-self.gamma*d) if self.decay_rate else self.alpha

        # Initialise labels and counts
        labels = np.zeros(self.V, dtype=int)
        labels[0] = 0
        K = 1
        counts = [1, alpha]

        # Sample cluster labels for the next words in vocabulary
        for v in range(1, self.V):
            probs = np.array(counts) / (alpha + v)
            z = np.random.choice(K+1, p=probs)
            labels[v] = z
            if z < K:
                counts[z] += 1
            else:
                counts[z] = 1
                counts.append(alpha)
                K+=1
        return labels_to_clusters(labels)

    def generate_children(self, current_node):
        """For a current node in the tree, generate its children via the CRP.

        Args:
            current_node (Node object): Current node in tree as a Node object
        """
        # Get depth of current node
        current_depth = current_node.get_depth()
        # If maximum depth not yet reached, attempt to generate children of current node
        if current_depth < self.max_depth:
            # Generate clusters forming a random partition of vocabulary
            clusters = self.chinese_restaurant_process(d=current_depth)

            # If the partition is non-trivial, i.e. more than one cluster is generated
            if len(clusters) > 1:
                # Iterate over each cluster
                for k, cluster in clusters.items():
                    # Index child node to represent its position among its siblings, appended to index of current node
                    child_index = current_node.get_index() + (k,)
                    # Create child node object with new index and corresponding cluster
                    child_node = Node(index=child_index, cluster=cluster)
                    # Add child node to current node object
                    current_node.add_child(child_node)
                    # Add index
                    self.indices.append(child_index)
                    # Add cluster to tree
                    self.clusters[child_index] = cluster
                    # Recursively generate the children of the child node
                    self.generate_children(child_node)
            else:
                # Set index of current node as a leaf index
                leaf_index = current_node.get_index()
                self.set_leaf_index(leaf_index)
                self.indices.append(leaf_index)
        else:
            # Set index of current node as a leaf index
            leaf_index = current_node.get_index()
            self.set_leaf_index(leaf_index)

    def generate_tree(self):
        # Initialise by creating root node object.
        # By default, index=(), cluster=None and children=[].
        self.root = Node()
        
        # Generate tree by recursively generating children, starting from the root node.
        self.generate_children(self.root)
    
    def simulate(self, N=10, eta=1.0):
        # Generate tree
        self.generate_tree()
        while not self.root.get_children():
            self.leaf_indices.clear()
            self.indices.clear()
            self.clusters.clear()
            self.generate_tree()

        # Simulate data
        self.draw_pmfs(eta=eta, wildcard=False)
        x0 = np.random.randint(self.V, size=self.max_depth)

        return self.simulate_data(N=N, x0=x0)[:-len(x0)]
            
class SimulateVBM(Tree):
    """Tree subclass to generate a variable-order Bayesian Markov tree structure"""
    def __init__(self, V, max_depth, theta=0.5, decay_theta=False, gamma=1.0):
        super().__init__(V, max_depth)

        if isinstance(theta, (float, int, np.int64)):
            if not theta>0:
                raise ValueError('Split probability (theta) must be a positive float.')
            else:
                self.theta=theta
        else:
            raise TypeError('Split probability (alpha) must be a positive float.')

        if not isinstance(decay_theta, bool):
            raise TypeError('decay_theta must be a boolean.')
        else:
            self.decay_theta = decay_theta
        
        self.gamma = gamma
    
    # With probability theta, generate V child nodes (split), else stop.
    def generate_children(self, current_node):
        d = current_node.get_depth()
        if d < self.max_depth: 
            # Decay split probability theta with depth if specified
            theta = self.theta*np.exp(-self.gamma*d) if self.decay_theta else self.theta

            # split w.p. theta (sample=1) else stop (sample=0)
            split = bool(np.random.binomial(1, theta))
            if split:
                for v in range(self.V):  
                    child_index = current_node.get_index() + (v,)
                    child_node = Node(index=child_index, cluster=[v])  
                    current_node.add_child(child_node)
                    # Add index
                    self.indices.append(child_index)
                    # Add cluster to tree
                    self.clusters[child_index] = [v]
                    # Recursively continue generating
                    self.generate_children(child_node)
            else:
                # Set index of current node as a leaf index
                leaf_index = current_node.get_index()
                self.set_leaf_index(leaf_index)
                self.indices.append(leaf_index)
        else:
            # Set index of current node as a leaf index
            self.set_leaf_index(current_node.get_index())
                
    def generate_tree(self):
        self.root = Node()
        self.generate_children(self.root)
        self.update_max_depth()

    def __str__(self):
        out = namedtuple('VBMTree', ['V', 'theta', 'max_depth','n_leaves'])
        return f'{out(self.V, self.theta, self.max_depth, len(self.leaf_indices))}'

