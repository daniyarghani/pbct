import numpy as np
from .utils import logB
from collections import Counter

class Node:
    """Node object with attributes for its index, children and corresponding cluster of
    elements. Contains methods getting/setting attributes and adding a child node.

    Attributes:
        index (tuple): Index of the current node (self) as a tuple of ints.
        cluster (list): List of elements forming the cluster corresponding to current node.
        children (list): List of child nodes of current node.
    Methods: 
        {get,set}_{index,cluster} (function): get/set index/cluster of node.
        get_depth (function): Return depth of node.
        get_children (function): Return children of node.
        add_child (function): Add a child to the node.
    """
    def __init__(self, index=(), cluster=None, children=None):
        """Initialise Node object and its attributes.
        
        Args:
            index (tuple): Index of the current node (self) as a tuple of ints. Defaults to ().
            cluster (list): List of elements of cluster corresponding to current node. Defaults to None.
            children (list): children (list): List of child nodes of current node. Defaults to None.
        """
        # If initial index is specified, check correct type (tuple of ints)
        if isinstance(index, tuple):
            if all([isinstance(i,(int,np.int64)) for i in index]):
                self.index=index
            else:
                raise TypeError('index must be a tuple of integers.')
        else:
            raise TypeError('index must be a tuple of integers.')
        
        # If initial cluster given, check correct type (list of ints)
        if cluster is None:
            self.cluster = None
        elif isinstance(cluster, list) or isinstance(cluster, np.ndarray):
            if all([isinstance(i,(int,np.int64)) for i in cluster]):
                self.cluster=cluster
            else:
                raise TypeError('cluster must be a list or numpy array of integers.')
        else:
            raise TypeError('cluster must be a list or numpy array of integers.')

        # If initial children given, check correct type (list of Node objects)
        if children is None:
            self.children = []
        elif isinstance(children, list):
            if all([isinstance(node, Node) for node in children]):
                self.children=children
            else:
                raise TypeError('children must be a list of Node objects.')
        else:
            raise TypeError('children must be a list of Node objects.')
        

    def get_index(self):
        """Returns the index of the node"""
        return self.index
    
    def set_index(self, index):
        """Sets the index of the node"""
        self.index = index

    def get_cluster(self):
        """Returns the cluster corresponding to the node"""
        return self.cluster

    def set_cluster(self, cluster):
        """Sets the cluster corresponding to the node"""
        self.cluster = cluster

    def get_depth(self):
        """Returns the depth of the node in the tree."""
        return len(self.index)
    
    def get_children(self, as_clusters=False):
        """Returns list of child nodes of current node
        Args: 
            as_clusters (bool): If True, returns the children as clusters,
                otherwise returns list of Node objects.
        """
        if as_clusters:
            return [child.get_cluster() for child in self.children]
        else:
            return self.children

    def add_child(self, child_node):
        """Add a child for the current node (self)

        Args:
            child_node (Node object): Child node to add to current node.
        """
        self.children.append(child_node)
    
    def remove_children(self, children=None):
        """Remove a child node from current node (self)
        
        Args:
            child_node (Node): Child node to remove from current node.
        """
        if children is None:
            self.children.clear()

        elif isinstance(children, Node):
            try:
                self.children.remove(children)
            except:
                print(f'Child node not found for current node. Children of current node remain unchanged.')
        elif isinstance(children, list) and all([isinstance(child_node, Node) for child_node in children]):
            for child_node in children:
                try:
                    self.children.remove(child_node)
                except:
                    print(f'Child node {child_node.get_index()}:{child_node.get_cluster()} not found for current node.')
        else:
            raise TypeError('Input children must a Node object or list of Node objects.')
        
    
    def set_children(self, children):
        """Set the children of the node. Note: overwrites the children of the node;
        to add children to node use the add_child method.

        Args:
            children (list): List of child Node objects to set as children of current node.
        """
        self.children.clear()
        self.children += children


class Tree:
    def __init__(self, V=1, max_depth=1):
        """Tree object with methods used in tree simulation, sequence simulation, and tree fitting.
        """
        if isinstance(V, (int, np.int64)):
            if V<1:
                raise ValueError('Vocabulary size (V) must be a positive integer.')
            else:
                self.V=V
        else:
            raise TypeError('Vocabulary size (V) must be a positive integer')
        
        if isinstance(max_depth, (int, np.int64)):
            if max_depth<0:
                raise ValueError('Maximum tree depth (max_depth) must be a non-negative integer.')
            else:
                self.max_depth=max_depth
        else:
            raise TypeError('Maximum tree depth (max_depth) must be a non-negative integer')

        self.root = Node()
        self.leaf_pmfs = None
        self.pmfs = None
        self.leaf_indices=[]
        self.indices=[]
        self.clusters={}
    
    def update_max_depth(self):
        # Update tree max depth if the tree is learned
        if self.leaf_indices:
            self.max_depth = max([len(index) for index in self.leaf_indices])

    def set_leaf_index(self,index):
        self.leaf_indices.append(index)

    def get_node(self, index, *, return_node=True, as_cluster=False):
        """Returns the node at a given index; returns only the cluster if specified
        
        Args:
            index (tuple): Node index.
            as_cluster (bool, optional): If True, returns the cluster at the given index,
                otherwise returns the Node object. Defaults to False.
        Returns:
            cluster (list): List of elements in cluster.
        """
        if isinstance(index, tuple):
            if not all([isinstance(i,(int,np.int64)) for i in index]):
                raise TypeError('Input index must be a tuple of integers.')
        else:
            raise TypeError('index must be a tuple of integers.')

        # Initialise current node to root node
        current_node = self.root
        # If index is non-empty, traverse tree to the node at given index
        if index != ():
            # Set current node to root
            current_node = self.root
            # Iterate over elements i of index
            for i in index:
                try:
                    # Move to the i-th child of current node
                    current_node = current_node.get_children()[i]
                except IndexError: # Handle nonexistent index
                    print(f'IndexError: index {index} does not exist in tree.')
                    return None
        
        # After iteration, current node will be the target node
        # Return the cluster at current index if specified else return Node object
        if return_node:
            return current_node.get_cluster() if as_cluster else current_node 

    def del_node(self, index):
        """Delete the node and its children at a given index.
        
        Args:
            index (tuple): Node index.
        """
        if isinstance(index, tuple):
            if not all([isinstance(i,(int,np.int64)) for i in index]):
                raise TypeError('Input index must be a tuple of integers.')
        else:
            raise TypeError('index must be a tuple of integers.')

        # Initialise current node to root node
        current_node = self.root
        # If index is non-empty, traverse tree to the node at given index
        if index != ():
            # Set current node to root
            current_node = self.root
            # Iterate over elements i of index until target node
            for i in index:
                try:
                    # Move to the i-th child of current node
                    current_node = current_node.children[i]
                except IndexError: # Handle nonexistent index
                    print(f'IndexError: index {index} does not exist in tree.')
                    return None
        
        # After iteration, current node will be the node to delete
        del current_node

    def del_nodes(self,indices):
        for index in indices:
            self.del_node(index)
    
    def visualise_tree(self, show_indices=False, counts=None, leaf_ll=None):
        """Print the tree structure.

        Args:
            show_indices (bool, optional): Option to display the index of each node. Defaults to False.
        """
        # if not self.root.get_children():
        #     raise AttributeError('Tree object does not have any non-root nodes. Check that the nodes of the tree have been properly generated.')
        if not isinstance(show_indices, bool):
            raise TypeError('show_indices argument must be a boolean (True/False)')

        # Function to print nodes at the appropriate depth
        def visualise_node(node, depth):
            index = node.get_index()
            cluster = node.get_cluster()
            if not node.children:
                is_leaf = True
            else:
                is_leaf = False

            if show_indices:
                if is_leaf:
                    if counts and leaf_ll:
                        N = np.sum([value for value in counts.values()], dtype=int)
                        print('|' + '-' * depth + f'{index}: {cluster} -- {counts[index]} ({100*counts[index]/N:.2f}%), {leaf_ll[index]:.2f}') 
                    elif counts:
                        N = np.sum([value for value in counts.values()], dtype=int)
                        print('|' + '-' * depth + f'{index}: {cluster} -- {counts[index]} ({100*counts[index]/N:.2f}%)') 
                    elif leaf_ll:
                        print('|' + '-' * depth + f'{index}: {cluster} -- {leaf_ll[index]:.2f}')
                    else:
                        print('|' + '-' * depth + f'{index}: {cluster}')
                else:
                    print('|' + '-' * depth + f'{index}: {cluster}')
            else:
                print('|'+'-'*depth + f'{cluster}') 
            
            # Recursively print child nodes for each node
            for child in node.children:  
                visualise_node(child, depth + 1) 

        # Visualise entire tree by starting recursion at root node (depth 0).
        visualise_node(self.root, 0)
    
    def find_leaf_index(self, x, start_index=()):
        """Find the leaf index of the tree corresponding to an input sequence.

        Args:
            x (list or numpy array): Observed data as a sequence of integers.
            start_index (tuple): Index of node at which to start traversal; if node is at depth d then the sequende is traversed starting from the d-th last element.

        Returns:
            (tuple): Index of leaf node corresponding to the input sequence.
        """
        # if not self.root.get_children():
        #     raise AttributeError('Tree object does not have any non-root nodes. Check that the nodes of the tree have been properly generated.')
        if not isinstance(x, (list, np.ndarray)):
            raise TypeError('Input sequence (x) must be a list or numpy array of integers.')

        if not isinstance(start_index, tuple):
            raise TypeError('Starting index must be a tuple of integers.')
        if not all([isinstance(i, (int,np.int64)) for i in start_index]):
            raise TypeError('Starting index must be a tuple of integers.')
        
        depth = len(start_index) # Depth at which to start traversal
        index = start_index # Set index to starting index
        current_node = self.get_node(start_index)  # Set current node to node at starting index
        # print(current_node.get_index(), current_node.get_cluster())

        # If the current node is a leaf, return the leaf index
        if not current_node.get_children():
            return index
        
        # Iterate over sequence elements in reverse order, from starting depth
        for element in x[-depth-1:-self.max_depth-1:-1]:
            # Track whether the current element is in the tree
            in_tree = False
            # For each child of current node
            for child_node in current_node.get_children():
                # Check if current sequence element is in child cluster
                if element in child_node.get_cluster():
                    in_tree = True # Element found in tree
                    # Move to the corresponding child node
                    index = child_node.get_index() # Update current index
                    current_node = child_node # Update current node to child node
                    # If the node is a leaf, return the leaf index
                    if not current_node.get_children():
                        return index
                    # Else, continue traversing the tree (go to previous element)
                    else:
                        break 

            if not in_tree:  # Element not found in tree
                raise AttributeError(f'Element {element} does not exist in the tree. Check that the input sequence and tree contain elements from the same vocabulary.')

        
        # Return -1 at end of index if there is missing data
        # or not enough sequence history to fully traverse the tree (wildcard index)
        return index+(-1,)

    def draw_pmfs(self, eta, wildcard=False, HDP=False, lam=0,
                  word_specific_eta=False, eta_prior=None):
        """Draw Dirichlet probability mass functions (pmfs) for each leaf node of the tree.
        Must only be used with a populated tree.

        Args:
            eta (float or int): Dirichlet hyperparameter, must be positive.
            wildcard (bool, optional): Option to draw "wildcard" distributions
            for the case where a tree cannot be fully traversed, so that every (non-leaf) node
            has an additional child distribution corresponding to the "missing"/"non-existent" element.
            Defaults to False.
            HDP (bool,optional): Option to draw pmfs from a HDP
            lam (float, 0<=lam<=1): Parameter to control pmf mixture, lam=0 gives Dirichlet, lam=1 gives discrete Uniform

        Returns:
            (dict): Dictionary of Dirichlet pmfs for each leaf node with key=index, value=pmf array.
            Example: {(2,3,1):[0.1,0.1,0.1,0.7]}
        """
        # if not self.root.get_children():
        #     raise AttributeError('Tree object does not have any non-root nodes. Check that the nodes of the tree have been properly generated.')
        if not isinstance(eta, (float,int,np.int64)):
            raise TypeError('Hyperparameter (eta) must be a positive number.')
        if not eta>0:
            raise ValueError('Hyperparameter (eta) must be a positive number.')
        if not isinstance(wildcard, bool):
            raise TypeError('wildcard argument must be a boolean.')

        leaf_pmfs = {}
        pmfs = {}

        # Initialise dictionary of pmfs
        # Queue of nodes to process
        node_queue = [self.root]
        # Dirichlet process base measure: np.ones(V) gives standard Dirichlet pmf
        base = np.ones(self.V)
        # Process nodes in tree while queue of nodes is non-empty
        while node_queue:
            # Get node from front of queue
            node = node_queue.pop(0)
            # Get node index
            index = node.get_index()
            # Get children of current node
            children = node.get_children()

            # Get depth
            d = len(index)

            # eta_d = eta*np.exp(d-1) if HDP else eta
            eta_d = eta
            # print('d', d)
            # print('eta_d', eta_d)

            # If HDP, draw Dirichlet pmf using parent pmf as base measure
            if HDP and d>1:
                base = pmfs[index[:-1]]

            # print(f'index:{index}')
            # print(f'base: {base}')
            # print(f'eta: {eta}')
            # print(f'eta*base: {eta*base}')

            if d==0:
                pmf = np.ones(self.V)/self.V
            elif d>0 and index[-1] == -1:
                pmf = np.ones(self.V)/self.V
            else:
                pmf = np.zeros(self.V)
                # Deal with zero-elements in Dirichlet argument
                eta_vec = eta_d*base
                nonzero_indices = np.nonzero(eta_vec)
                eta_nonzero = eta_vec[nonzero_indices]

                # Word-specific eta
                if word_specific_eta:
                    eta_nonzero = eta_prior()
                    print(index, eta_nonzero)
                
                # print(eta_nonzero)
                V_nonzero = len(eta_nonzero)
                pmf[nonzero_indices] = lam*np.ones(V_nonzero)/V_nonzero + (1-lam)*np.random.dirichlet(eta_nonzero)
            
            pmfs[index] = pmf
            # If the node is a leaf, i.e. node has no children
            if not children:
                leaf_pmfs[index] = pmf
            # Otherwise, add children of current node to queue
            else: 
                for child_node in children:
                    node_queue.append(child_node)
                if wildcard:
                    # Add wildcard node with index ending in -1
                    wildcard_index = index + (-1,)
                    wildcard_node = Node(index=wildcard_index) 
                    node_queue.append(wildcard_node)

        self.leaf_pmfs=leaf_pmfs
        self.pmfs = pmfs
        self.eta=eta
        return leaf_pmfs

    def set_pmf(self, index, pmf, leaf=True):
        """Set the pmf corresponding to node at given index.

        Args:
            index (tuple of ints): node index
            pmf (size V array): probability mass function
        """
        assert (len(pmf)==self.V and sum(pmf)-1 < 1e-10), f'Input pmf must be a valid probability mass function of size {self.V}. Inputs must sum to 1'
        
        if not self.pmfs:
            self.pmfs={}
        if not self.leaf_pmfs:
            self.leaf_pmfs = {}

        self.pmfs[index] = pmf
        if leaf:
            self.leaf_pmfs[index] = pmf

    def predict_next_element(self, x, pmfs=None):
        """Sample the next element of sequence using the Markov tree model.

        Args:
            x (list or numpy array): Input data, a sequence of integers.
            pmfs (dict, optional): Predictive distributions corresponding to each leaf of the tree.
                Format: {leaf_index:pmf_array} 
                If pmfs not specified, uses self.leaf_pmfs if exists.

        Returns:
            (int): Next element of sequence.
        """
        if not isinstance(x, (list, np.ndarray)):
            raise TypeError('Input sequence (x) must be a list or numpy array of integers.')
        if not all([isinstance(i, (int,np.int64)) for i in x]):
            raise TypeError('Input sequence (x) must be a list or numpy array of integers.')
        if not all([i in range(self.V) for i in x]):
            raise ValueError('Input sequence (x) must be a list or numpy array of integers from 0 to V-1.')
        
        if not pmfs:
            if not self.leaf_pmfs:
                raise AttributeError('Call draw_pmfs method before predict_next_element, or provide a custom dictionary of distributions for each leaf as an argument.')
            else:
                pmfs = self.leaf_pmfs
        
        elif not isinstance(pmfs,dict):
            raise TypeError('Input distributions (pmfs) must be a dictionary with key=leaf_index (tuple) and value=probability mass function (list/array)')

        # Locate sequence in tree (find corresponding leaf index)
        index = self.find_leaf_index(x) 

        # Get corresponding predictive distribution
        phi = pmfs[index]
        # Predict next element
        next_x = np.random.choice(len(phi),p=phi)
        return next_x

    def simulate_data(self, N, pmfs=None, x0=[],return_leaf_counts=False):
        """Simulate a sequence of length N using the Markov tree model. 

        Args:
            N (int): Number of sequence elements to simulate.
            pmfs (dict, optional): Dictionary of predictive distributions for each leaf in tree.
                Format: {leaf_index:pmf_array} 
                If pmfs not specified, uses self.leaf_pmfs if exists.
            x0 (list or numpy array, optional): Initial sequence. Defaults to an empty list.

        Returns:
            (list): Simulated sequence of length len(x0)+N as a list.
        """
        if not (isinstance(N, (int,np.int64)) and N>0):
            raise ValueError('Sequence length (N) must be a positive integer.')
        if not isinstance(x0, (list, np.ndarray)):
            raise TypeError('Initial sequence (x0) must be a list or numpy array of integers.')
        if not all([isinstance(i, (int,np.int64)) for i in x0]):
            raise TypeError('Initial sequence (x0) must be a list or numpy array of integers.')
        if not all([i in range(self.V) for i in x0]):
            raise ValueError('Initial sequence (x0) must be a list or numpy array of integers from 0 to V-1.')
        
        if not pmfs:
            if not self.leaf_pmfs:
                raise AttributeError('Call draw_pmfs method before simulate_data, or provide a custom dictionary of distributions for each leaf as an argument.')
            else:
                pmfs = self.leaf_pmfs
        
        elif not isinstance(pmfs,dict):
            raise TypeError('Input distributions (pmfs) must be a dictionary with key=leaf_index (tuple) and value=probability mass function (list/array)')

        # Initialise sequence
        x = list(x0)
        N0 = len(x)
        if return_leaf_counts:
            leaf_counts = Counter({index:0 for index in self.leaf_indices})
        # Iteratively predict next element of sequence until N new elements are generated
        while len(x) < N + N0:
            # Traverse tree to find leaf corresponding to current element
            index = self.find_leaf_index(x)
            if return_leaf_counts:
                leaf_counts[index] += 1
            # Predict next element from leaf pmf
            pmf = pmfs[index]
            next_x = np.random.choice(len(pmf), p=pmf)
            # Append to sequence
            x.append(next_x)
        return (x, leaf_counts) if return_leaf_counts else x
    
    def plot_simulated_pmfs(self, all_pmfs=False, word_specific_eta=False, init_pmfs=False):
        import matplotlib.pyplot as plt
        import numpy as np
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'

        if not self.leaf_pmfs:
            print('PMFs have not been drawn for the leaf nodes of the tree.')
            return None
        
        if all_pmfs:
            pmfs = self.pmfs
            init_pmfs=all_pmfs
        else:
            pmfs = self.leaf_pmfs

        if not init_pmfs:
            drop_list = []
            for index in pmfs.keys():
                if index[-1] == -1:
                    drop_list.append(index)
            pmfs = {e:phi for e,phi in pmfs.items() if e not in drop_list}

        n_params = len(pmfs)
        size = int(np.sqrt(n_params))+1
        fig, axs = plt.subplots(nrows=size, ncols=size,
                                sharey=True, sharex=False, figsize=(6,6), dpi=300)

        for n, (e, phi) in enumerate(pmfs.items()):
            axs.flat[n].plot(phi)
            axs.flat[n].set_xticks(range(0,len(phi),len(phi)//4))
            axs.flat[n].set_xticklabels(range(0,len(phi),len(phi)//4))
            axs.flat[n].set_yticks(np.linspace(start=0,stop=1,num=5))
            # axs.flat[n].set_yticklabels(np.linspace(start=0,stop=1,num=4))
            index = "".join([str(i) for i in e])
            axs.flat[n].set_title(fr'$\phi_{{{index}}}$')
            
        for ax in axs.flat:
            if not ax.lines:
                fig.delaxes(ax)
            
            ax.set_xticks(range(len(phi)))
        
        try:
            alpha = self.alpha
        except:
            alpha = -1
            
        try:
            eta = self.eta
        except:
            eta = -1
        
        if word_specific_eta:
            fig.suptitle(fr'Simulated word distributions ($\alpha={alpha:.2f}$, word-specific $\eta$)')
        else:
            fig.suptitle(fr'Simulated word distributions $(\alpha={alpha:.2f}, \eta={eta:.2f})$')
        fig.tight_layout()
        plt.show()

    def get_counts(self,x,leaf_indices=None, d=None, return_total_counts=False):
        if not d:
            self.update_max_depth()
            d = self.max_depth
        if not leaf_indices:
            leaf_indices = self.leaf_indices
        # Define counts for each leaf index
        counts = {index:np.zeros(self.V) for index in leaf_indices}
        # For each sequence
        for s in range(len(x)):
            # For each subsequence x_d, x_{d+1}, ..., x_i (subsequences in x excluding first d terms)
            for i in range(d,len(x[s])):
                # Check if the subsequence corresponds to one of the given leaf indices
                for index in leaf_indices:
                    # Boolean to track if the subsequence corresponds to the path from root to leaf index
                    in_path = True
                    # Initialise traversal at current node
                    current_node = self.root
                    # Traverse the tree using the subsequence
                    for j, k in enumerate(index):
                        # Get children of current node
                        children = current_node.get_children()
                        # Check if the current element is still in the path induced by the leaf index
                        try:
                            child = children[k].get_cluster()
                        except IndexError:
                            print('index', index)
                            print('k', k)
                            print('children',current_node.get_children(as_clusters=True))

                        if x[s][i-j-1] not in children[k].get_cluster():
                            in_path = False
                            break
                        else:
                            # Move to the next child node
                            current_node = children[k]
                    if in_path:
                        # If subsequence traverses path to leaf, increment count of the next element
                        next_x = x[s][i]
                        counts[index][next_x] += 1
        
        counts_total = {index:np.sum(count_vec) for index, count_vec in counts.items()}

        if return_total_counts:
            return counts_total
        else:
            return counts

    def log_marginal_likelihood(self, x, eta, leaf_indices=None, out_prop=False):
        """Compute the log marginal likelihood of observed data for the current tree.

        Args:
            x (list or numpy array): Data sequence of integers.
            eta (float): Dirichlet prior hyperparameter for element distributions.
            leaf_indices (list, optional): List of leaf indices over which to compute the marginal likelihood.
                Defaults to None. If not specified, then the full marginal likelihood (over all leaves) is calculated.

        Returns:
            (float): Log marginal likelihood.
        """
        # Initialise
        ll = 0 
        if leaf_indices is None:
            leaf_indices=self.leaf_indices

        if out_prop:
            # Output likelihood contributions of each leaf
            leaf_ll = {}

        # Traverse tree to get counts
        counts = self.get_counts(x=x, leaf_indices=leaf_indices)

        # Calculate likelihood
        for index, count in counts.items():
            # print(count)
            temp_ll = logB(count + eta) - logB(eta * np.ones(self.V))
            
            ll += temp_ll

            if out_prop:
                leaf_ll[index] = temp_ll

        # print(counts)
        
        if out_prop:
            return ll, leaf_ll
        else:
            return ll
        
    def out_of_sample_likelihood(self, x_train, x_test, eta, leaf_indices=None, out_prop=False):
        # Initialise
        ll = 0 
        if leaf_indices is None:
            leaf_indices=self.leaf_indices

        if out_prop:
            # Output likelihood contributions of each leaf
            leaf_ll = {}

        # Traverse tree to get counts
        counts_train = self.get_counts(x=x_train, leaf_indices=leaf_indices)
        counts_test = self.get_counts(x=x_test, leaf_indices=leaf_indices)

        # Calculate likelihood
        for index in counts_train.keys():
            # print(count)
            temp_ll = logB(counts_test[index] + counts_train[index] + eta) - logB(counts_train[index] + eta)
            
            ll += temp_ll

            if out_prop:
                leaf_ll[index] = temp_ll

        # print(counts)
        
        if out_prop:
            return ll, leaf_ll
        else:
            return ll
        
    def log_loss(self, x_train, x_test, eta, leaf_indices=None, out_prop=False, base=np.e):
        """ log_loss = - 1/N_test * log_2 p(x_test|x_train)
        Logarithm is in base 2.
        """
        # Initialise
        ll = 0 
        if leaf_indices is None:
            leaf_indices=self.leaf_indices

        if out_prop:
            # Output loss contribution of each leaf
            leaf_ll = {}

        # Traverse tree to get counts
        counts_train = self.get_counts(x=x_train, leaf_indices=leaf_indices)
        counts_test = self.get_counts(x=x_test, leaf_indices=leaf_indices)

        # Get number of test sequence elements, adjusted for max depth
        S = len(x_test)
        D = self.max_depth
        N_test_adj = sum(len(x_test[s])-D for s in range(S))

        # Calculate log loss
        for index in counts_train.keys():
            # print(count)
            temp_ll = logB(counts_test[index] + counts_train[index] + eta) - logB(counts_train[index] + eta)
            
            # Change of base to log base 2
            temp_ll /= np.log(base)

            # Normalise by test size, adjusted for the first discarded D elements
            temp_ll /= (N_test_adj)


            temp_ll = -temp_ll
            ll += temp_ll

            if out_prop:
                leaf_ll[index] = temp_ll

        # print(counts)
        
        if out_prop:
            return ll, leaf_ll
        else:
            return ll
        
        
    def estimate_leaf_pmfs(self, x, eta):
        """
        Given a tree structure fitted to data,
        estimate the leaf pmfs as posterior means
        """
        from scipy.special import logsumexp
        leaf_counts = self.get_counts(x=x)
        fitted_pmfs = {}
        for leaf, counts in leaf_counts.items():
            eta_new = counts + eta
            log_pmf = np.log(eta_new) - np.log(np.sum(eta_new))
            fitted_pmfs[leaf] = np.exp(log_pmf - logsumexp(log_pmf))
            
        return fitted_pmfs

    def log_predictive_probability(self, x, leaf_pmfs=None):
        """
        Compute the joint predictive probabilities p(x|leaf_pmfs)
        where x is a list/dict of sequences and leaf_pmfs is a dict of true leaf pmfs.
        """
        if not leaf_pmfs:
            leaf_pmfs = self.leaf_pmfs
        D = self.max_depth
        
        # log probabity of sequence given true pmfs
        log_p = 0
        # For each test sequence
        for s in range(len(x)):
            # For each element starting from x_D
            for i in range(D, len(x[s])):
                # Get the context/leaf for current element
                leaf_index = self.find_leaf_index(x[s][i-D:i])
                # Update log probability with corresponding pmf and next element v
                v = x[s][i]
                log_p += np.log(leaf_pmfs[leaf_index][v])
        
        return log_p

    def empirical_log_loss(self, x_test, leaf_pmfs=None):
        """
        log_loss = - 1/N_test * log p(x_test|leaf_pmfs)
        
        Calculate the log-loss under the estimated/true leaf pmfs
        """
        if not leaf_pmfs:
            leaf_pmfs = self.leaf_pmfs
        # Get number of test sequence elements
        S_test = len(x_test)
        D=self.max_depth
        N_test_adj = sum(len(x_test[s])-D for s in range(S_test))
        D = self.max_depth

        # Calculate log-loss; adjusted for the discarded first D elements
        ll = -1/(N_test_adj) * self.log_predictive_probability(x_test, leaf_pmfs=leaf_pmfs)

        return ll


    
