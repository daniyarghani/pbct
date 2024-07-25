import numpy as np
from .tree_objects import Node, Tree
from .utils import logB, validate_input_x
from scipy.special import loggamma, logsumexp
from collections import namedtuple, defaultdict, Counter
import matplotlib.pyplot as plt
from copy import deepcopy

class PBCT(Tree):
    """
    Class to estimate a PBCT given an observed sequence using agglomerative clustering and/or MCMC.
    """
    def __init__(self, V=0, max_depth=1, eta=1.0, alpha=1.0, 
                 decay_rate=True, gamma=1.0,
                 crp_prior=True
                 ,normalize_similarities=False,
                 penalize_prior=False,
                 beta=0.0,
                 agg_clust=True,
                 merge_empty=False,
                 MCMC=False,
                 n_its=100,
                 n_burn=0,
                 size=1,
                 anneal = False,
                 cooling_size = 10,
                 cooling_rate=0.5,
                 init_temperature = 1.0,
                 out_MCMC_likelihood=True,
                 hill_climbing=False,
                 mcmc_solution='last',
                 show_dendrograms=False,
                 true_tree=None,
                 verbose_MCMC=False
                 ):

        super().__init__(V=V, max_depth=max_depth)

        if not isinstance(eta, (float, int, np.int64)):
            raise TypeError('Hyperparameter (eta) must be a positive number.')
        if not eta>0:
            raise ValueError('Hyperparameter (eta) must be a positive number.')

        if not isinstance(alpha, (float, int, np.int64)):
            raise TypeError('Hyperparameter (alpha) must be a positive number.')
        if not alpha>0:
            raise ValueError('Hyperparameter (alpha) must be a positive number.')

        self.eta=eta
        self.alpha=alpha
        self.counts={}
        self.crp_prior = crp_prior
        self.decay_rate = decay_rate
        self.gamma=gamma
        self.beta=beta
        self.penalize_prior = penalize_prior
        self.normalize_similarities = normalize_similarities

        self.agg_clust = agg_clust
        self.merge_empty = merge_empty
        self.MCMC=MCMC

        # Cannot have both agg_clust and MCMC False
        if not(self.agg_clust) and not(self.MCMC):
            raise ValueError('Either agg_clust or MCMC must be True')

        if self.MCMC:
            # Cluster allocations of vocabulary elements
            self.z = np.zeros(self.V, dtype=int)
            # MCMC parameters
            self.n_its=n_its
            self.n_burn=n_burn
            self.size=size
            self.anneal= anneal # Boolean for simulated annealing
            self.init_temperature = init_temperature # Temperature for simulated annealing
            self.cooling_size = cooling_size # Number of MCMC iterations before cooling temperature
            self.cooling_rate = cooling_rate # Parameter between 0 and 1 to control the geometric cooling (decay) of temperature T_i = T_0 * rate^i
            self.out_MCMC_likelihood=out_MCMC_likelihood
            self.hill_climbing=hill_climbing
            self.verbose_MCMC = verbose_MCMC
            self.mcmc_solution = mcmc_solution
        
        self.true_tree=true_tree
        self.show_dendrograms = show_dendrograms
        

    def init_MCMC(self, current_node=None):
        # Initialise cluster labels
        for k, cluster in enumerate(current_node.get_children(as_clusters=True)):
            for v in cluster:
                self.z[v] = k

        # Count of cluster sizes
        Z_counter = Counter(self.z)
        # Convert counts to numpy array
        self.Z = np.zeros(len(Z_counter), dtype=int)
        for k, count in Z_counter.items():
            self.Z[k] = count

        # Number of child clusters
        self.K = len(self.Z)

    
    def resample_clusters(self, current_node, indices=None, temperature=1.0):
        
        # List of indices (vocabulary elements) to resample
        if not indices:
            indices = np.random.choice(self.V, size=self.size)

        # Set rate parameter
        e = current_node.get_index()
        d=len(e)
        # Decay CRP rate parameter alpha with depth if specified
        alpha = self.alpha*np.exp(self.gamma*-d) if self.decay_rate else self.alpha

        # For each vocabulary element, resample its cluster allocation
        for v in indices:
            # Get children of current node
            child_nodes = deepcopy(current_node.get_children())
            child_nodes_old = deepcopy(child_nodes)
            # Cluster allocation z of element v
            z_old = deepcopy(self.z)

            Z_old = deepcopy(self.Z)

            zv = self.z[v]

            # Old likelihood
            leaf_indices_old=[e+(k,) for k in range(len(np.unique(z_old)))]
            ll_old=self.log_marginal_likelihood(x=self.x, eta=self.eta,leaf_indices=leaf_indices_old)

            if self.verbose_MCMC:
                print('node children old', current_node.get_children(as_clusters=True))
                print('z old', z_old)

            # Remove element v from child node z
            child_nodes[zv].cluster.remove(v)

            # Decrement cluster size excl. element v
            self.Z[zv] -= 1 

            # If removing element v gives empty cluster, delete the cluster
            del_z = (self.Z[zv] == 0)
            if del_z:
                # Decrement number of clusters
                self.K -= 1

                # Delete empty child node
                del child_nodes[zv]

                # Delete empty cluster from counts array
                self.Z = np.delete(self.Z, zv)
                
                # Decrement child cluster labels
                for node in child_nodes:
                    idx = node.get_index()
                    k = idx[-1]
                    if k >= zv:
                        idx_new = idx[:-1] + (k-1,)
                        node.set_index(idx_new)

                # Decrement cluster labels greater than deleted cluster
                self.z[self.z >= zv] -= 1

            # Add new child node corresponding to singleton {v}
            node_v = Node(index=e+(self.K,), cluster=[v])
            child_nodes.append(node_v)

            # Add new cluster (K+1) to counts array with pseudo-count alpha
            self.Z = np.append(self.Z, alpha)
            # print('Z', self.Z)
            
            # Get child indices
            child_indices = [node.get_index() for node in child_nodes]

            # Update tree with pseudo-nodes
            current_node.set_children(children=child_nodes)

            # Compute pseudo-counts X
            X_counts = self.get_counts(x=self.x, d=d+1, leaf_indices=child_indices)
            
            # Convert X counts to array
            X = np.zeros((self.K+1, self.V), dtype=int)
            for idx, counts in X_counts.items():
                k = idx[-1]
                X[k] = counts

            # Calculate log probabilities for cluster allocations (full conditionals)
            # Prior factor p(z)=alpha if cluster=K+1 else p(z)=count
            p = np.log(self.Z)

            # print('log p(z)', p)
            if self.verbose_MCMC:
                print('v', v)
                print('z_v old', zv)
                print('K -v', self.K)
                print('Z -v', self.Z)
                print('init log(p)', p)

            # Likelihood factor p(x|z) = marginal likelihood of data restricted to children of node e
            for h in range(self.K+1):
                if h < self.K:
                    p[h] += np.sum([logB(self.eta+X[k]+X[-1]) if h==k else logB(self.eta+X[k]) for k in range(self.K)])
                else:
                    p[h] += np.sum([logB(self.eta+X[k]) for k in range(self.K+1)])

                p[h] -= logB(self.eta*np.ones(self.V))
                
            p = p/temperature

            # Renormalise probabilities
            p = np.exp(p-logsumexp(p))

            # Resample cluster label
            if self.hill_climbing:
                zv_new = np.argmax(p)
            else:
                zv_new = np.random.choice(len(p), p=p)

            # Propose move
            self.z[v] = zv_new

            # Update cluster sizes
            self.Z[zv_new] += 1

            # Update tree:
            # If new cluster is a singleton (zv_new=K+1), child_nodes remains unchanged
            # Else, remove the singleton cluster v from child_nodes and append v to cluster zv_new
            if zv_new == self.K:
                self.K += 1
            else:
                self.Z = np.delete(self.Z, -1)

                current_node.children.pop()
                current_node.children[zv_new].cluster.append(v)

                if self.verbose_MCMC:
                    print('node children proposal', current_node.get_children(as_clusters=True))
            
            # New likelihood
            leaf_indices_new=[e+(k,) for k in range(len(np.unique(self.z)))]
            ll_new=self.log_marginal_likelihood(x=self.x,eta=self.eta,leaf_indices=leaf_indices_new)
            ll_change = ll_new-ll_old
            if self.verbose_MCMC:
                print('Change in likelihood', ll_new-ll_old)
            
            # Revert changes if likelihood decreases
            if self.hill_climbing:
                if ll_change<0:
                    # Revert cluster allocation
                    self.z = z_old
                    # Revert cluster sizes
                    self.Z = Z_old
                    # Revert number of clusters
                    self.K = len(self.Z)
                    # Revert changes to tree
                    current_node.set_children(child_nodes_old)
                    if self.verbose_MCMC:
                        print('Reverting changes')
                        print('node children new', current_node.get_children(as_clusters=True))


            # For debugging
            if self.verbose_MCMC:
                print('parent index', e)
                print('log p', np.log(p))
                print('z_v new', zv_new)
                print('z new', self.z)
    
    def MCMC_step(self, current_node):
        
        out = {}
        if self.out_MCMC_likelihood:
            out['likelihood'] = np.zeros(self.n_its+self.n_burn+1)
        # Initialise counts and cluster labels
        self.init_MCMC(current_node=current_node)
        # z_vals = np.zeros((self.n_its,self.V),dtype=int)
        e = current_node.get_index()

        # Get initial likelihood
        leaf_indices=[e+(k,) for k in range(len(np.unique(self.z)))]
        if self.out_MCMC_likelihood:
            out['likelihood'][0] += self.log_marginal_likelihood(x=self.x, eta=self.eta, 
                                                                 leaf_indices=leaf_indices)
        
        if self.anneal:
            temperature = self.init_temperature
        else:
            temperature = 1.0

        # Iterate MCMC sampler
        for i in range(self.n_its+self.n_burn):
            if self.verbose_MCMC:
                print('iteration', i)
                print('-'*50)
            # Resample cluster allocations for vocabulary elements
            self.resample_clusters(current_node=current_node, temperature=temperature)

            # Compute log marginal likelihood if specified
            if self.out_MCMC_likelihood:
                leaf_indices=[e+(k,) for k in range(np.max(self.z)+1)]
                out['likelihood'][i+1] += self.log_marginal_likelihood(x=self.x, eta=self.eta, 
                                                                 leaf_indices=leaf_indices)
                
            if i%self.cooling_size==0:
                temperature *= self.cooling_rate
            
            if self.verbose_MCMC:
                print('='*50)
                
        
        # Get optimal children
        optimal_children = deepcopy(current_node.get_children())
        current_node.remove_children()
        
        out['optimal_children'] = optimal_children
        # print('MCMC step done')
        return out


    def cluster_similarity(self, index_i, index_j):
        count_i = self.counts[index_i]
        count_j = self.counts[index_j]
        # Compute log ratio of marginal likelihoods of clusters i and j
        similarity = logB(count_i + count_j + self.eta) + logB(self.eta * np.ones(self.V))
        similarity -= (logB(count_i + self.eta) + logB(count_j + self.eta))

        # CRP cluster prior
        if self.crp_prior:
            # Get number of clusters in configuration (to identify which clustering is being considered)
            K = len(self.counts)
            # Get depth of clusters
            assert len(index_i)==len(index_j)
            d = len(index_i)-1
            # Decay CRP rate parameter alpha with depth if specified
            alpha = self.alpha*np.exp(self.gamma*-d) if self.decay_rate else self.alpha
            # Clusters to propose merge
            i,j=index_i[-1],index_j[-1]
            # Sizes of clusters
            N_i=len(self.clusterings[K][i])
            N_j=len(self.clusterings[K][j])
            # Log CRP prior
            similarity += loggamma(N_i+N_j)
            similarity -= (np.log(alpha) + loggamma(N_i) + loggamma(N_j))
        
        if self.normalize_similarities:
            # similarity -= np.log(N_i+N_j)
            similarity -= self.beta*np.log(N_i+N_j+1)

        return similarity
    
    def objective(self,current_node):
        score = 0
        child_indices = [child_node.get_index() for child_node in current_node.get_children()]
        child_clusters = current_node.get_children(as_clusters=True)
        K = len(child_indices)
        # Get marginal likelihood of child nodes
        score += self.log_marginal_likelihood(x=self.x, eta=self.eta, leaf_indices=child_indices)
        # Adjust for CRP cluster prior
        if self.crp_prior:
            # Get depth of clusters
            d = len(child_indices[0])-1
            # Decay CRP rate parameter alpha with depth if specified
            alpha = self.alpha*np.exp(self.gamma*-d) if self.decay_rate else self.alpha
            score += K*np.log(alpha) + loggamma(alpha) + np.sum([loggamma(len(clust)) for clust in child_clusters])
            score -= loggamma(alpha+self.V)
        
        # Penalize large number of clusters
        if self.penalize_prior:
            score -= self.beta*K
            # score -= np.log(1+self.beta*K)

        return score

    def agglomerative_clustering(self, current_node):
        index = current_node.get_index() # Current index
        d=len(index)
        self.clusterings = {} # Store clusterings for each number of clusters
        K = self.V # Number of clusters in configuration: initial value is size of vocabulary

        # Initialise child clusters: each element of V in its own cluster
        self.clusterings[K] = []
        for v in range(self.V):
            # Add child node to tree
            child_index = index + (v,)
            child_node = Node(index=child_index, cluster=[v])
            current_node.add_child(child_node)
            # Store current clustering
            self.clusterings[K].append([v])

        child_indices = [child_node.get_index() for child_node in current_node.get_children()]

        # Initialise scores (marginal posterior/likelihood of cluster configurations)
        scores = np.zeros(K)
        scores[K-1] = self.objective(current_node=current_node)

        # Initialise matrix of similarities between clusters
        similarities = np.zeros((self.V,self.V))
        np.fill_diagonal(similarities, -np.inf)

        # Get counts X_{e,v} for each child
        self.counts = self.get_counts(x=self.x, d=d+1, leaf_indices=child_indices)

        for i in range(K-1):
            for j in range(i+1,K):
                # Compute similarity of clusters i and j (children of current node)
                index_i, index_j = index + (i,), index + (j,)
                similarities[i,j] = self.cluster_similarity(index_i, index_j)
                similarities[j,i] = similarities[i,j]
        
        sizes_i = []
        sizes_j = []
        # Merge best pairs of clusters until all are merged into one cluster
        while K>1:
            
            # Identify clusters with highest similarity
            i_hat, j_hat = np.unravel_index(np.argmax(similarities, axis=None), similarities.shape)
            # Merge clusters
            merged_cluster = self.clusterings[K][i_hat] + self.clusterings[K][j_hat]
            merged_cluster = merged_cluster.copy()
            sizes_i.append(len(self.clusterings[K][i_hat]))
            sizes_j.append(len(self.clusterings[K][j_hat]))

            # Update children of current node
            children = current_node.get_children()
            child_i_hat = children[i_hat]
            child_i_hat.set_cluster(merged_cluster) # Update child i_hat to merged cluster

            # Relabel child nodes
            for j in range(j_hat+1,K):
                child_j = children[j]
                child_j.set_index(index+(j-1,)) # Decrement the index of child j

            # Delete old child j_hat
            child_to_remove = children[j_hat]
            current_node.remove_children(child_to_remove) # Delete child j_hat
            
            # Update clusterings
            K -= 1
            assert(len(current_node.get_children(as_clusters=True))==K)
            child_clusters = current_node.get_children(as_clusters=True)
            self.clusterings[K] = child_clusters

            # Compute marginal likelihood of new clustering
            child_indices = [child_node.get_index() for child_node in current_node.get_children()]
            scores[K-1] = self.objective(current_node=current_node)
            
            # Update counts 
            self.counts = self.get_counts(x=self.x, d=d+1, leaf_indices=child_indices)

            # Recalculate similarities
            similarities = np.delete(np.delete(similarities,j_hat,axis=0),j_hat,axis=1)
            for j in range(K):
                if j != i_hat:
                    # Compute similarity of merged cluster i_hat with other cluster j
                    index_i_hat, index_j = index + (i_hat,), index + (j,)
                    similarities[i_hat, j] = self.cluster_similarity(index_i_hat, index_j)
                    similarities[i_hat, j] = similarities[j, i_hat]
            
        # Get optimal clustering
        optimal_K = np.argmax(scores)+1
        optimal_clustering = self.clusterings[optimal_K]

        # pprint(self.clusterings)
        if self.show_dendrograms:
            self.plot_dendrogram(clusterings=self.clusterings,
                                depth=len(index)+1,
                                parent=index,
                                optimal_K=optimal_K)
            print('scores =',scores)


        # Get optimal clusters/children
        optimal_children = []
        for k,cluster in enumerate(optimal_clustering):
            child_index = index+(k,)
            child_node = Node(index=child_index, cluster=cluster)
            optimal_children.append(child_node)

        # Merge the zero-count (empty) clusters
        if self.merge_empty:
            # Get the indices of the zero-count clusters
            # optimal_indices = [index+(k,) for k in range(len(optimal_clustering))]
            child_indices = [child_node.get_index() for child_node in current_node.get_children()]
            counts_optimal = self.get_counts(x=self.x, d=d+1, leaf_indices=child_indices, return_total_counts=True)
            counts_zero = {idx: count for idx, count in counts_optimal.items() if count == 0}

            if counts_zero:
                # Merge the zero-count clusters
                cluster_merged = []
                index_merged = list(counts_zero.keys())[0]
                for index in counts_zero:
                    k = index[-1]
                    cluster = optimal_clustering[k]
                    cluster_merged.extend(cluster)
                
                # Update optimal clustering
                k_merged = index_merged[-1]
                optimal_clustering[k_merged] = cluster_merged
                indices_del = list(range(k_merged+1,len(optimal_clustering)))
                for k_del in sorted(indices_del, reverse=True):
                    del optimal_clustering[k_del]

                current_node.remove_children()
                # Get optimal clusters/children
                optimal_children = []
                for k,cluster in enumerate(optimal_clustering):
                    child_index = index+(k,)
                    child_node = Node(index=child_index, cluster=cluster)
                    optimal_children.append(child_node)

        # Remove children of current node; the current node's children should be set in recursive_clustering method
        current_node.remove_children()
        
        return optimal_children
    
    def recursive_clustering(self, current_node):
        """For a current node in the tree, estimate its children using recursive agglomerative clustering.

        Args:
            current_node (Node object): Current node in tree as a Node object.
        """
        
        # Get depth of current node
        e = current_node.get_index()

        current_depth = len(e)
        # If maximum depth not yet reached, attempt to generate children of current node
        if current_depth < self.max_depth:
            # Estimate children using agglomerative clustering and return the optimal children
            if self.agg_clust:
                optimal_children = self.agglomerative_clustering(current_node=current_node)
            else:
                optimal_children = [Node(index=e+(v,), cluster=[v]) for v in range(self.V)]

            # Perform MCMC
            if self.MCMC:
                # Set initial child nodes
                current_node.set_children(optimal_children)
                # print(current_node)
                # print(current_node.children)
                # Perform MCMC to get updated optimal children
                out_mcmc = self.MCMC_step(current_node=current_node)
                optimal_children = out_mcmc['optimal_children']

                if self.out_MCMC_likelihood:
                    MCMC_likelihood = out_mcmc['likelihood']
                    plt.plot(range(self.n_its+self.n_burn+1),MCMC_likelihood, c='b')
                    plt.ylabel('log likelihood')
                    plt.xlabel('Iteration')
                    plt.title(f'Marginal likelihood, parent node {e}')
                    # plt.xticks(range(1, self.n_its+self.n_burn+1))
                    # plt.grid(True,alpha=0.5)
                    plt.show()

            # If the partition is non-trivial, i.e. more than one cluster is generated
            if len(optimal_children) > 1:
                # Set optimal children as children of current node
                current_node.set_children(optimal_children)

                # For each child node, repeat the clustering scheme
                for child_node in current_node.get_children():
                    # Add index to list of indices
                    child_index = child_node.get_index()
                    self.indices.append(child_index)
                    # Add clusters to tree
                    self.clusters[child_index] = child_node.get_cluster()
                    self.recursive_clustering(current_node=child_node)
            else:
                # Set index of current node as a leaf index
                self.set_leaf_index(current_node.get_index())
                # self.indices.append(current_node.get_index())
        else:
            # Set index of current node as a leaf index
            self.set_leaf_index(current_node.get_index())

        
    def plot_dendrogram(self,clusterings,depth=None,parent=None,optimal_K=None):
        colour='k'
        # Get final merged cluster to use as the x ticks
        ordered_V = clusterings[1][0]
        mapping = {v:i for i, v in enumerate(ordered_V)}
        
        # Number of cluster configurations = vocabulary length
        K = len(clusterings)
        assert K==self.V

        fig, ax = plt.subplots(figsize=(10,8))
        fig.canvas.draw()

        # Get "old" clustering to compare with the next clustering
        old_clustering = clusterings[K]
        # Store the step for each merge
        merge_step_map = defaultdict(int)
        # For each merge step
        for h in range(1,K):
            # Get next clustering after merge
            new_clustering = clusterings[K-h]
            # Identify the merged clusters
            merged_clusters = [tuple(clust) for clust in old_clustering if clust not in new_clustering]
            # print(merged_clusters)
            assert len(merged_clusters)==2
            # Get midpoints of clusters
            clust_1,clust_2 = merged_clusters[0],merged_clusters[1]
            x_1 = sum([mapping[v]for v in clust_1])/len(clust_1)
            x_2 = sum([mapping[v]for v in clust_2])/len(clust_2)
            # Draw horizontal line
            ax.hlines(y=h-0.5, xmin=x_1, xmax=x_2,colors=colour)
            # Draw vertical lines
            ymin_1 = merge_step_map[clust_1]
            ymin_2 = merge_step_map[clust_2]
            ax.vlines(x=x_1,ymin=ymin_1-0.5,ymax=h-0.5,colors=colour) # left line
            ax.vlines(x=x_2,ymin=ymin_2-0.5,ymax=h-0.5,colors=colour) # right line
            # Update clustering
            old_clustering = new_clustering
            # Store merge step
            merge_step_map[clust_1+clust_2]=h
            # print(merge_step_map)
        
        ax.vlines(x=(K-1)/2,ymin=K-1.5,ymax=K-0.5,colors=colour) # root line
        ax.hlines(y=self.V-optimal_K,xmin=0,xmax=self.V-1,colors='r', linestyle='--') # optimal K

        ax.set_xticks(range(K))
        ax.set_xticklabels([str(v) for v in ordered_V])
        ax.set_yticks(range(self.V))
        ax.set_yticklabels([str(v) for v in range(self.V, 0, -1)])
        ax.set_xlabel('v')
        ax.set_ylabel('K')
        title = f'Agglomerative clustering, depth = {depth}, parent = {parent}'
        ax.set_title(title)
        plt.show()

    def build_tree(self):
        # Initialise root node
        self.root = Node()
        # Clear indices and clusters
        self.leaf_indices.clear()
        self.indices.clear()
        self.clusters.clear()
        # Estimate tree structure using recursive agglomerative clustering
        self.recursive_clustering(self.root)
        # Start from initial tree structure, if given
    
    def fit(self, x):
        # Method to build a Markov tree
        # Validate type of input data x
        x_is_single_sequence = validate_input_x(x, self.V)
        if x_is_single_sequence:
            x=[x]
        
        # Self instantiate data
        self.x=x

        # Estimate tree structure
        self.build_tree()

        self.update_max_depth()
        # print(self)
    
    def __str__(self):
        out = namedtuple('PCTree', ['V','alpha','max_depth', 'n_leaves'])
        return f'{out(self.V, self.alpha, self.max_depth, len(self.leaf_indices))}'

class VBM(Tree):
    """
    Fit a Variable-order Bayesian Markov tree given an observed sequence.
    """
    def __init__(self, V=0, max_depth=1, eta=1.0, theta=1.0 
                 ,decay_theta=True
                 ,gamma=1.0
                 ,bernoulli_prior=True
                 ):

        super().__init__(V=V, max_depth=max_depth)

        if not isinstance(eta, (float, int, np.int64)):
            raise TypeError('Hyperparameter (eta) must be a positive number.')
        if not eta>0:
            raise ValueError('Hyperparameter (eta) must be a positive number.')

        if not isinstance(theta, (float, int, np.int64)):
            raise TypeError('Hyperparameter (theta) must be a positive number.')
        if not theta>0:
            raise ValueError('Hyperparameter (theta) must be a positive number.')

        self.eta=eta
        self.theta=theta
        self.counts={}
        self.bernoulli_prior = bernoulli_prior
        self.decay_theta = decay_theta
        self.gamma=gamma

    def calculate_stop_split_scores(self, current_node):
        # Initialise scores (marginal posteriors) for stop (0) vs split (1)
        scores = np.zeros(2)
        # Index of current node
        e = current_node.get_index()
        # Indices of V split child nodes
        
        # Temporarily add split children to node for likelihood calculation
        split_indices = [e+(v,) for v in range(self.V)]
        split_children = [Node(index=e+(v,), cluster=[v]) for v in range(self.V)]
        current_node.set_children(split_children)

        # Get log marginal likelihoods of stop (0) and split (1)
        scores[0] += self.log_marginal_likelihood(x=self.x, eta=self.eta, leaf_indices=[e]) # Assumes current node is a leaf
        scores[1] += self.log_marginal_likelihood(x=self.x, eta=self.eta, leaf_indices=split_indices) # Assumes V leaf children of current node
            
        # Adjust for Bernoulli prior
        if self.bernoulli_prior:
            # Get depth of clusters
            d = len(e)
            # Decay CRP rate parameter theta with depth if specified
            theta = self.theta*np.exp(self.gamma*-d) if self.decay_theta else self.theta

            scores[0] += np.log(1-theta) # Stop w.p. 1-theta
            scores[1] += np.log(theta) # Split w.p. theta

        # Delete the split children of current node
        current_node.remove_children()
        
        return scores
    
    def recursive_stop_split(self, current_node):
        """
        """

        # Index of current node
        e = current_node.get_index()
        d = len(e)
        # If maximum depth not yet reached, decide split or stop move
        if d < self.max_depth:
            # Get stop or split scores
            stop_split_scores = self.calculate_stop_split_scores(current_node)
            # If move is split, then split=True, else split=False
            split = bool(np.argmax(stop_split_scores))
            
            # Split and continue recursively (split=1) or stop (split=0)
            if split:
                # Set split children of current node
                current_node.set_children([Node(index=e+(v,), cluster=[v]) for v in range(self.V)])
                # For each child node, repeat the clustering scheme
                for child_node in current_node.get_children():
                    # Add index to list of indices
                    child_index = child_node.get_index()
                    self.indices.append(child_index)
                    # Add clusters
                    self.clusters[child_index] = child_node.get_cluster()
                    # Recursively continue
                    self.recursive_stop_split(current_node=child_node)
            else:
                # Set index of current node as a leaf index
                self.set_leaf_index(current_node.get_index())
                # self.indices.append(current_node.get_index())
        else:
            # Set index of current node as a leaf index
            self.set_leaf_index(current_node.get_index())

    def build_tree(self):
        # Initialise root node
        self.root = Node()
        # Clear indices and clusters
        self.leaf_indices.clear()
        self.indices.clear()
        self.clusters.clear()
        # Estimate tree structure
        self.recursive_stop_split(self.root)
        # Start from initial tree structure, if given
    
    def fit(self, x):
        # Method to build a Markov tree
        if isinstance(x, dict):
            if not all([isinstance(x[s], (list,np.ndarray)) for s in x]):
                raise TypeError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1.')
            if not all([i in range(self.V) for s in x for i in x[s]]):
                raise ValueError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1.')
            
        elif isinstance(x,(list,np.ndarray)):
            if not all([isinstance(s, (list,np.ndarray)) for s in x]):
                raise TypeError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1.')
            if not all([i in range(self.V) for s in x for i in s]):
                raise ValueError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1.')
        else:
            raise TypeError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1.')
        
        
        # Self instantiate data
        self.x=x

        # Estimate tree structure
        self.build_tree()

        self.update_max_depth()

    
    def __str__(self):
        out = namedtuple('VBMTree', ['V','theta','max_depth','n_leaves'])
        return f'{out(self.V, self.theta, self.max_depth, len(self.leaf_indices))}'
