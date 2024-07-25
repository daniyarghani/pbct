import numpy as np
from scipy.special import loggamma
from sklearn.model_selection import train_test_split

def labels_to_clusters(labels):
    """Convert an array of cluster labels to dictionary format
    Example: [1,1,2,0,0,0] -> {0: [3,4,5], 1: [0,1], 2: [2]}

    Args:
        labels (list): List of cluster labels.
    
    Returns:
        clusters (dict): Dictionary of clusters in {label: cluster} form.
    """
    clusters = {}
    for v, k in enumerate(labels):
        if k not in clusters:
            clusters[k] = []
        clusters[k].append(v)
    
    return clusters

def clusters_to_labels(clusters,V):
    """Convert dictionary clusters to array label format
    Example: {0: [3,4,5], 1: [0,1], 2: [2]} -> [1,1,2,0,0,0]  

    Args:
        clusters (dict): Dictionary of clusters in {label: cluster} form.
    Returns:
        labels (list): List of cluster labels.
    """  
    labels = np.zeros(V, dtype=int)
    for k, cluster in clusters.items():
        for v in cluster:
            labels[v] = k

    return labels


# Computes logarithm of the multivariate beta function
def logB(vec):
    out = np.sum([loggamma(v) for v in vec])
    out -= loggamma(np.sum(vec))
    return out

def validate_input_x(x, V):
    # Check type of input data x
    if isinstance(x, (dict,list,np.ndarray)):
        if all([isinstance(x[i], (list, np.ndarray)) for i in range(len(x))]):
            x_is_single_sequence = False
        elif all([isinstance(x[i], (int,np.int64)) for i in range(len(x))]):
            x_is_single_sequence = True
        else:
            raise TypeError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1, or a single list of integers from 0 to V-1.')
        
        if x_is_single_sequence:
            if not all([i in range(V) for i in x]):
                raise ValueError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1, or a single list of integers from 0 to V-1.')
        else:
            if not all([i in range(V) for s in range(len(x)) for i in x[s]]):
                raise ValueError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1, or a single list of integers from 0 to V-1.')
            
    else:
        raise TypeError('Input (x) must be a dictionary or list containing lists or numpy arrays of integers from 0 to V-1.')
    
    return x_is_single_sequence

def _train_test_split(x, V, train_size=None, test_size=None, return_statistics=False):
    # Check if data is a single sequence or a dict/list of sequences
    
    x_is_single_sequence = validate_input_x(x, V)
    
    if x_is_single_sequence:
        # Sequence length
        N = N_total = len(x)
        S = 1
        S_train = 1
        S_test = 1
        if train_size:
            N_train = round(N_total*train_size)
        elif test_size:
            N_train = round(N_total*(1-test_size))
        else:
            raise ValueError('train or test size must be specified between 0 and 1')
        
        # Simple train/test split of sequence
        x_train, x_test = [x[:N_train]], [x[N_train:]]
    
    else:
        # sklearn train/test split into collections of sequences
        x_train, x_test = train_test_split(x
                                           ,train_size=train_size
                                           ,test_size=test_size
                                           ,shuffle=True
                                           ,random_state=42
                                          )
        
        S = len(x)
        N = round(np.mean([len(x[s]) for s in range(S)]))
        
        N_total = sum([len(x[s]) for s in range(S)])
        # assert N_total == S*N
        
        S_train = len(x_train)
        S_test = S-S_train
        N_train = sum([len(x_train[s]) for s in range(S_train)])
    
    N_test = N_total - N_train
        
        
    if return_statistics:
        stats = {}
        stats['S'] = S
        stats['N'] = N
        stats['S_train'] = S_train
        stats['S_test'] = S_test
        stats['N_total'] = N_total
        stats['N_train'] = N_train
        stats['N_test'] = N_test
        
        return x_train, x_test, stats
    else:
        return x_train, x_test
