from matplotlib import pyplot as plt
import numpy as np

def plot_leaf_counts(crp_tree, x):
    # Get counts
    sum_counts = crp_tree.get_counts(x=x, return_total_counts=True)

    pos = np.arange(len(sum_counts))
    
    plt.subplots(dpi=300)
    plt.bar(pos, sum_counts.values(), align='center')
    plt.xticks(pos, sum_counts.keys(), rotation = 50)
    plt.ylabel('Count')
    plt.title('Leaf counts')
    plt.show()

def plot_simulated_word_frequencies(crp_tree, x, all_pmfs=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    
    leaf_indices = crp_tree.leaf_indices
    leaf_counts_dict = crp_tree.get_counts(x=x, leaf_indices=leaf_indices)
    
    d_max = max(len(e) for e in leaf_indices)
    all_indices = crp_tree.indices
    
    all_counts_dict = crp_tree.get_counts(x=x,leaf_indices=all_indices)
    
    if all_pmfs:
        leaf_counts_dict = all_counts_dict
        
    n_params = len(leaf_counts_dict)
    size = int(np.sqrt(n_params))+1
    fig, axs = plt.subplots(nrows=size, ncols=size,
                            sharey=True, sharex=False, figsize=(6,6), dpi=300)

    N_max = np.max([np.max(counts) for counts in leaf_counts_dict.values()])

    for n, (e, phi) in enumerate(leaf_counts_dict.items()):
        axs.flat[n].plot(phi/sum(phi) if sum(phi)>0 else np.zeros_like(phi))
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
        
    fig.suptitle(fr'Simulated word frequency densities')
    fig.tight_layout()
    plt.show()

def plot_pmfs(pmfs, title='Leaf pmfs', init_pmfs=False):
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'

    if not init_pmfs:
        drop_list = []
        for index in pmfs.keys():
            if len(index)==0 or index[-1] == -1:
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
    
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()