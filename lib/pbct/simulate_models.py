from .simulate_trees import CRPTree, SimulateVBM

def simulate_pbct_model(V=10, N=1000, S=1, alpha_sim=1.0
                        , max_depth=3
                        , min_depth=1
                ,eta_sim=1.0, decay_rate=True, gamma_sim=1.0, lam_sim=0.0,
                leaf_threshold=0.001, HDP=False):
    out={}
    simulated=False
    while not simulated:
        
        # Initialise CRP tree object
        crp_tree = CRPTree(V=V, alpha=alpha_sim, max_depth=max_depth, decay_rate=decay_rate, gamma=gamma_sim)
        crp_tree.generate_tree()
        simulated=True
        
        while not crp_tree.root.get_children():
            del crp_tree
            crp_tree = CRPTree(V=V, alpha=alpha_sim, max_depth=max_depth, decay_rate=decay_rate, gamma=gamma_sim)
            crp_tree.generate_tree()
            
        crp_tree.draw_pmfs(eta=eta_sim, wildcard=True, HDP=HDP, lam=lam_sim)

        x = []
        for i in range(S):
            D=max_depth
            x_i = crp_tree.simulate_data(N=N+D)
            x.append(x_i[D:])
        
        crp_counts = crp_tree.get_counts(x, return_total_counts=True)
        for index, count in crp_counts.items():
            if index[-1] != -1:
                if count/(N*S) < leaf_threshold or crp_tree.max_depth < min_depth:
                    simulated=False
                    break
    
    out['tree'] = crp_tree
    out['leaf_counts'] = crp_counts
    out['x'] = x[0] if S==1 else x
    return out


def simulate_vbm_model(V=10, N=100, S=10, theta_sim=1.0, max_depth=3, min_depth=1,
                eta_sim=1.0, decay_theta=False, gamma_sim=1.0, lam_sim=0.0,
                leaf_threshold=0.001, HDP=False):
    out={}
    # Initialise VBMtree object
    vbm_sim_tree = SimulateVBM(V=V, theta=theta_sim, max_depth=max_depth
                                        ,decay_theta=decay_theta, gamma=gamma_sim
                                        )
    simulated=False
    while not simulated:
        simulated=True
        vbm_sim_tree.leaf_indices.clear()
        vbm_sim_tree.indices.clear()
        vbm_sim_tree.clusters.clear()
        vbm_sim_tree.generate_tree()
        while not vbm_sim_tree.root.get_children():
            vbm_sim_tree.leaf_indices.clear()
            vbm_sim_tree.indices.clear()
            vbm_sim_tree.clusters.clear()
            vbm_sim_tree.generate_tree()
            
        vbm_sim_tree.draw_pmfs(eta=eta_sim, wildcard=True, HDP=HDP, lam=lam_sim)

        x = []

        for i in range(S):
            # Simulate sequence of length N+D, discard first D elements
            D=max_depth
            x_i = vbm_sim_tree.simulate_data(N=N+D)
            x.append(x_i[D:])
        
        vbm_sim_counts = vbm_sim_tree.get_counts(x, return_total_counts=True)
        for index, count in vbm_sim_counts.items():
            if index[-1] != -1:
                if count/(N*S) < leaf_threshold or vbm_sim_tree.max_depth < min_depth:
                    simulated=False
                    break
    
    out['tree'] = vbm_sim_tree
    out['leaf_counts'] = vbm_sim_counts
    out['x'] = x[0] if S==1 else x
    return out

def simulate_model_given_tree(sim_tree,
                            N=1000, S=1,
                            eta_sim=1.0, lam_sim=0.0,
                            leaf_threshold=0.001,
                            HDP=False,
                            fixed_pmfs=False):
        
    simulated=False
    while not simulated:
        simulated=True
        
        if not fixed_pmfs:
            sim_tree.draw_pmfs(eta=eta_sim, wildcard=True, HDP=HDP, lam=lam_sim)
        else:
            sim_tree.pmfs = fixed_pmfs['pmfs']
            sim_tree.leaf_pmfs = fixed_pmfs['leaf_pmfs']
            
        x = []

        for i in range(S):
            x_i = sim_tree.simulate_data(N=N)
            x.append(x_i)
        
        leaf_counts = sim_tree.get_counts(x, return_total_counts=True)
        for index, count in leaf_counts.items():
            if index[-1] != -1:
                if count/(N*S) < leaf_threshold:
                    simulated=False
                    break
    
    out = {}
    out['leaf_counts'] = leaf_counts
    out['leaf_pmfs'] = sim_tree.leaf_pmfs
    out['pmfs'] = sim_tree.pmfs
    out['x'] = x[0] if S==1 else x
    return out