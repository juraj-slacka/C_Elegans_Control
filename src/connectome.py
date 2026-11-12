"""
C. elegans Connectome Loader and Topology Builder.
"""

import os
import numpy as np
import pandas as pd


def load_raw_connectome(data_dir="data/connectome"):
    """Load edgelist and neuron classifications."""
    edgelist_path = os.path.join(data_dir, "herm_full_edgelist.csv")
    if not os.path.exists(edgelist_path):
        raise FileNotFoundError(f"Connectome edgelist not found at {edgelist_path}")
    
    df_edges = pd.read_csv(edgelist_path, skipinitialspace=True)
    df_edges['Source'] = df_edges['Source'].str.strip()
    df_edges['Target'] = df_edges['Target'].str.strip()
    return df_edges


def build_tap_withdrawal_circuit(data_dir="data/connectome"):
    """
    Builds the C. elegans Tap-Withdrawal (TW) circuit topology.
    Neurons:
      Sensory: PLM (tail touch), AVM (anterior touch), ALM (anterior touch), PVD (harsh touch)
      Interneurons: AVD, PVC, AVA, AVB, DVA
      Motor command outputs: FWD (forward locomotion), REV (backward locomotion)
    Returns:
      neuron_names: list of neuron names
      adj_matrix: (N, N) connectivity matrix with biological edge weights
      is_sensory: boolean mask for sensory neurons
      is_motor: boolean mask for motor output neurons
      edge_types: dictionary mapping (i, j) to 'chemical' or 'electrical'
    """
    df = load_raw_connectome(data_dir)
    
    # Canonical TW circuit neurons
    # Sensory: PLM, AVM, ALM, PVD
    # Interneurons: AVD, PVC, AVA, AVB, DVA
    # Command motor outputs: FWD, REV
    neurons = ['PLM', 'AVM', 'ALM', 'PVD', 'AVD', 'PVC', 'DVA', 'AVA', 'AVB', 'FWD', 'REV']
    n_neurons = len(neurons)
    name_to_idx = {name: i for i, name in enumerate(neurons)}
    
    adj_chem = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    adj_elec = np.zeros((n_neurons, n_neurons), dtype=np.float32)
    
    # Aggregate bilateral pairs from full edgelist
    tw_base_set = set(neurons[:9])
    for _, row in df.iterrows():
        s = row['Source']
        t = row['Target']
        # Map bilateral to base (e.g. PLML -> PLM)
        s_base = s.rstrip('LR') if s not in ['AVM', 'DVA'] else s
        t_base = t.rstrip('LR') if t not in ['AVM', 'DVA'] else t
        
        if s_base in tw_base_set and t_base in tw_base_set:
            i, j = name_to_idx[s_base], name_to_idx[t_base]
            w = float(row['Weight'])
            if row['Type'] == 'chemical':
                adj_chem[i, j] += w
            elif row['Type'] == 'electrical':
                adj_elec[i, j] += w
                adj_elec[j, i] += w  # Gap junctions are bidirectional
                
    # Biological command connections to motor outputs (AVA -> REV, AVB -> FWD):
    # In C. elegans neurobiology, AVA/AVD drive backward locomotion (reversal / REV)
    # AVB/PVC drive forward locomotion (FWD)
    # Cross-inhibition between forward and backward systems
    fwd_idx = name_to_idx['FWD']
    rev_idx = name_to_idx['REV']
    ava_idx = name_to_idx['AVA']
    avb_idx = name_to_idx['AVB']
    avd_idx = name_to_idx['AVD']
    pvc_idx = name_to_idx['PVC']
    
    adj_chem[avb_idx, fwd_idx] = 10.0
    adj_chem[pvc_idx, fwd_idx] = 5.0
    adj_chem[ava_idx, rev_idx] = 10.0
    adj_chem[avd_idx, rev_idx] = 5.0
    
    # Normalize adjacency weights for stable neural integration
    total_adj = adj_chem + adj_elec
    max_w = np.max(total_adj)
    if max_w > 0:
        total_adj = total_adj / max_w
        
    return {
        'neurons': neurons,
        'adj_matrix': total_adj,
        'adj_chem': adj_chem,
        'adj_elec': adj_elec,
        'sensory_indices': [name_to_idx['PLM'], name_to_idx['AVM'], name_to_idx['ALM'], name_to_idx['PVD']],
        'inter_indices': [name_to_idx[n] for n in ['AVD', 'PVC', 'DVA', 'AVA', 'AVB']],
        'motor_indices': [name_to_idx['FWD'], name_to_idx['REV']],
        'fwd_idx': fwd_idx,
        'rev_idx': rev_idx
    }


def build_full_connectome(data_dir="data/connectome", min_synapses=2):
    """
    Builds the full somatic C. elegans connectome adjacency matrix.
    Uses the 279 somatic neurons standard connectome (Varshney et al. 2011).
    """
    varshney_path = os.path.join(data_dir, "Celegans_Varshney.csv")
    if os.path.exists(varshney_path):
        df_var = pd.read_csv(varshney_path, header=None)
        # 1-indexed node IDs up to 279
        n_nodes = int(max(df_var[0].max(), df_var[1].max()))
        adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
        for _, row in df_var.iterrows():
            i, j, w = int(row[0]) - 1, int(row[1]) - 1, float(row[2])
            adj[i, j] += w
            
        adj[adj < min_synapses] = 0.0
        # Normalize
        if np.max(adj) > 0:
            adj = adj / np.max(adj)
        return {
            'n_neurons': n_nodes,
            'adj_matrix': adj,
            'density': float(np.count_nonzero(adj)) / (n_nodes * n_nodes)
        }
    else:
        raise FileNotFoundError(f"Varshney connectome not found at {varshney_path}")
