import torch as pt
from pathlib import Path

from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.ResidualDiffusionE3NN import ResidualDiffusionE3NN

def loadDiffusionModel( store_root : str, 
                        device : pt.device, 
                        dtype : pt.dtype) -> ResidualDiffusionE3NN:
    residual_scale = 0.15
    
    d_cutoff = 5.0
    xA_emb = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)
    xB_emb = MolecularEmbeddingGNN(64, 64, 5, d_cutoff)
    diffusion_model = ResidualDiffusionE3NN(
        xA_emb,
        xB_emb,
        "48x0e + 16x1o + 16x1e + 8x2e",
        3,
        d_cutoff,
        n_arclength_freq=8,
        n_rbf=10,
        residual_scale=residual_scale,
    ).to(device=device, dtype=dtype)
    diffusion_model.load_state_dict(pt.load(Path(store_root) / 'best_diffusion_gnn.pth', map_location=device, weights_only=True))
    
    return diffusion_model