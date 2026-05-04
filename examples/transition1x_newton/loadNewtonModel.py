import torch as pt
from pathlib import Path

from chemdm.MolecularEmbeddingNetwork import MolecularEmbeddingGNN
from chemdm.NewtonE3NN import NewtonE3NN

def loadNewtonModel( store_root : str, device : pt.device, dtype : pt.dtype) -> NewtonE3NN:
    newton_weights = pt.load( Path(store_root) / 'best_gnn.pth', map_location=device, weights_only=True )

    d_cutoff = 5.0
    n_rbf = 10

    # Endpoint embedding networks
    embedding_state_size = 64
    embedding_message_size = 64
    n_embedding_layers = 5
    xA_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )
    xB_embedding = MolecularEmbeddingGNN( embedding_state_size, embedding_message_size, n_embedding_layers, d_cutoff )

    # E3NN transition-path network
    irreps_node_str = "48x0e + 16x1o + 16x1e + 8x2e"
    n_refinement_steps = 7
    tp_network = NewtonE3NN(
        xA_embedding_network=xA_embedding,
        xB_embedding_network=xB_embedding,
        irreps_node_str=irreps_node_str,
        n_refinement_steps=n_refinement_steps,
        d_cutoff=d_cutoff,
        n_freq=8,
        n_rbf=n_rbf,
    )
    tp_network.load_state_dict( newton_weights )
    tp_network.to( device=device, dtype=dtype )

    return tp_network