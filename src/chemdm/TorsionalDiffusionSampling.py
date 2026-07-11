import math
import numpy as np
import torch as pt

from rdkit import Chem
from rdkit.Chem import AllChem

from dataclasses import dataclass

from chemdm.geometry import kabsch_aligned_rmsd_torch, apply_torsion_update
from chemdm.TorsionalScoreNetwork import TorsionalScoreNetwork
from chemdm.MoleculeGraph import MoleculeGraph, Molecule, findAllNeighbors_gpu
from chemdm.TorsionalDiffusionData import TorsionalDiffusionData, build_torsional_data
from chemdm.TorsionalDataset import collate_torsional


N_SDE_STEPS = 20                  # reverse-SDE steps (matches sampler default)
CUTOFF = 5.0                      # spatial-graph cutoff (matches training)

# Defaults match the noise schedule in `examples/torsional_diffusuion/diffusion.py`.
SIGMA_MIN_DEFAULT = 0.01 * math.pi
SIGMA_MAX_DEFAULT = math.pi


def kabsch_aligned_heavy_rmsd( x_pred: pt.Tensor, x_ref: pt.Tensor, Z: pt.Tensor ) -> float:
    """Heavy-atom Kabsch-aligned RMSD. Float64 on CPU for SVD stability."""
    heavy = (Z != 1)
    if int(heavy.sum()) < 3:
        heavy = pt.ones_like(Z, dtype=pt.bool)
    return float( kabsch_aligned_rmsd_torch(
        x_pred[heavy].cpu().to(pt.float64),
        x_ref[heavy].cpu().to(pt.float64),
    ) )


def generate_rdkit_conformers( rdmol: Chem.Mol, n_conf: int, seed: int = 42 ) -> list[pt.Tensor]:
    """
    ETKDGv3 generation, no mmf94 relaxation. Returns a list of (N, 3) float32 tensors,
    one per successfully embedded conformer (≤ n_conf if RDKit drops any).
    """

    # Prepare the molecule
    rdmol.RemoveAllConformers()
    rdmol = Chem.AddHs( rdmol )

    # No optimization, just molecule generation
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = -1.0  # We do our own clustering.
    cids = list( AllChem.EmbedMultipleConfs(rdmol, numConfs=n_conf, params=params) )

    return [ pt.tensor( rdmol.GetConformer(cid).GetPositions(), dtype=pt.float32 ) for cid in cids ]


# Sampling
@pt.no_grad()
def sample_conformers( model: TorsionalScoreNetwork, d: TorsionalDiffusionData, starts: list[np.ndarray], device: pt.device ) -> np.ndarray:
    """Draw one conformer per starting structure. Returns (K, N, 3) float32."""
    N = int( d.Z.shape[0] )

    # K copies of the same molecule, each seeded with its own starting geometry;
    # collate_torsional applies the per-copy atom/bond offsets the sampler needs.
    examples = [ {
            "mol": MoleculeGraph( Z=d.Z, x=pt.tensor( x0, dtype=pt.float32 ), bonds=d.edge_index ),
            "rotatable_bonds": d.rotatable_bonds,
            "side_atom_idx": d.side_atom_idx,
            "side_bond_idx": d.side_bond_idx,
        }
        for x0 in starts
    ]
    batch = collate_torsional( examples )

    mol = batch["mol"].to( device=device, dtype=pt.float32 )
    x_sampled = sample_torsional_diffusion(
        model, mol,
        rotatable_bonds = batch["rotatable_bonds"].to( device ),
        side_atom_idx = batch["side_atom_idx"].to( device ),
        side_bond_idx = batch["side_bond_idx"].to( device ),
        bond_batch = batch["bond_batch"].to( device ),
        n_steps = N_SDE_STEPS,
        cutoff = CUTOFF,
    )
    return x_sampled.detach().cpu().numpy().reshape( len( starts ), N, 3 )


@pt.no_grad()
def sample_torsional_diffusion( model: TorsionalScoreNetwork,
                                mol: Molecule,               # BatchedMoleculeGraph carrying Z, x_init, molecule_id
                                rotatable_bonds: pt.Tensor,  # (m_total, 2)   rotatable bond endpoints (global atom indices)
                                side_atom_idx: pt.Tensor,    # (P_total,)     atoms on the c-side
                                side_bond_idx: pt.Tensor,    # (P_total,)     bond index per side-atom
                                bond_batch: pt.Tensor,       # (m_total,)     molecule index per bond
                                *,
                                n_steps:  int   = 20,
                                sigma_min:  float = SIGMA_MIN_DEFAULT,
                                sigma_max: float = SIGMA_MAX_DEFAULT,
                                cutoff: float = 5.0,
                                dtype: pt.dtype = pt.float32,
    ) -> pt.Tensor:
    """
    Run the reverse SDE and return the sampled atomic positions.

    Returns
    -------
    x_sampled : (N_total, 3) float — atomic positions after K reverse steps.
                Same dtype/device as `mol.x`.
    """
    device = mol.x.device

    B = int( bond_batch.max().item() ) + 1
    m = rotatable_bonds.shape[0]

    # Uniform-on-torus initialization. Adding Uniform([0, 2π)) to whatever
    # torsion the input molecule has gives Uniform([0, 2π)) mod 2π.
    delta_tau_init = pt.rand( m, device=device, dtype=dtype ) * (2.0 * math.pi)
    x = apply_torsion_update( mol.x, rotatable_bonds, side_atom_idx, side_bond_idx, delta_tau_init )

    # Reverse SDE loop. Steps go from t = 1 down to t = 0 in dt = 1/K
    # increments. For the geometric noise schedule, g²(t) = 2 σ²(t) log(σmax/σmin).
    log_sigma_ratio = math.log( sigma_max / sigma_min )
    dt = 1.0 / n_steps

    for k in range( n_steps, 0, -1 ):
        t_scalar = k * dt
        t_per_mol = pt.full( (B,), t_scalar, device=device, dtype=dtype )

        sigma = sigma_min ** (1.0 - t_scalar) * sigma_max ** t_scalar
        g_sq = 2.0 * (sigma ** 2) * log_sigma_ratio

        # Score from the model: rebuild union neighbor graph from current x.
        mol_t = mol.copyWithNewPositions(x)
        neighbors_E2, is_bond = findAllNeighbors_gpu( mol_t, cutoff )
        neighbors = neighbors_E2.T.contiguous()
        is_bond = is_bond.to( dtype=dtype )

        score = model( mol=mol_t, t=t_per_mol,
                       neighbors=neighbors, is_bond=is_bond,
                       rotatable_bonds=rotatable_bonds, bond_batch=bond_batch )

        # Reverse SDE step. Drop the noise on the last step (k == 1) for a
        # deterministic finish — standard score-based-sampling practice.
        drift = g_sq * score * dt
        if k > 1:
            noise = math.sqrt( g_sq * dt ) * pt.randn( m, device=device, dtype=dtype )
        else:
            noise = pt.zeros( m, device=device, dtype=dtype )
        delta_tau_step = drift + noise

        # Apply the torsions for the next step.
        x = apply_torsion_update( x, rotatable_bonds, side_atom_idx, side_bond_idx, delta_tau_step )

    return x


@pt.no_grad()
def sample_conformers_from_mol( model: TorsionalScoreNetwork,
                                mol: Chem.Mol,
                                n_conformers: int,
                                *,
                                device: pt.device = pt.device( "cpu" ),
                                seed: int = 42, ) -> np.ndarray:
    """End-to-end torsional-diffusion conformer generation from an RDKit molecule.

    Pipeline: ETKDGv3 embeds `n_conformers` backbones, then the reverse-SDE sampler
    re-draws the torsions of each. This is the production "generate 3D conformers"
    entry point -- feed its output to the usual cluster -> xTB-relax -> cluster tail.

    Parameters
    ----------
    model : TorsionalScoreNetwork
        Trained torsional score network.
    mol: RDKit molecule (2D or 3D). 
        Hydrogens are added internally.
    n_conformers : int
        Number of ETKDG backbones to seed (one diffusion sample per backbone).

    Returns
    -------
    (K, N, 3) float32 positions, atom order matching `Chem.AddHs(mol)`. K <=
    n_conformers (RDKit may drop embeds). A molecule with no rotatable bonds is
    rigid for torsional diffusion, so the ETKDG backbones are returned unchanged.
    """
    mol = Chem.AddHs( mol )
    data = build_torsional_data( mol )

    starts = [ e.numpy() for e in generate_rdkit_conformers( mol, n_conformers, seed=seed ) ]
    if len( starts ) == 0:
        return np.empty( (0, mol.GetNumAtoms(), 3), dtype=np.float32 )
    if data.rotatable_bonds.shape[0] == 0:
        return np.stack( starts ).astype( np.float32 )   # rigid: no torsions to sample

    return sample_conformers( model, data, starts, device )
