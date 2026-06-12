"""
Worker for `evaluate.py`. Processes exactly one chunk of test molecules and
writes its RMSD matrices to `./eval_chunks/<mode>_chunk_<idx>.pt`. Spawned
as a fresh subprocess per chunk by the orchestrator so the MPSGraph cache
is released when the process exits.

`mode` is "rdkit" (paper inference protocol — start each sample from a fresh
ETKDGv3 embed) or "crest" (start from the molecule's ground-truth conformer
to isolate the score net from the RDKit-vs-CREST distributional shift).

Run
---
    python eval_worker.py <chunk_idx> <mode>
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch as pt
from dotenv import load_dotenv

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from chemdm.MoleculeGraph import MoleculeGraph, batchMolecules
from chemdm.geometry import kabsch_align_torch

from score_network import TorsionalScoreNetwork
from sampler import sample_torsional_diffusion

RDLogger.DisableLog("rdApp.*")


# Config — N_SAMPLES_PER_MOL, N_STEPS, CUTOFF, DTYPE are worker-only.
# B_MOL must match the value in evaluate.py.
N_SAMPLES_PER_MOL = 10
N_STEPS = 20
B_MOL = 8
CUTOFF = 5.0
DTYPE = pt.float32


def rdkit_starting_geometry( qm9_dir: Path, smiles: str, dtype: pt.dtype = DTYPE, seed: int = 42 ) -> pt.Tensor | None:
    """
    Generate a starting geometry L̂ for one test molecule by re-loading its
    QM9 pickle, taking the stored rd_mol, and running ETKDGv3. Returns None
    if RDKit's embed step fails. Matches the paper's inference protocol —
    only the SMILES is assumed available at test time.
    """
    pickle_path = qm9_dir / f"{smiles}.pickle"
    with open( pickle_path, "rb" ) as f:
        raw = pickle.load(f)

    rdmol = Chem.Mol( raw["conformers"][0]["rd_mol"] )
    rdmol.RemoveAllConformers()

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule( rdmol, params ) < 0:
        return None

    pos = rdmol.GetConformer(0).GetPositions()
    return pt.tensor( pos, dtype=dtype )


def crest_starting_geometry( conf_idx: int, x_flat: pt.Tensor, x_offset: pt.Tensor, dtype: pt.dtype = DTYPE ) -> pt.Tensor:
    """
    Use the molecule's first ground-truth (CREST) conformer as the starting
    geometry L_gt. Bypasses the RDKit-vs-CREST shift in local structure so
    the resulting AMR isolates score-net quality. Numbers from this mode are
    optimistic and NOT comparable to the paper's headline results (which use
    `rdkit_starting_geometry`).
    """
    s = int(x_offset[conf_idx])
    e = int(x_offset[conf_idx + 1])
    return x_flat[s:e].to(dtype=dtype)


def batched_kabsch_rmsd(x_samples: pt.Tensor, # (K, N, 3)
                        x_gts: pt.Tensor, # (G, N, 3)
                        heavy_mask: pt.Tensor, # (N,) bool
    ) -> pt.Tensor:
    """
    Pairwise heavy-atom Kabsch-aligned RMSDs between every sampled /
    ground-truth pair. One batched SVD across all K*G covariances.
    SVD is done in float64 on CPU for numerical stability.
    """
    og_dtype = x_samples.dtype

    K = int( x_samples.shape[0] )
    G = int( x_gts.shape[0] )

    if K == 0 or G == 0:
        return pt.empty( (K, G), dtype=og_dtype )

    dtype = pt.float64
    x_s = x_samples[:, heavy_mask, :].detach().cpu().to(dtype=dtype)
    x_g = x_gts[:, heavy_mask, :].detach().cpu().to(dtype=dtype)

    x_s = x_s - x_s.mean( dim=1, keepdim=True )
    x_g = x_g - x_g.mean( dim=1, keepdim=True )

    C = pt.einsum( "kim,gmj->kgij", x_s.transpose(-1, -2), x_g )

    U, _, Vh = pt.linalg.svd(C)
    det = pt.linalg.det( U @ Vh )
    D = pt.eye(3, dtype=dtype).expand(K, G, 3, 3).clone()
    D[..., 2, 2] = pt.sign(det)
    R = U @ D @ Vh

    x_s_rot = pt.einsum( "kmi,kgij->kgmj", x_s, R )
    diff = x_s_rot - x_g.unsqueeze(0)
    rmsd_sq = (diff ** 2).sum(dim=-1).mean(dim=-1)
    return pt.sqrt(rmsd_sq).to( dtype=og_dtype )


@pt.no_grad()
def sample_chunk( model: TorsionalScoreNetwork,
                  mol_data: list[dict],
                  K: int,
                  device: pt.device,
                  dtype: pt.dtype = DTYPE,
    ) -> list[pt.Tensor]:
    """
    Sample K conformers for EACH molecule in `mol_data`, all in one batched
    model forward. Returns a list of (K, N_i, 3) tensors, one per molecule.
    """
    if len(mol_data) == 0:
        return []

    molecules: list[MoleculeGraph] = []
    bonds_chunks: list[pt.Tensor]     = []
    sa_chunks: list[pt.Tensor]     = []
    sb_chunks: list[pt.Tensor]     = []
    bond_batch_ch: list[pt.Tensor]     = []

    atom_ranges: list[tuple[int, int]] = []
    Ns: list[int] = []

    atom_offset = 0
    bond_offset = 0
    bond_batch_idx = 0

    for md in mol_data:
        N = int( md["Z"].numel() )
        m = int( md["rot_bonds"].shape[0] )
        Ns.append(N)

        for _ in range(K):
            molecules.append( MoleculeGraph( Z=md["Z"], x=md["x_init"], bonds=md["cov_edges"] ) )
            bonds_chunks .append( md["rot_bonds"]      + atom_offset )
            sa_chunks.append( md["side_atom_idx"]  + atom_offset )
            sb_chunks.append( md["side_bond_idx"]  + bond_offset )
            bond_batch_ch.append( pt.full((m,), bond_batch_idx, dtype=pt.long) )

            atom_ranges.append( (atom_offset, atom_offset + N) )
            atom_offset += N
            bond_offset += m
            bond_batch_idx += 1

    mol_batched = batchMolecules( molecules ).to( device=device, dtype=dtype )
    bonds_b = pt.cat( bonds_chunks  ).to( device=device )
    sa_b = pt.cat( sa_chunks     ).to( device=device )
    sb_b = pt.cat( sb_chunks     ).to( device=device )
    bond_batch_b = pt.cat( bond_batch_ch ).to( device=device )

    x_sampled_flat = sample_torsional_diffusion(
        model = model,
        mol = mol_batched,
        bonds = bonds_b,
        side_atom_idx = sa_b,
        side_bond_idx = sb_b,
        bond_batch = bond_batch_b,
        n_steps = N_STEPS,
        cutoff = CUTOFF,
        dtype = dtype,
    )

    per_mol: list[pt.Tensor] = []
    for mi, N in enumerate(Ns):
        x_mi = pt.empty( (K, N, 3), dtype=dtype, device=device )
        for k in range(K):
            r = mi * K + k
            s, e = atom_ranges[r]
            x_mi[k] = x_sampled_flat[s:e]
        per_mol.append( x_mi )

    return per_mol


def pairwise_rmsd_matrix( sampled: pt.Tensor, ground_truth: list[pt.Tensor], Z: pt.Tensor ) -> np.ndarray:
    """
    (K, n_gt) heavy-atom Kabsch-aligned RMSDs via one batched SVD.
    """
    if len(ground_truth) == 0 or sampled.shape[0] == 0:
        return np.zeros((sampled.shape[0], len(ground_truth)), dtype=np.float32)

    heavy_mask = (Z != 1)
    if int(heavy_mask.sum()) < 3:
        heavy_mask = pt.ones_like(Z, dtype=pt.bool)

    gt_stack = pt.stack( ground_truth, dim=0 )
    rmsd = batched_kabsch_rmsd( sampled, gt_stack, heavy_mask )
    return rmsd.numpy()


def main():
    load_dotenv()
    with open("./data_config.json", "r") as f:
        data_config = json.load(f)
    qm9_dir = Path( data_config["qm9_folder"] )
    parsed_dir = qm9_dir.parent / "parsed"
    out_dir = Path( "./eval_chunks" )
    model_dir = Path( "./models" )

    chunk_idx = int(sys.argv[1])
    mode = sys.argv[2]
    if mode not in ("rdkit", "crest"):
        raise ValueError(f"unknown mode {mode!r}; expected 'rdkit' or 'crest'")
    device_str = 'mps'
    device = pt.device(device_str)

    data = pt.load(parsed_dir / "test.pt", weights_only=False)
    mol_Z = data["mol_Z"]
    mol_edge_index = data["mol_edge_index"]
    mol_bonds = data["mol_bonds"]
    mol_side_atom = data["mol_side_atom"]
    mol_side_bond = data["mol_side_bond"]
    x_flat = data["x_flat"]
    x_offset = data["x_offset"]
    mol_id = data["mol_id"]

    M = len(mol_Z)
    conformers_per_mol: list[list[int]] = [[] for _ in range(M)]
    for c in range(int(mol_id.numel())):
        conformers_per_mol[int(mol_id[c])].append(c)

    chunk_start = chunk_idx * B_MOL
    chunk_end   = min(chunk_start + B_MOL, M)

    model = TorsionalScoreNetwork().to(device=device, dtype=DTYPE)
    state = pt.load(model_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    mol_data = []
    per_mol_gt = []
    for mi in range(chunk_start, chunk_end):
        confs = conformers_per_mol[mi]
        if len(confs) == 0:
            continue
        Z = mol_Z[mi]

        if mode == "rdkit":
            smiles = data["mol_smiles"][mi]
            x_init = rdkit_starting_geometry(qm9_dir, smiles, dtype=DTYPE)
            if x_init is None:
                continue
        else:
            x_init = crest_starting_geometry(confs[0], x_flat, x_offset, dtype=DTYPE)
        mol_data.append({
            "Z":             Z,
            "cov_edges":     mol_edge_index[mi],
            "rot_bonds":     mol_bonds[mi],
            "side_atom_idx": mol_side_atom[mi],
            "side_bond_idx": mol_side_bond[mi],
            "x_init":        x_init,
        })
        gt_xs = []
        for c in confs:
            s = int(x_offset[c])
            e = int(x_offset[c + 1])
            gt_xs.append(x_flat[s:e])
        per_mol_gt.append((gt_xs, Z))

    rmsd_matrices: list[np.ndarray] = []
    if len(mol_data) > 0:
        samples_per_mol = sample_chunk(
            model    = model,
            mol_data = mol_data,
            K        = N_SAMPLES_PER_MOL,
            device   = device,
            dtype    = DTYPE,
        )
        for sampled, (gt_xs, Z) in zip(samples_per_mol, per_mol_gt):
            rmsd_matrices.append(pairwise_rmsd_matrix(sampled, gt_xs, Z))

    out_dir.mkdir( exist_ok=True )
    pt.save(rmsd_matrices, out_dir / f"{mode}_chunk_{chunk_idx}.pt")


if __name__ == "__main__":
    main()
