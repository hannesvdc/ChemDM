import math
import numpy as np

import py3Dmol
from conformerViewer import make_page, launch_html_viewer
from rdkit import Chem
from rdkit.Chem import AllChem, rdchem
from rdkit.Geometry import Point3D
import matplotlib.pyplot as plt

from chemdm.xtbSetup import create_xtb_context
from chemdm.geometry import kabsch_align_numpy
from stabilizeConformer import stabilizeConformer

def print_heavy_atom_table(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            print( atom.GetIdx(), atom.GetSymbol(), "neighbors:", [(n.GetIdx(), n.GetSymbol()) for n in atom.GetNeighbors()], )

def rdkit_positions_to_numpy(mol: Chem.Mol, conf_id: int) -> np.ndarray:
    """ RDKIT uses Angstrom internally, and so do we. """
    conf = mol.GetConformer(conf_id)

    positions = np.zeros( (mol.GetNumAtoms(), 3), dtype=float )
    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        positions[i] = [p.x, p.y, p.z]

    return positions

def numpy_positions_to_rdkit(mol: Chem.Mol, conf_id: int, positions: np.ndarray) -> None:
    conf = mol.GetConformer(conf_id)
    for i, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(i, (float(x), float(y), float(z)))

def mol_with_new_positions(template_mol: Chem.Mol, x: np.ndarray) -> Chem.Mol:
    """
    Create a new RDKit Mol with the same atoms/bonds as template_mol,
    but with one conformer whose coordinates are x.

    Arguments:
    ----------
    template_mol: RDKit Mol containing the desired atom/bond topology.
    x: NumPy array of shape (N_atoms, 3), in Angstrom.

    Returns:
    --------
    new_mol: New RDKit Mol with same topology and new 3D coordinates.
    """
    x = np.asarray(x, dtype=float)
    if x.shape != (template_mol.GetNumAtoms(), 3):
        raise ValueError( f"x must have shape ({template_mol.GetNumAtoms()}, 3), got {x.shape}" )

    mol = Chem.Mol(template_mol)   # copy topology/properties
    mol.RemoveAllConformers()

    conf = rdchem.Conformer( mol.GetNumAtoms() )
    conf.Set3D(True)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition( i, Point3D(float(x[i, 0]), float(x[i, 1]), float(x[i, 2])), )
    mol.AddConformer(conf, assignId=True)

    return mol

def cluster( heavy_atoms : np.ndarray, 
             conformers : list[np.ndarray], 
             energies : list[float],
             rmsd_tol : float = 0.5
            ) -> tuple[list[np.ndarray], list[float]]:
    assert len(conformers) == len(energies), f"`confomers` and `energies` must contain the same number of elements."
    if len(conformers) == 0:
        return conformers, energies
    
    initial_conformer = conformers[0] - np.mean( conformers[0], axis=0, keepdims=True )
    optimal_conformers = [ initial_conformer ]
    optimal_energies = [ energies[0] ]
    for n in range(1, len(conformers)):
        print( 'Conformer', n )
        x_conf = conformers[n] - np.mean( conformers[n], axis=0, keepdims=True )

        is_new = True
        for k in range( len(optimal_conformers) ):
            conformer_heavy = optimal_conformers[k][heavy_atoms,:]

            # Center and Kabsch align
            x_conf_aligned = x_conf[heavy_atoms,:]
            x_conf_aligned = kabsch_align_numpy( x_conf_aligned, conformer_heavy )

            # Compute the per-atom RMSD
            rmsd = np.sqrt(np.mean(np.sum( (x_conf_aligned - conformer_heavy)**2, axis=1 ) ))
            print(rmsd)
            
            if rmsd <= rmsd_tol:
                is_new = False
                break

        if is_new:
            print( "New Conformer Found!" )
            optimal_conformers.append( x_conf )
            optimal_energies.append( energies[n] )

    return optimal_conformers, energies

def wrap_to_pi( angle ):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def compute_dihedral(p0 : np.ndarray, 
                     p1 : np.ndarray, 
                     p2 : np.ndarray, 
                     p3 : np.ndarray) -> np.ndarray:
    """
    Signed dihedral angle in radians.

    Parameters
    ----------
    p0, p1, p2, p3 : ndarray
        Arrays of shape (..., 3)

    Returns
    -------
    angle : ndarray
        Array of shape (...) with signed dihedral angles in radians.
    """
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1, axis=-1, keepdims=True)
    b1_hat = b1 / (b1_norm + 1e-12)

    v = b0 - np.sum(b0 * b1_hat, axis=-1, keepdims=True) * b1_hat
    w = b2 - np.sum(b2 * b1_hat, axis=-1, keepdims=True) * b1_hat

    x = np.sum(v * w, axis=-1)
    y = np.sum( np.cross(b1_hat, v, axis=-1) * w, axis=-1 )

    return np.arctan2(y, x)

def compute_torsion_from_xyz(xyz : np.ndarray, 
                             atoms : tuple[int,int,int,int]):
    """
    Compute one torsion angle from coordinates.

    Parameters
    ----------
    xyz : ndarray
        Array of shape (..., n_atoms, 3)
    atoms : tuple[int, int, int, int]
        Atom quartet defining the torsion.

    Returns
    -------
    angle : ndarray
        Array of shape (...) in degrees.
    """
    if xyz.ndim < 2 or xyz.shape[-1] != 3:
        raise ValueError( f"xyz must have shape (..., n_atoms, 3), got {xyz.shape}" )

    i, j, k, l = atoms
    return 180.0 / math.pi * wrap_to_pi( compute_dihedral(
            xyz[..., i, :],
            xyz[..., j, :],
            xyz[..., k, :],
            xyz[..., l, :],
        ) )

def free_energy_from_hist(phi, psi, temperature=300.0, bins=72, eps=1e-12):
    H, phi_edges, psi_edges = np.histogram2d(
        phi,
        psi,
        bins=bins,
        range=[[-np.pi, np.pi], [-np.pi, np.pi]],
        density=True,
    )

    H = H + eps
    kB = 0.00831446261815324  # kJ/mol/K
    F = -kB * temperature * np.log(H)
    F -= np.min(F)

    phi_centers = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    psi_centers = 0.5 * (psi_edges[:-1] + psi_edges[1:])
    return phi_centers, psi_centers, F.T

def alanine():
    """
    Main driver script for the Alanine-Dipeptide testcase.

    TODO: 
    1. Parallelize xTB conformer search
    """

    smiles = "CC(=O)N[C@@H](C)C(=O)NC"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    print( Chem.MolToSmiles(mol) )

    # Add hydrogens
    mol_with_h = Chem.AddHs( mol )
    heavy_atom_indices = np.array( [ atom.GetIdx() for atom in mol_with_h.GetAtoms() if atom.GetAtomicNum() != 1 ], dtype=np.long )
    print( '\nHeavy Atom Indices:', heavy_atom_indices )
    print_heavy_atom_table( mol_with_h )
    phi_atoms = (1, 3, 4, 6)  # C_prev - N - Cα - C
    psi_atoms = (3, 4, 6, 8)  # N - Cα - C - N_next

    # Generate rotamers / conformers
    params = AllChem.ETKDGv3()
    params.pruneRmsThresh = 0.1  # remove very similar conformers during embedding

    n_confs = 100_000
    print( f'\nGenerating {n_confs} initial conformers...')
    conf_ids = AllChem.EmbedMultipleConfs( mol_with_h, numConfs=n_confs, params=params )
    print(f"Generated {len(conf_ids)} distinct conformers")
    print( "ConfIds:", list(conf_ids) )

    # Stabilize all generated conformers.
    atomic_numbers = np.array( [ int(atom.GetAtomicNum()) for atom in mol_with_h.GetAtoms() ], dtype=np.long )
    xtb_context = create_xtb_context( atomic_numbers )
    optimal_conformers = []
    energies = []
    for conf_id in conf_ids:
        print( f'\nStabilizing Conformer {conf_id}.' )
        conf_positions = rdkit_positions_to_numpy( mol_with_h, conf_id )
        conf_opt, E_opt, F_opt, _ = stabilizeConformer( xtb_context, conf_positions, force_tol=1e-2 )
        print( f'Conformer {conf_id} stabilized to E = {E_opt} and |F| = { np.linalg.norm(F_opt)}.' )
        print( f"Torsion angles: ({compute_torsion_from_xyz(conf_opt,phi_atoms)}, {compute_torsion_from_xyz(conf_opt,psi_atoms)})")

        optimal_conformers.append( conf_opt )
        energies.append( E_opt )

    # Cluster to determine unique conformers
    optimal_conformers, energies = cluster( heavy_atom_indices, optimal_conformers, energies, rmsd_tol=0.5 )

    # Create RDKit Conformer objects
    conformers_mol_objects = [ mol_with_new_positions(mol_with_h, optimal_conformers[ii]) for ii in range(len(optimal_conformers)) ]
    def mol_to_py3dmol_html(mol) -> str:
        mol_block = Chem.MolToMolBlock(mol)
        viewer = py3Dmol.view(width=500, height=400)
        viewer.addModel(mol_block, "mol")
        viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
        viewer.zoomTo()
        return viewer._make_html()
    molecule_html_blocks = [ mol_to_py3dmol_html(mol) for mol in conformers_mol_objects ]
    phi_values = []
    psi_values = []
    energy_values = []
    for conf in optimal_conformers:
        phi = compute_torsion_from_xyz( conf, phi_atoms )
        psi = compute_torsion_from_xyz( conf, psi_atoms )
        phi_values.append( phi )
        psi_values.append( psi )
        print( f"Stable unique conformer ({phi}, {psi}).")

    # Plot the unique conformers on the FEC.
    with np.load('./data/fec.npz') as fec:
        phi_fec = fec['phi']
        psi_fec = fec['psi']
    phi_c, psi_c, F = free_energy_from_hist(phi_fec, psi_fec, temperature=300.0, bins=72)
    PHI, PSI = np.meshgrid(np.degrees(phi_c), np.degrees(psi_c), indexing="xy")

    # Make a contour plot and display the unique stable conformers.
    plt.figure(figsize=(7, 6))
    cs = plt.contourf(PHI, PSI, F, levels=25)
    plt.colorbar(cs, label="Free energy [kJ/mol] + const")
    plt.scatter( phi_values, psi_values, color='red', label='Conformers' )
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel(r"$\phi$ [deg]")
    plt.ylabel(r"$\psi$ [deg]")
    plt.tight_layout()

    plt.show( block=False )

    html = make_page(molecule_html_blocks, phi_values, psi_values, energies)
    launch_html_viewer( html, title="Alanine Dipeptide Conformers", width=1500, height=900, use_temp_file=True )

    def write_conformers_mol_files(conformer_mols, prefix: str = "conformer"):
        for i, mol in enumerate(conformer_mols):
            filename = f"./conformers_ad/{prefix}_{i}.mol"
            Chem.MolToMolFile(mol, filename)
            print(f"Wrote {filename}")
    write_conformers_mol_files( conformers_mol_objects )

if __name__ == '__main__':
    alanine()