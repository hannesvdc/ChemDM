from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import mdtraj as md


def wrap_to_pi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


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


def print_atoms(traj):
    top = traj.topology
    for atom in top.atoms:
        print(atom.index, atom.residue.name, atom.name)


def main():
    outdir = Path("outputs")
    traj = md.load(str(outdir / "traj.dcd"), top=str(outdir / "topology.pdb"))

    print("Atom list:")
    print_atoms(traj)

    # Replace these with the quartets that match your topology.
    # Example placeholders only:
    phi_atoms = np.array([[4, 6, 8, 14]])
    psi_atoms = np.array([[6, 8, 14, 16]])

    phi = md.compute_dihedrals(traj, phi_atoms)[:, 0]
    psi = md.compute_dihedrals(traj, psi_atoms)[:, 0]

    phi = wrap_to_pi(phi)
    psi = wrap_to_pi(psi)

    np.savez(outdir / "phi_psi.npz", phi=phi, psi=psi)

    # Scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(np.degrees(phi), np.degrees(psi), s=3, alpha=0.35)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel(r"$\phi$ [deg]")
    plt.ylabel(r"$\psi$ [deg]")
    plt.tight_layout()
    plt.savefig(outdir / "phi_psi_scatter.png", dpi=200)
    plt.close()

    # FE surface
    phi_c, psi_c, F = free_energy_from_hist(phi, psi, temperature=300.0, bins=72)
    PHI, PSI = np.meshgrid(np.degrees(phi_c), np.degrees(psi_c), indexing="xy")

    plt.figure(figsize=(7, 6))
    cs = plt.contourf(PHI, PSI, F, levels=25)
    plt.colorbar(cs, label="Free energy [kJ/mol] + const")
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel(r"$\phi$ [deg]")
    plt.ylabel(r"$\psi$ [deg]")
    plt.tight_layout()
    plt.savefig(outdir / "fe_surface.png", dpi=300)
    print("Saved phi/psi and FE surface.")

    # Make a scatter plot
    plt.figure()
    plt.scatter(np.degrees(phi), np.degrees(psi), s=2, alpha=0.3)
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel(r'$\phi$ [deg]')
    plt.ylabel(r'$\psi$ [deg]')


    plt.figure()
    plt.plot(np.degrees(phi), lw=0.8, alpha=0.4, label=r'$\phi$')
    plt.plot(np.degrees(psi), lw=0.8, alpha=0.4, label=r'$\psi$')
    plt.legend()
    plt.xlabel("frame")
    plt.ylabel("angle [deg]")

    plt.show()


if __name__ == "__main__":
    main()