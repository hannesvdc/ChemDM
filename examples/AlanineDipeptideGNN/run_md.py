from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import openmm.app as app
import numpy as np
from system import build_alanine_dipeptide_simulation

def run_seed(phi, psi, outdir_str, report_stride, n_steps):
    print( f"Running MD simulation with torsion angles ({phi}, {psi})..." )
    outdir = Path(outdir_str)

    simulation, topology, positions = build_alanine_dipeptide_simulation(
        temperature=300.0,
        friction=1.0,
        timestep_fs=2.0,
        platform_name="CPU",
        platform_properties={"Threads": "1"},
        phi_target=np.deg2rad(phi),
        psi_target=np.deg2rad(psi),
        minimize=True,
    )

    dcd_path = outdir / f"traj_phi={phi:+06.1f}_psi={psi:+06.1f}.dcd"
    simulation.reporters.append(app.DCDReporter(str(dcd_path), report_stride))

    simulation.step(n_steps)
    print( f"Done with ({phi}, {psi}).")

    return phi, psi

def main():
    outdir = Path( "outputs" )
    outdir.mkdir( exist_ok=True )

    report_stride = 1000
    n_steps = 2_000_000

    torsion_angles = np.linspace(-180, 180, 10, endpoint=False)
    jobs = [(phi, psi) for phi in torsion_angles for psi in torsion_angles]

    max_workers = 6
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(run_seed, phi, psi, str(outdir), report_stride, n_steps)
            for phi, psi in jobs
        ]

        for fut in as_completed(futures):
            phi, psi = fut.result()
            print(f"Finished ({phi:.1f}, {psi:.1f})")

    print("Done.")

if __name__ == "__main__":
    main()