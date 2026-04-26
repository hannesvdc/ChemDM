import csv
import re
import json
from pathlib import Path
from typing import Any

import pandas as pd

from xtbSetup import create_xtb_context
from testTransition1xPipeline import build_molecule_reaction_map, runRelaxation

REACTION_FILE_PATTERN = re.compile( r"^(?P<split>.+?)_reaction_(?P<reaction_id>\d+)_molecule_(?P<molecule>.+)\.json$" )
REACTION_FILE_TEMPLATE = "{split}_reaction_{reaction_id}_molecule_{molecule}.json"

def append_row_to_csv(csv_path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def run_sweep( data_dir: str | Path,
               output_csv: str | Path,
               kind: str,
             ) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_csv = Path(output_csv)

    molecule_map = build_molecule_reaction_map( data_dir, kind )

    fieldnames = [
        # Dataset identifiers
        "split",
        "molecule",
        "reaction_id",

        # Relaxation info returned by runRelaxation
        "status",
        "lr",
        "force_tolerance_ev_A",
        "max_step_A",
        "initial_energy_kj_mol",
        "final_energy_kj_mol",
        "delta_energy_kj_mol",
        "initial_max_force_ev_A",
        "final_max_force_ev_A",

        # Displacement stats from disp_info
        "rmsd_displacement_A",
        "mean_displacement_A",
        "max_displacement_A",

        # Error info from relaxation itself
        "error_type",
        "error_message",
    ]

    rows: list[dict[str, Any]] = []
    n_done = 0

    for molecule, reaction_records in molecule_map.items():

        for reaction_id in reaction_records:
            filename = REACTION_FILE_TEMPLATE.format(split="train", reaction_id=reaction_id, molecule=molecule )
            with open( data_dir / filename, "r" ) as jsonfile:
                trajectory = json.load( jsonfile )
                print( "Reaction Loaded." )
            context = create_xtb_context( trajectory["Z"] )

            base_row = { "split": kind, "molecule": molecule, "reaction_id": reaction_id, }

            print( f"Running {kind} molecule {molecule}  reaction {reaction_id}" )

            try:
                _, info = runRelaxation( context, trajectory, verbose=False )
                row = { **base_row, **info }
            except:
                row = base_row,
                print( f"FAILED: {molecule} reaction {reaction_id}." )

            append_row_to_csv(output_csv, row, fieldnames=fieldnames)
            rows.append(row)
            n_done += 1

            del context

    return pd.DataFrame(rows)


def main() -> None:
    data_dir = Path( "/Users/hannesvdc/Open Numerics/ReactionStudio/data" )
    output_csv = data_dir / "xtb_relaxation_sweep.csv"

    df_train = run_sweep( data_dir=data_dir, output_csv=output_csv, kind="train" )
    df_val = run_sweep( data_dir=data_dir, output_csv=output_csv, kind="val" )
    df_test = run_sweep( data_dir=data_dir, output_csv=output_csv, kind="test" )
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)

    print()
    print(f"Wrote: {output_csv}")
    print(f"Rows written this run: {len(df)}")

    if len(df) > 0:
        success = df[df["status"] == "converged"]

        print()
        print("Relaxation status counts:")
        print(success["status"].value_counts(dropna=False))

        for column in [ "max_displacement_A",  "rmsd_displacement_A", "delta_energy_kj_mol", "final_max_force_ev_A", ]:
            if column in success.columns:
                print()
                print(column)
                print(pd.to_numeric(success[column], errors="coerce").describe())

if __name__ == "__main__":
    main()