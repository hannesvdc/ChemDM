import argparse

from pathlib import Path

import pandas as pd

import matplotlib.pyplot as plt

def load_relaxation_table(csv_path: str | Path) -> pd.DataFrame:

    df = pd.read_csv(csv_path)

    numeric_columns = [
        "initial_energy_kj_mol",
        "final_energy_kj_mol",
        "delta_energy_kj_mol",
        "initial_max_force_ev_A",
        "final_max_force_ev_A",
        "rmsd_displacement_A",
        "mean_displacement_A",
        "max_displacement_A",
        "lr",
        "force_tolerance_ev_A",
        "max_step_A",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

def plot_histogram(df: pd.DataFrame, column: str, xlabel: str, bins: int = 80):
    values = df[column].dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(xlabel)
    plt.tight_layout()
    plt.show()

def plot_scatter( df: pd.DataFrame, x: str, y: str, xlabel: str, ylabel: str ):
    plot_df = df[[x, y]].dropna()

    plt.figure(figsize=(7, 5))
    plt.scatter(plot_df[x], plot_df[y], s=8, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {xlabel}")
    plt.tight_layout()
    plt.show()

def main():
    data_dir = Path( "/Users/hannesvdc/Open Numerics/ReactionStudio/data" )
    csv_path = "./Results/xtb_relaxation_sweep.csv"
    df = load_relaxation_table( csv_path )

    print("Total rows:", len(df))
    print()
    print("Status counts:")
    print(df["status"].value_counts(dropna=False))

    success = df[df["status"].isin(["converged", "max_steps"])].copy()
    success["energy_decrease_kj_mol"] = -success["delta_energy_kj_mol"]

    print()
    print("Rows used for plots:", len(success))

    columns = [ "energy_decrease_kj_mol",
                "max_displacement_A",
                "rmsd_displacement_A",
                "mean_displacement_A",
                "initial_max_force_ev_A",
                "final_max_force_ev_A",
    ]

    print()
    print("Summary statistics:")
    print( success[columns].describe( percentiles=[0.01, 0.05, 0.10, 0.50, 0.90, 0.95, 0.99] ) )

    print()
    max_displacement_count = 100
    print( f"Top {max_displacement_count} by max displacement:" )
    display_cols = [ "split",  "molecule", "reaction_id", "max_displacement_A", "rmsd_displacement_A",  "energy_decrease_kj_mol", "initial_max_force_ev_A", "final_max_force_ev_A", ]
    display_cols = [c for c in display_cols if c in success.columns]
    print(  success.sort_values("max_displacement_A", ascending=False).head(max_displacement_count)[display_cols].to_string(index=False) )

    plot_histogram( success, "energy_decrease_kj_mol", "Energy decrease after xTB relaxation [kJ/mol]" )

    plot_histogram( success, "max_displacement_A", "Maximum atomic displacement after xTB relaxation [Å]", )

    plot_histogram( success, "rmsd_displacement_A", "RMSD displacement after xTB relaxation [Å]", )

    plot_histogram( success, "final_max_force_ev_A", "Final max force [eV/Å]", )

    plot_scatter( success, x="energy_decrease_kj_mol", y="max_displacement_A", xlabel="Energy decrease [kJ/mol]", ylabel="Maximum atomic displacement [Å]" )

    plot_scatter( success, x="initial_max_force_ev_A", y="max_displacement_A", xlabel="Initial max force [eV/Å]", ylabel="Maximum atomic displacement [Å]" )

if __name__ == "__main__":
    main()