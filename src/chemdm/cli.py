"""chemdm — command-line interface for ChemDM experiments.

Usage:
    chemdm <experiment> --input <path> --output <path>

Each subcommand reads a JSON input file and writes a JSON output file. The CLI
is the contract surface used by ReactionStudio's compute server, but the same
binary works directly from a shell or notebook.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Sequence


# One-shot experiments -> (command module, name of its model-loader function or None).
# Each is provisioned like the ReactionStudio worker: a progress callback plus, for
# commands that need one, a freshly loaded network.
_ONESHOT: dict[str, tuple[str, str | None]] = {
    "transition-path":        ( "chemdm.commands.transition_path",     "load_attention_model" ),
    "generate-conformers":    ( "chemdm.commands.generate_conformers", "load_torsional_diffusion_model" ),
    "stabilize-conformation": ( "chemdm.commands.stable_conformer",    None ),
}


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalars and arrays."""

    def default(self, obj):
        try:
            import numpy as np
        except ImportError:
            return super().default(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)


class _CliProgress:
    """Progress callback for CLI runs: prints stages to stderr and tracks the
    cumulative fraction, so commands that read `getTotalProgress()` behave the same
    as they do under the ReactionStudio worker."""

    def __init__(self) -> None:
        self.total_progress = 0.0

    def __call__(self, stage: str, message: str, fraction: float | None = None, **extra) -> None:
        if fraction is not None:
            self.total_progress = fraction
        pct = f" [{fraction:.0%}]" if fraction is not None else ""
        print(f"chemdm: {stage}: {message}{pct}", file=sys.stderr, flush=True)

    def getTotalProgress(self) -> float:
        return self.total_progress


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chemdm",
        description="ChemDM compute CLI. Each subcommand runs one experiment.",
    )
    sub = parser.add_subparsers(dest="experiment", required=True, metavar="<experiment>")
    for name in _ONESHOT:
        s = sub.add_parser(name, help=f"Run the {name} experiment.")
        s.add_argument( "--output", required=True, type=Path, help="Path to write output JSON." )
        if name == "generate-conformers":
            # A SMILES is the natural input here; --input JSON still works (the RS
            # contract), but --smiles is the convenient shortcut for local runs.
            src = s.add_mutually_exclusive_group( required=True )
            src.add_argument( "--smiles", help="SMILES string to generate conformers for." )
            src.add_argument( "--input", type=Path, help="Path to input JSON (RS contract)." )
            s.add_argument( "--n-conformers", type=int, default=100,
                            help="Number of conformers to generate (used with --smiles)." )
        else:
            s.add_argument( "--input", required=True, type=Path, help="Path to input JSON." )
        s.set_defaults( command_kind="oneshot" )

    # Long-lived worker command
    worker = sub.add_parser( "worker", help="Start the long-lived ChemDM worker." )
    worker.set_defaults( command_kind="worker" )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Build a persistent worker
    if args.command_kind == "worker":
        from chemdm.worker import run_worker
        return run_worker()

    # One-shot experiment. Provision it exactly like the worker does: a progress
    # callback plus (for commands that need one) a freshly loaded model.
    module_name, loader_name = _ONESHOT[args.experiment]
    module = importlib.import_module(module_name)

    if not hasattr(module, "run"):
        print( f"chemdm: subcommand {args.experiment!r} module {module_name!r} has no run()", file=sys.stderr )
        return 2

    # Assemble the input dict. --smiles (generate-conformers only) is a convenience
    # that builds the same dict the JSON contract would; otherwise read --input JSON.
    if getattr( args, "smiles", None ):
        input_data = { "smiles": args.smiles, "n_conformers": args.n_conformers }
    else:
        try:
            with open( args.input ) as f:
                input_data = json.load(f)
        except Exception as e:
            print( f"chemdm: failed to read input {args.input}: {e}", file=sys.stderr )
            return 1

    on_progress = _CliProgress()
    network = getattr( module, loader_name )() if loader_name else None
    output_data = ( module.run( input_data, on_progress )
                    if network is None
                    else module.run( input_data, on_progress, network ) )

    # Write the output file.
    args.output.parent.mkdir( parents=True, exist_ok=True )
    try:
        with open(args.output, "w") as f:
            json.dump(output_data, f, cls=_NumpyEncoder)
    except Exception as e:
        print(f"chemdm: failed to write output {args.output}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
