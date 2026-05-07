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


_SUBCOMMANDS: dict[str, str] = {
    "transition-path": "chemdm.commands.transition_path",
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chemdm",
        description="ChemDM compute CLI. Each subcommand runs one experiment.",
    )
    sub = parser.add_subparsers(dest="experiment", required=True, metavar="<experiment>")
    for name in _SUBCOMMANDS:
        s = sub.add_parser(name, help=f"Run the {name} experiment.")
        s.add_argument( "--input", required=True, type=Path, help="Path to input JSON." )
        s.add_argument( "--output", required=True, type=Path, help="Path to write output JSON." )
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

    # Existing one-shot behavior
    module_name = _SUBCOMMANDS[args.experiment]
    module = importlib.import_module(module_name)

    if not hasattr(module, "run"):
        print( f"chemdm: subcommand {args.experiment!r} module {module_name!r} has no run()", file=sys.stderr )
        return 2

    try:
        with open( args.input ) as f:
            input_data = json.load(f)
    except Exception as e:
        print( f"chemdm: failed to read input {args.input}: {e}", file=sys.stderr )
        return 1

    # Run the one-shot job (debugging / testing mode only typically)
    output_data = module.run(input_data)

    # Write the output file.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(args.output, "w") as f:
            json.dump(output_data, f, cls=_NumpyEncoder)
    except Exception as e:
        print(f"chemdm: failed to write output {args.output}: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
