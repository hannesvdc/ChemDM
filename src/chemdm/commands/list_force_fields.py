"""List the force fields / potentials available in this ChemDM deployment.

Metadata command consumed by the ReactionStudio front end. It returns the
catalog of selectable ``force_field`` values, each annotated with:

* ``available`` -- whether the backend's dependencies are installed here, so
  the UI can grey out what it cannot run (with a ``reason``);
* ``supports_tp`` -- whether the method can drive a transition-path (NEB) run,
  so the UI only offers the TP workflow for methods that can do one.

Takes no input; ``run`` follows the standard ``run(input_data, ...) -> dict``
command contract so it dispatches identically through the CLI and the worker.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from chemdm.potentialInterface import available_force_fields


def run( input_data: dict | None = None,
         on_progress: Optional[Callable] = None ) -> dict[str, Any]:
    return { "force_fields": available_force_fields() }
