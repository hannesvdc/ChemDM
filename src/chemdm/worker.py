# chemdm/worker.py

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Callable

from chemdm.commands.transition_path import run as run_transition_path
from chemdm.commands.transition_path import load_transition_path_model


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy scalars and arrays."""

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


def emit( obj: dict[str, Any] ) -> None:
    """
    Emit one protocol message.

    Important:
    - stdout is reserved for NDJSON protocol messages only.
    - every message is one JSON object followed by one newline.
    - flush is required so the server receives progress immediately.
    """
    sys.stdout.write( json.dumps(obj, cls=_NumpyEncoder) + "\n" )
    sys.stdout.flush()


class WorkerState:
    """
    Long-lived state owned by the worker process.

    Put expensive objects here: ML models, diffusion models, cached configs, etc.
    This object is created once when `chemdm worker` starts.
    """

    def __init__(self) -> None:
        self.transition_path_model = None

    def warm_up(self) -> None:
        """
        Load heavy resources once.

        For the first version, this can be empty. Later you can move your
        Newton model loading here so it is not repeated per job.
        """
        self.transition_path_model = load_transition_path_model()


def run_worker() -> int:
    """
    Main worker loop. Reads NDJSON jobs from stdin and writes NDJSON events to stdout.
    """

    state = WorkerState()
    try:
        state.warm_up()
    except Exception as e:
        emit( {
                "kind": "fatal",
                "message": f"Worker warm-up failed: {e}",
                "trace": traceback.format_exc(),
            } )
        return 1

    emit( {
            "kind": "ready",
            "protocol_version": 1,
            "message": "ChemDM worker is ready.",
        } )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        job: dict[str, Any] | None = None

        try:
            job = json.loads( line )

            assert job is not None
            handle_job( job, state )
        except SystemExit:
            raise
        except Exception as e:
            emit( {
                    "kind": "error",
                    "job_id": job.get("job_id") if isinstance(job, dict) else None,
                    "message": str(e),
                    "trace": traceback.format_exc(),
                } )
    return 0


def handle_job(job: dict[str, Any], state: WorkerState) -> None:
    """
    Handle one job message. Expected input shape:

    {
        "kind": "run",
        "job_id": "...",
        "experiment": "transition-path",
        "body": {...}
    }

    `kind` may be omitted and defaults to "run".
    """

    kind = job.get( "kind", "run" )

    if kind == "shutdown":
        emit({"kind": "shutdown"})
        raise SystemExit(0)

    if kind != "run":
        emit( {
                "kind": "error",
                "job_id": job.get("job_id"),
                "message": f"Unknown job kind: {kind!r}",
            } )
        return

    job_id = job["job_id"]
    experiment = job["experiment"]
    body = job["body"]

    emit( {
            "kind": "accepted",
            "job_id": job_id,
            "experiment": experiment,
        } )

    def on_progress( stage: str,
                     message: str,
                     fraction: float | None = None,
                     **extra: Any, ) -> None:
        event: dict[str, Any] = {
            "kind": "progress",
            "job_id": job_id,
            "stage": stage,
            "message": message,
        }

        if fraction is not None:
            event["fraction"] = fraction

        event.update(extra)
        emit(event)

    try:
        if experiment == "transition-path":
            result = run_transition_path(
                body,
                on_progress=on_progress,
                tp_network=state.transition_path_model, ) # type: ignore
        else:
            raise ValueError(f"Unknown experiment: {experiment!r}")

        emit( {
                "kind": "done",
                "job_id": job_id,
                "result": result,
            } )

    except Exception as e:
        emit( {
                "kind": "error",
                "job_id": job_id,
                "message": str(e),
                "trace": traceback.format_exc(),
            } )