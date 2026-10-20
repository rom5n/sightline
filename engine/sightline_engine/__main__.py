"""Entry point: `python -m sightline_engine --port 0 --token X --parent-pid N`."""
from __future__ import annotations

import os

# Before numpy loads: one BLAS/OpenMP thread per process. The data is small, and
# OpenBLAS otherwise commits a large buffer per core (hundreds of MB per process).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import logging
import multiprocessing as mp
import sys


def default_log_path() -> str:
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "Sightline", "logs", "engine.log")


def main() -> None:
    mp.freeze_support()
    ap = argparse.ArgumentParser(prog="sightline-engine")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--token", default=os.environ.get("SIGHTLINE_TOKEN", "dev"))
    ap.add_argument("--parent-pid", type=int, default=0)
    ap.add_argument("--log", default=default_log_path())
    a = ap.parse_args()

    from .runner import setup_logging, watch_parent
    setup_logging(a.log)
    log = logging.getLogger("sightline")
    log.info("engine starting on port %s", a.port)

    import uvicorn

    from .server import create_app
    app = create_app(a.token, a.log)

    def bye() -> None:
        hub = app.state.hub
        if hub.worker:
            hub.worker.shutdown()
        os._exit(0)

    watch_parent(a.parent_pid or None, bye)
    if sys.stdout is None:  # windowed PyInstaller build has no console
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", ws="websockets", lifespan="on",
                ws_max_size=64 * 1024 * 1024)


if __name__ == "__main__":
    main()
