"""Training runner.

Training runs in a separate, pre-warmed worker process so the engine never blocks.
Pause / step / speed / cancel go through a shared control block that the worker reads
between iterations (≤ 10 ms latency). A fit that cannot be interrupted (one long
libsvm call, a big batch of trees) is cancelled by killing the worker and starting
a fresh one.
"""
from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .errors import friendly

# control block slots (shared RawArray of doubles)
C_JOB, C_PAUSED, C_SPEED, C_BLOCKING, C_WORKER_JOB, C_FAST = range(6)
FRAME_DT = 1 / 30
VIZ_BUDGET_MS = 18.0

log = logging.getLogger("sightline")


@dataclass
class Job:
    id: int
    model: str
    params: dict[str, Any]
    prep: Any  # Prepared
    seed: int
    view: dict[str, float] | None
    fast: bool
    probe_id: int | None


def setup_logging(log_path: str | None) -> None:
    if not log_path:
        return
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    h = logging.FileHandler(log_path, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(process)d %(levelname)s %(message)s"))
    root = logging.getLogger("sightline")
    root.addHandler(h)
    root.setLevel(logging.INFO)


def watch_parent(pid: int | None, on_exit: Callable[[], None]) -> None:
    """Exit when the parent process dies (Tauri app or engine), so nothing is orphaned."""
    if not pid:
        return

    def run() -> None:
        if sys.platform == "win32":
            import ctypes
            SYNCHRONIZE = 0x00100000
            k32 = ctypes.windll.kernel32
            h = k32.OpenProcess(SYNCHRONIZE, False, pid)
            if not h:
                on_exit()
                return
            k32.WaitForSingleObject(h, 0xFFFFFFFF)
            on_exit()
        else:
            while True:
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                except OSError:
                    on_exit()
                    return

    threading.Thread(target=run, daemon=True).start()


# ====================================================================== worker side


def worker_main(cmd_q: Any, out_q: Any, ctrl: Any, tokens: Any, log_path: str | None, parent_pid: int) -> None:
    setup_logging(log_path)
    watch_parent(parent_pid, lambda: os._exit(0))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import sklearn

    from .adapters import schemas  # heavy import happens here, in the worker
    w = Worker(cmd_q, out_q, ctrl, tokens)
    out_q.put({"type": "worker_ready", "pid": os.getpid(), "schemas": schemas(), "sklearn": sklearn.__version__})
    w.loop()


class Worker:
    def __init__(self, cmd_q: Any, out_q: Any, ctrl: Any, tokens: Any) -> None:
        self.cmd_q, self.out_q, self.ctrl, self.tokens = cmd_q, out_q, ctrl, tokens
        self.pending: deque[tuple[Any, ...]] = deque()
        self.adapter: Any = None
        self.job: Job | None = None
        self.res_scale = 1.0
        self.last_step: Any = None
        self.last_progress = 0.0
        self.training = False

    # ---------------------------------------------------------------- loop

    def loop(self) -> None:
        while True:
            cmd = self.pending.popleft() if self.pending else self.cmd_q.get()
            try:
                self.handle(cmd)
            except Exception as e:  # noqa: BLE001 - every failure becomes a friendly message
                self.out_q.put(friendly("worker", e, cmd[-1] if cmd and isinstance(cmd[-1], int) else None))

    def handle(self, cmd: tuple[Any, ...]) -> None:
        kind = cmd[0]
        if kind == "train":
            self.run(cmd[1])
        elif kind == "explain":
            self.explain(cmd[1], cmd[2])
        elif kind == "view":
            if self.job is not None:
                self.job.view = cmd[1]
            if not self.training and self.adapter is not None and self.adapter.model is not None and self.last_step is not None:
                self.emit(self.last_step, final=True)
        elif kind == "probe":
            if self.job is not None:
                self.job.probe_id = cmd[1]
            self.send_probe()
        elif kind == "reset":
            self.adapter = None
            self.job = None
            self.last_step = None
        elif kind == "quit":
            os._exit(0)

    def poll(self) -> None:
        while True:
            try:
                cmd = self.cmd_q.get_nowait()
            except queue.Empty:
                return
            if cmd[0] in ("explain", "view", "probe"):
                self.handle(cmd)
            else:
                self.pending.append(cmd)

    # ---------------------------------------------------------------- training

    def cancelled(self) -> bool:
        return self.job is None or int(self.ctrl[C_JOB]) != self.job.id

    def progress(self, p: float) -> None:
        now = time.perf_counter()
        if now - self.last_progress < 0.1 and p < 1.0:
            return
        self.last_progress = now
        if self.job is not None:
            self.out_q.put({"type": "status", "state": "fitting", "job": self.job.id, "progress": float(p)})

    def run(self, job: Job) -> None:
        from .adapters import REGISTRY
        from .adapters.base import Ctx

        if int(self.ctrl[C_JOB]) != job.id:
            return  # superseded before it started
        self.job = job
        self.ctrl[C_WORKER_JOB] = job.id
        ctx = Ctx(job.prep, job.params, job.seed, cancelled=self.cancelled, progress=self.progress)
        adapter = REGISTRY[job.model](ctx)
        if hasattr(adapter, "probe_row") and job.probe_id is not None:
            try:
                adapter.probe_row = job.prep.row_of(job.probe_id)
            except KeyError:
                adapter.probe_row = None
        self.adapter = adapter
        self.last_step = None
        self.res_scale = 1.0
        from .adapters import RATE
        rate = RATE.get(job.model, 20.0)
        t0 = time.perf_counter()
        last_emit = 0.0
        emitted_iter = -1
        self.training = True
        self.out_q.put({"type": "status", "state": "paused" if self.ctrl[C_PAUSED] else "training", "job": job.id})
        reason = "cancelled"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gen = adapter.fit_stepwise()
                while True:
                    try:
                        st = next(gen)
                    except StopIteration:
                        break
                    if st.blocking is not None:
                        self.ctrl[C_BLOCKING] = 1
                        self.progress(st.blocking)
                        continue
                    self.ctrl[C_BLOCKING] = 0
                    if self.cancelled():
                        return
                    self.last_step = st
                    self.poll()
                    now = time.perf_counter()
                    if st.done or self.ctrl[C_PAUSED] or now - last_emit >= FRAME_DT:
                        self.emit(st, final=bool(st.done))
                        last_emit = time.perf_counter()
                        emitted_iter = st.iteration
                    if st.done:
                        reason = st.done
                        break
                    emitted_iter = self.wait_next(rate, st, emitted_iter)
                    if self.cancelled():
                        return
            if self.last_step is not None and emitted_iter != self.last_step.iteration:
                self.emit(self.last_step, final=True)
            if self.cancelled():
                return
            try:
                summary = adapter.summary() if adapter.model is not None else {}
            except Exception as e:  # noqa: BLE001 - a summary problem must not hide a finished fit
                log.exception("summary failed: %s", e)
                summary = {}
            self.out_q.put({"type": "done", "job": job.id, "reason": reason,
                            "iteration": self.last_step.iteration if self.last_step else 0,
                            "seconds": time.perf_counter() - t0, "summary": summary})
            self.out_q.put({"type": "status", "state": "idle", "job": job.id})
            if job.probe_id is not None:
                self.send_probe()
        except Exception as e:  # noqa: BLE001
            self.adapter = None
            self.out_q.put(friendly("train_failed", e))
            self.out_q.put({"type": "status", "state": "idle", "job": job.id})
        finally:
            self.ctrl[C_BLOCKING] = 0
            self.training = False

    def wait_next(self, rate: float, st: Any, emitted_iter: int) -> int:
        """Pacing + pause/step handling between two iterations."""
        assert self.job is not None
        start = time.perf_counter()
        was_paused = False
        while not self.cancelled():
            if self.ctrl[C_PAUSED]:
                if not was_paused:
                    was_paused = True
                    if emitted_iter != st.iteration:
                        self.emit(st, final=False)
                        emitted_iter = st.iteration
                    self.out_q.put({"type": "status", "state": "paused", "job": self.job.id, "iteration": st.iteration})
                with self.tokens.get_lock():
                    if self.tokens.value > 0:
                        self.tokens.value -= 1
                        return emitted_iter
                self.poll()
                time.sleep(0.008)
                continue
            if was_paused:
                self.out_q.put({"type": "status", "state": "training", "job": self.job.id})
                return emitted_iter
            if self.job.fast or self.ctrl[C_FAST]:
                return emitted_iter
            speed = max(0.05, float(self.ctrl[C_SPEED]))
            if time.perf_counter() - start >= 1.0 / (rate * speed):
                return emitted_iter
            self.poll()
            time.sleep(0.004)
        return emitted_iter

    def grid(self, final: bool) -> Any:
        from .adapters.base import Grid

        assert self.job is not None
        p = self.job.prep
        v = self.job.view
        if v:
            x0, x1, y0, y1 = v["x0"], v["x1"], v["y0"], v["y1"]
            w, h = int(v.get("w", 160)), int(v.get("h", 100))
        else:
            x0, x1, y0, y1 = p.plane.bounds
            w, h = 160, 100
        s = 1.0 if final else self.res_scale
        return Grid(x0, x1, y0, y1, max(16, int(round(w * s))), max(10, int(round(h * s))))

    def emit(self, st: Any, final: bool) -> None:
        assert self.job is not None and self.adapter is not None
        g = self.grid(final)
        t = time.perf_counter()
        viz = self.adapter.visualize(g)
        ms = (time.perf_counter() - t) * 1000
        if final:
            pp = self.adapter.point_predictions()
            if pp is not None:
                viz["point_pred"] = pp
        if not final:
            if ms > VIZ_BUDGET_MS * 1.3:
                self.res_scale = max(0.3, self.res_scale * max(0.5, math.sqrt(VIZ_BUDGET_MS / ms)))
            elif ms < VIZ_BUDGET_MS * 0.4:
                self.res_scale = min(1.0, self.res_scale * 1.15)
        self.out_q.put({
            "type": "frame", "job": self.job.id, "iteration": int(st.iteration), "total": int(st.total),
            "loss": st.loss, "loss_test": st.loss_test, "metric_train": st.metric_train, "metric_test": st.metric_test,
            "extra": st.extra, "viz": viz, "final": final, "viz_ms": ms, "t": time.time(),
        })

    def explain(self, point_id: int, rid: int | None) -> None:
        if self.adapter is None or self.adapter.model is None or self.job is None:
            self.out_q.put(friendly("not_trained", None, rid))
            return
        try:
            row = self.job.prep.row_of(point_id)
        except KeyError:
            self.out_q.put(friendly("unknown_point", None, rid))
            return
        data = self.adapter.explain(row)
        data["check"] = _plain(self.adapter.predict_row(row))
        p = self.job.prep
        data["point_id"] = point_id
        data["xy"] = [float(p.xy[row, 0]), float(p.xy[row, 1])]
        # the row as the user knows it: original columns, categories as text (not one-hot)
        if p.columns:
            vals = p.raw(p.X[row:row + 1])[0]
            data["raw"] = {c.name: (str(vals[j]) if c.kind == "category" else float(vals[j])) for j, c in enumerate(p.columns)}
        else:
            data["raw"] = {n: float(p.X[row, j]) for j, n in enumerate(p.feature_names)}
        data["is_test"] = bool(row in set(p.test_idx.tolist()))
        self.out_q.put({"type": "explain_steps", "point_id": point_id, "family": self.adapter.explain_family,
                        "data": data, "rid": rid})

    def send_probe(self) -> None:
        if self.adapter is None or self.job is None or self.job.probe_id is None or not hasattr(self.adapter, "probe"):
            return
        try:
            row = self.job.prep.row_of(self.job.probe_id)
        except KeyError:
            return
        if hasattr(self.adapter, "probe_row"):
            self.adapter.probe_row = row
        data = self.adapter.probe(row)
        if data is not None:
            self.out_q.put({"type": "probe", "point_id": self.job.probe_id, "data": data})


def _plain(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:  # pragma: no cover
        pass
    return v


# ====================================================================== engine side


class WorkerHandle:
    """Owns the worker process; relays its messages to `on_msg` (called from a thread)."""

    def __init__(self, on_msg: Callable[[dict[str, Any]], None], log_path: str | None) -> None:
        self.mpctx = mp.get_context("spawn")
        self.on_msg = on_msg
        self.log_path = log_path
        self.ready = False
        self.lock = threading.Lock()
        self.proc: Any = None
        self.spawn()

    def spawn(self) -> None:
        c = self.mpctx
        self.cmd_q = c.Queue()
        self.out_q = c.Queue()
        self.ctrl = c.RawArray("d", 8)
        self.ctrl[C_SPEED] = 1.0
        self.ctrl[C_JOB] = -1
        self.tokens = c.Value("i", 0)
        self.ready = False
        proc = c.Process(target=worker_main, args=(self.cmd_q, self.out_q, self.ctrl, self.tokens, self.log_path, os.getpid()),
                         daemon=True, name="sightline-worker")
        proc.start()
        self.proc = proc
        threading.Thread(target=self._reader, args=(self.out_q, proc), daemon=True).start()

    def _reader(self, q: Any, proc: Any) -> None:
        while True:
            try:
                msg = q.get(timeout=0.5)
            except queue.Empty:
                if not proc.is_alive():
                    if proc is self.proc:
                        log.error("worker %s died (exit %s)", proc.pid, proc.exitcode)
                        self.on_msg(friendly("worker_crashed"))
                        with self.lock:
                            self.spawn()
                    return
                continue
            except (EOFError, OSError):
                return
            if proc is not self.proc:
                continue  # message from a killed worker
            if msg.get("type") == "worker_ready":
                self.ready = True
            self.on_msg(msg)

    # ---------------------------------------------------------------- commands

    def submit(self, job: Job, paused: bool) -> None:
        with self.lock:
            if self.ctrl[C_BLOCKING]:
                self._restart()
            self.ctrl[C_PAUSED] = 1 if paused else 0
            with self.tokens.get_lock():
                self.tokens.value = 0
            self.ctrl[C_JOB] = job.id
            self.cmd_q.put(("train", job))

    def cancel(self) -> None:
        with self.lock:
            self.ctrl[C_JOB] = -1
            if self.ctrl[C_BLOCKING]:
                self._restart()

    def _restart(self) -> None:
        old = self.proc
        self.proc = None
        try:
            old.kill()
        except Exception:  # noqa: BLE001
            pass
        self.on_msg({"type": "status", "state": "starting", "job": None})
        self.spawn()

    def pause(self, on: bool) -> None:
        self.ctrl[C_PAUSED] = 1 if on else 0

    def step(self) -> None:
        self.ctrl[C_PAUSED] = 1
        with self.tokens.get_lock():
            self.tokens.value += 1

    def speed(self, s: float) -> None:
        self.ctrl[C_SPEED] = float(s)

    def send(self, *cmd: Any) -> None:
        self.cmd_q.put(cmd)

    @property
    def blocking(self) -> bool:
        return bool(self.ctrl[C_BLOCKING])

    def shutdown(self) -> None:
        try:
            if self.proc is not None:
                self.proc.kill()
        except Exception:  # noqa: BLE001
            pass
