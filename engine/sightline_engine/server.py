"""FastAPI WebSocket server: one session, many windows (main + pop-outs)."""
from __future__ import annotations

import asyncio
import hmac
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from . import datasets as dsv
from .errors import friendly
from .prep import Prepared, prepare
from .protocol import (
    Cancel, ClientMessage, DatasetStats, EditPoints, Explain, Hello, LoadDataset, ParseCsv, Pause, PreviewGenerator,
    PreviewPreset, Probe, RemoveRows, Reset, Resume, SetModel, SetParams, SetScale, SetSpeed, SetSplitSeed, SetView, Step,
    Train,
)
from .runner import Job, WorkerHandle
from .wire import pack, unpack

log = logging.getLogger("sightline")
CLIENT = TypeAdapter(ClientMessage)
DEFAULT_SEED = 42


@dataclass
class Session:
    dataset: dsv.Dataset | None = None
    version: int = 0
    model: str = "LogisticRegression"
    params: dict[str, Any] = field(default_factory=dict)
    scale: bool = False
    split_seed: int = DEFAULT_SEED
    trained: bool = False
    job: int = 0
    view: dict[str, float] | None = None
    probe_id: int | None = None
    prep_cache: dict[tuple[Any, ...], Prepared] = field(default_factory=dict)
    lang: str = "en"
    finished_job: int = -1


class Hub:
    def __init__(self, log_path: str | None) -> None:
        self.s = Session()
        self.clients: dict[WebSocket, asyncio.Queue[bytes | None]] = {}
        self.loop: asyncio.AbstractEventLoop | None = None
        self.log_path = log_path
        self.worker: WorkerHandle | None = None
        self.t_start = time.perf_counter()
        # model schemas come from the worker, which imports scikit-learn; this process does not
        self.models: dict[str, dict[str, Any]] = {}
        self.sklearn = ""
        self.models_ready: asyncio.Event | None = None

    def start_worker(self) -> None:
        if self.worker is None:
            self.worker = WorkerHandle(self.from_worker, self.log_path)

    # ---------------------------------------------------------------- outbound

    def from_worker(self, msg: dict[str, Any]) -> None:
        if msg.get("type") == "frame" and msg.get("job") != self.s.job:
            return
        if msg.get("type") == "done":
            self.s.finished_job = int(msg.get("job", -1))
        if msg.get("type") == "worker_ready":
            if not self.models:
                self.models = {m["name"]: m for m in msg["schemas"]}
                self.sklearn = str(msg["sklearn"])
                if self.loop is not None and self.models_ready is not None:
                    self.loop.call_soon_threadsafe(self.models_ready.set)
            msg = {"type": "status", "state": "idle", "job": None, "message": "worker_ready",
                   "engine_ms": (time.perf_counter() - self.t_start) * 1000}
        data = pack(msg)
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._broadcast_bytes, data, msg.get("type") == "frame")

    def _broadcast_bytes(self, data: bytes, droppable: bool = False) -> None:
        for q in self.clients.values():
            if droppable and q.qsize() > 6:
                continue  # client is behind; skip a frame rather than build a backlog
            q.put_nowait(data)

    def broadcast(self, msg: dict[str, Any]) -> None:
        self._broadcast_bytes(pack(msg))

    async def send(self, ws: WebSocket, msg: dict[str, Any]) -> None:
        q = self.clients.get(ws)
        if q is not None:
            q.put_nowait(pack(msg))

    # ---------------------------------------------------------------- helpers

    @property
    def schema(self) -> dict[str, Any]:
        return self.models[self.s.model]

    def params(self) -> dict[str, Any]:
        sc = self.schema
        return {**{p["name"]: p["initial"] for p in sc["main"] + sc["advanced"]}, **self.s.params}

    def prep(self) -> Prepared:
        s = self.s
        assert s.dataset is not None
        task = self.schema["task"]
        key = (s.version, task, s.scale, s.split_seed)
        if key not in s.prep_cache:
            s.prep_cache = {key: prepare(s.dataset, task, s.scale, s.split_seed)}
        return s.prep_cache[key]

    def compat(self) -> dict[str, Any]:
        assert self.s.dataset is not None
        return dsv.compat(self.s.dataset, self.s.model, self.schema["task"])

    def dataset_msg(self, rid: int | None = None) -> dict[str, Any]:
        s = self.s
        ds = s.dataset
        assert ds is not None
        c = self.compat()
        base: dict[str, Any] = {
            "type": "dataset", "name": ds.name, "kind": ds.kind, "preset": ds.preset, "settings": ds.settings,
            "n_rows": len(ds.df), "n_features": len(ds.feature_columns), "feature_names": [c_.name for c_ in ds.feature_columns],
            "target": ds.target, "task": ds.task, "classes": ds.classes(), "compat": c, "table": ds.table(),
            "scale": s.scale, "split_seed": s.split_seed, "rid": rid, "version": s.version, "model": s.model,
        }
        try:
            p = self.prep()
            label = p.y.astype(np.float32) if p.y is not None else np.zeros(len(p.ids), np.float32)
            if p.y is None and ds.task == "classification" and ds.target is not None:
                lut = {cl: i for i, cl in enumerate(ds.classes())}
                label = np.array([lut.get(str(v), 0) for v in ds.df[ds.target]], np.float32)
            test = np.zeros(len(p.ids), np.uint8)
            test[p.test_idx] = 1
            base.update(plane=p.plane.to_dict(), ids=p.ids.astype(np.int32), xy=p.xy.astype(np.float32), label=label,
                        test=test, label_kind="value" if (p.task == "regression") else "class",
                        encoded=p.feature_names, n_train=int(len(p.train_idx)), n_test=int(len(p.test_idx)))
        except Exception as e:  # noqa: BLE001 - incompatible data still shows its table
            log.info("prepare failed: %s", e)
            base.update(plane=None, ids=np.zeros(0, np.int32), xy=np.zeros((0, 2), np.float32),
                        label=np.zeros(0, np.float32), test=np.zeros(0, np.uint8), label_kind="class",
                        encoded=[], n_train=0, n_test=0)
        return base

    def status(self, state: str, **kw: Any) -> None:
        self.broadcast({"type": "status", "state": state, "job": self.s.job, **kw})

    def new_dataset(self, ds: dsv.Dataset, rid: int | None = None) -> None:
        self.cancel_job()
        self.s.dataset = ds
        self.s.version += 1
        self.s.trained = False
        self.s.view = None
        self.status("idle")  # a paused job is gone with its data: the UI must not offer "Continue"
        self.broadcast(self.dataset_msg(rid))

    def cancel_job(self) -> None:
        self.s.job += 1
        if self.worker is not None:
            self.worker.cancel()
            self.worker.send("reset")

    def start_job(self, fast: bool = False, paused: bool = False, rid: int | None = None) -> None:
        s = self.s
        if s.dataset is None:
            self.broadcast(friendly("no_dataset", None, rid))
            return
        c = self.compat()
        if not c["ok"]:
            self.broadcast(friendly("incompatible", None, rid))
            return
        self.start_worker()
        assert self.worker is not None
        s.job += 1
        params = self.params()
        seed = params.get("random_state")
        job = Job(s.job, s.model, params, self.prep(), int(seed) if isinstance(seed, (int, float)) else DEFAULT_SEED,
                  s.view, fast, s.probe_id)
        s.trained = True
        self.worker.submit(job, paused)
        self.status("paused" if paused else "training", queued=not self.worker.ready)

    # ---------------------------------------------------------------- inbound

    async def handle(self, ws: WebSocket, raw: dict[str, Any]) -> None:
        msg: ClientMessage
        try:
            msg = CLIENT.validate_python(raw)
        except ValidationError as e:
            await self.send(ws, friendly("bad_message", e, raw.get("rid") if isinstance(raw, dict) else None))
            return
        t = msg.type
        s = self.s
        rid = getattr(msg, "rid", None)
        try:
            if isinstance(msg, Hello):
                s.lang = msg.lang
            elif isinstance(msg, LoadDataset):
                self.new_dataset(self.build_dataset(msg.spec), rid)
            elif isinstance(msg, PreviewPreset):
                X, y = dsv.make_preset_arrays(msg.preset, msg.settings)
                n = min(len(X), 600)
                xy = np.c_[X[:n, 0], y[:n] if X.shape[1] == 1 else X[:n, 1]].astype(np.float32)
                lab = np.zeros(n, np.float32) if X.shape[1] == 1 else y[:n].astype(np.float32)
                extra: dict[str, Any] = {"n": int(len(X))}
                if X.shape[1] == 1:  # dashed trend line on the regression thumbnail
                    b, a = np.polyfit(X[:, 0], y, 1)
                    x0, x1 = float(X[:, 0].min()), float(X[:, 0].max())
                    extra["line"] = [x0, float(a + b * x0), x1, float(a + b * x1)]
                await self.send(ws, {"type": "preview", "preset": msg.preset, "xy": xy, "label": lab, "rid": rid, "extra": extra})
            elif isinstance(msg, PreviewGenerator):
                ds = dsv.generate(msg.generator)
                await self.send(ws, self.generator_preview(ds, rid))
            elif isinstance(msg, ParseCsv):
                tbl = dsv.parse_csv(msg.csv)
                await self.send(ws, {"type": "table", "table": tbl, "rid": rid})
            elif isinstance(msg, SetModel):
                self.cancel_job()
                s.model = msg.model if msg.model in self.models else s.model
                s.params = dict(msg.params)
                s.trained = False
                s.view = None
                self.status("idle")
                if s.dataset is not None:
                    self.broadcast(self.dataset_msg())
            elif isinstance(msg, SetParams):
                s.params.update(msg.params)
                if msg.retrain and s.dataset is not None and s.trained:
                    self.start_job()
            elif isinstance(msg, Train):
                self.start_job(fast=msg.fast, paused=msg.paused)
            elif isinstance(msg, Pause):
                if self.worker:
                    self.worker.pause(True)
                self.status("paused")
            elif isinstance(msg, Resume):
                if self.worker is not None and s.trained and not self._job_finished():
                    self.worker.pause(False)
                    self.status("training")
                else:  # nothing left to resume (data or model changed): train from scratch
                    self.start_job()
            elif isinstance(msg, Step):
                if self.worker is not None and s.trained and not self._job_finished():
                    self.worker.step()
                else:
                    self.start_job(paused=True)
            elif isinstance(msg, Reset):
                self.cancel_job()
                s.trained = False
                self.status("idle", reset=True)
            elif isinstance(msg, Cancel):
                self.cancel_job()
                self.status("idle", cancelled=True)
            elif isinstance(msg, SetSpeed):
                self.start_worker()
                assert self.worker is not None
                self.worker.speed(msg.speed)
            elif isinstance(msg, SetView):
                s.view = msg.view.model_dump() if msg.view else None
                if self.worker:
                    self.worker.send("view", s.view)
            elif isinstance(msg, Explain):
                if self.worker is None or not s.trained:
                    await self.send(ws, friendly("not_trained", None, rid))
                else:
                    self.worker.send("explain", msg.point_id, rid)
            elif isinstance(msg, Probe):
                s.probe_id = msg.point_id
                if self.worker:
                    self.worker.send("probe", msg.point_id)
            elif isinstance(msg, EditPoints):
                self.edit_points(msg.add, msg.move, msg.delete, msg.retrain)
            elif isinstance(msg, DatasetStats):
                if s.dataset is None:
                    await self.send(ws, friendly("no_dataset", None, rid))
                else:
                    await self.send(ws, {"type": "stats", "stats": dsv.stats(s.dataset), "rid": rid})
            elif isinstance(msg, RemoveRows):
                if s.dataset is not None:
                    ds = s.dataset
                    ds.df = ds.df.drop(index=[i for i in msg.ids if i in ds.df.index])
                    self.data_changed(retrain=s.trained)
            elif isinstance(msg, SetScale):
                s.scale = msg.scale
                self.data_changed(retrain=s.trained, bump=False)
            elif isinstance(msg, SetSplitSeed):
                s.split_seed = int(msg.seed)
                self.data_changed(retrain=s.trained, bump=False)
        except ValueError as e:
            code = str(e) if str(e) in ("duplicate_columns", "empty_csv") else ("bad_csv" if t == "parse_csv" else "")
            await self.send(ws, friendly(code or "invalid", e, rid))
        except Exception as e:  # noqa: BLE001
            await self.send(ws, friendly("bad_csv" if t == "parse_csv" else "engine", e, rid))

    def _job_finished(self) -> bool:
        return self.s.finished_job == self.s.job

    def build_dataset(self, spec: Any) -> dsv.Dataset:
        if spec.kind == "preset":
            return dsv.dataset_from_preset(spec.preset or "moons", spec.settings)
        if spec.kind == "generator":
            return dsv.generate(spec.generator or {})
        if spec.kind == "csv":
            return dsv.dataset_from_table(dsv.parse_csv(spec.csv or ""), spec.name)
        return dsv.dataset_from_table(spec.table or {"columns": [], "rows": []}, spec.name)

    def generator_preview(self, ds: dsv.Dataset, rid: int | None) -> dict[str, Any]:
        nums = [c.name for c in ds.feature_columns if c.type == "numeric"]
        n = len(ds.df)
        if len(nums) >= 2:
            xy = ds.df[nums[:2]].to_numpy(np.float32)
        elif nums:
            xy = np.c_[ds.df[nums[0]].to_numpy(np.float32), np.zeros(n, np.float32)]
        else:
            xy = np.zeros((n, 2), np.float32)
        classes = ds.classes()
        lut = {c: i for i, c in enumerate(classes)}
        lab = np.array([lut.get(str(v), 0) for v in ds.df[ds.target]], np.float32) if ds.target else np.zeros(n, np.float32)
        tbl = ds.table()
        bal = [int((lab == i).sum()) for i in range(len(classes))]
        return {"type": "preview", "preset": None, "xy": xy[:2000], "label": lab[:2000], "rows": tbl["rows"][:5],
                "columns": [c["name"] for c in tbl["columns"]], "classes": classes, "rid": rid,
                "extra": {"n": n, "balance": bal, "axes": nums[:2]}}

    def data_changed(self, retrain: bool, bump: bool = True) -> None:
        s = self.s
        if bump:
            s.version += 1
        self.broadcast(self.dataset_msg())
        if retrain and s.dataset is not None and self.compat()["ok"]:
            self.start_job(fast=True)

    def edit_points(self, add: list[list[float]], move: list[list[float]], delete: list[int], retrain: bool) -> None:
        s = self.s
        ds = s.dataset
        if ds is None:
            return
        p = self.prep()
        if not p.plane.editable:
            return
        df = ds.df
        feats = [c.name for c in ds.feature_columns]
        mode = p.plane.mode
        if delete:
            df = df.drop(index=[i for i in delete if i in df.index])
        for pid, x, y in move:
            pid = int(pid)
            if pid not in df.index:
                continue
            if mode == "xy":
                df.at[pid, feats[0]] = float(x)
                if ds.target:
                    df.at[pid, ds.target] = float(y)
            else:
                df.at[pid, feats[0]] = float(x)
                df.at[pid, feats[1]] = float(y)
        next_id = int(df.index.max()) + 1 if len(df) else 0
        classes = ds.classes()
        for row in add:
            x, y = float(row[0]), float(row[1])
            lab = int(row[2]) if len(row) > 2 else 0
            rec: dict[str, Any] = {c: None for c in df.columns}
            if mode == "xy":
                rec[feats[0]] = x
                if ds.target:
                    rec[ds.target] = y
            else:
                rec[feats[0]], rec[feats[1]] = x, y
                if ds.target and ds.task == "classification":
                    names = classes if classes else dsv.CLASS_NAMES
                    rec[ds.target] = names[lab] if lab < len(names) else dsv.CLASS_NAMES[min(lab, 4)]
            df.loc[next_id] = rec
            next_id += 1
        ds.df = df
        self.data_changed(retrain=retrain and s.trained)


def create_app(token: str, log_path: str | None) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    hub = Hub(log_path)
    app.state.hub = hub

    @app.on_event("startup")
    async def _startup() -> None:
        hub.loop = asyncio.get_running_loop()
        hub.models_ready = asyncio.Event()
        hub.start_worker()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if hub.worker:
            hub.worker.shutdown()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "worker": bool(hub.worker and hub.worker.ready)}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        tok = ws.query_params.get("token", "")
        if not hmac.compare_digest(tok, token):
            await ws.close(code=4401)
            return
        await ws.accept()
        q: asyncio.Queue[bytes | None] = asyncio.Queue()
        hub.clients[ws] = q

        async def sender() -> None:
            while True:
                b = await q.get()
                if b is None:
                    return
                await ws.send_bytes(b)

        task = asyncio.create_task(sender())
        if hub.models_ready is not None:
            await hub.models_ready.wait()
        await hub.send(ws, {"type": "ready", "sklearn": hub.sklearn, "models": list(hub.models.values()),
                            "presets": dsv.preset_catalog(),
                            "worker": bool(hub.worker and hub.worker.ready),
                            "session": {"model": hub.s.model, "params": hub.s.params, "trained": hub.s.trained}})
        if hub.s.dataset is not None:
            await hub.send(ws, hub.dataset_msg())
        try:
            while True:
                data = await ws.receive_bytes()
                try:
                    raw = unpack(data)
                except Exception as e:  # noqa: BLE001
                    await hub.send(ws, friendly("bad_message", e))
                    continue
                await hub.handle(ws, raw)
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            pass
        finally:
            hub.clients.pop(ws, None)
            q.put_nowait(None)
            task.cancel()

    return app
