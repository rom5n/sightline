"""End to end over the WebSocket with the real worker process."""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sightline_engine.server import create_app
from sightline_engine.wire import pack, unpack

TOKEN = "t3st"


def nd(v: Any) -> np.ndarray:
    dt = {"f4": np.float32, "i4": np.int32, "u1": np.uint8}[v["$nd"]]
    return np.frombuffer(v["data"], dtype=dt).reshape(v["shape"])


class Conn:
    def __init__(self, ws: Any) -> None:
        self.ws = ws

    def send(self, **msg: Any) -> None:
        self.ws.send_bytes(pack(msg))

    def recv(self) -> dict[str, Any]:
        return unpack(self.ws.receive_bytes())

    def until(self, pred: Callable[[dict[str, Any]], bool], limit: int = 2000) -> dict[str, Any]:
        for _ in range(limit):
            m = self.recv()
            if m["type"] == "error" and m.get("code") not in ("not_trained",):
                raise AssertionError(m)
            if pred(m):
                return m
        raise AssertionError("message not received")


@pytest.fixture(scope="module")
def client() -> Any:
    app = create_app(TOKEN, None)
    with TestClient(app) as c:
        yield c


def test_rejects_bad_token(client: Any) -> None:
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws?token=wrong") as ws:
            ws.receive_bytes()


def test_train_pause_step_explain(client: Any) -> None:
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        c = Conn(ws)
        ready = c.until(lambda m: m["type"] == "ready")
        assert len(ready["models"]) == 22
        c.send(type="set_model", model="LogisticRegression", params={})
        c.send(type="load_dataset", spec={"kind": "preset", "preset": "moons", "settings": {"n_samples": 300}}, rid=1)
        ds = c.until(lambda m: m["type"] == "dataset" and m.get("rid") == 1)
        assert ds["compat"]["ok"] and nd(ds["xy"]).shape == (300, 2) and nd(ds["test"]).sum() == 75
        c.send(type="set_speed", speed=4)
        c.send(type="train")
        f = c.until(lambda m: m["type"] == "frame")
        grid = nd(f["viz"]["grid"]["data"])
        assert grid.dtype == np.float32 and grid.shape[0] == 2
        c.send(type="pause")
        st = c.until(lambda m: m["type"] == "status" and m["state"] == "paused" and m.get("iteration") is not None)
        it0 = st["iteration"]
        time.sleep(0.15)
        c.send(type="step")
        f2 = c.until(lambda m: m["type"] == "frame" and m["iteration"] > it0)
        assert f2["iteration"] == it0 + 1
        c.send(type="resume")
        done = c.until(lambda m: m["type"] == "done")
        assert done["reason"] in ("converged", "max_iter")
        pid = int(nd(ds["ids"])[0])
        c.send(type="explain", point_id=pid, rid=7)
        ex = c.until(lambda m: m["type"] == "explain_steps")
        assert ex["data"]["pred"] == ex["data"]["check"]


def test_param_change_latency(client: Any) -> None:
    """Parameter change -> first new frame, for a fast model (target < 100 ms)."""
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        c = Conn(ws)
        c.until(lambda m: m["type"] == "ready")
        c.send(type="set_model", model="LogisticRegression", params={})
        c.send(type="load_dataset", spec={"kind": "preset", "preset": "moons", "settings": {}}, rid=2)
        c.until(lambda m: m["type"] == "dataset" and m.get("rid") == 2)
        c.send(type="train", fast=True)
        c.until(lambda m: m["type"] == "done")
        times = []
        for C in (0.1, 3.0, 10.0, 0.5, 1.0):
            t = time.perf_counter()
            c.send(type="set_params", params={"C": C}, retrain=True)
            c.until(lambda m: m["type"] == "frame")
            times.append((time.perf_counter() - t) * 1000)
            c.until(lambda m: m["type"] == "done")
        med = float(np.median(times))
        print(f"\nparam->first frame ms: {[round(x, 1) for x in times]} median {med:.1f}")
        assert med < 100


def test_cancel_blocking_fit_restarts_worker(client: Any) -> None:
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        c = Conn(ws)
        c.until(lambda m: m["type"] == "ready")
        c.send(type="set_model", model="SVC", params={"C": 100.0, "gamma": 50.0})
        c.send(type="load_dataset", spec={"kind": "preset", "preset": "spirals", "settings": {"n_samples": 2000, "noise": 0.6}}, rid=3)
        c.until(lambda m: m["type"] == "dataset" and m.get("rid") == 3)
        c.send(type="train")
        c.until(lambda m: m["type"] == "status" and m["state"] == "fitting")
        c.send(type="cancel")
        c.until(lambda m: m["type"] == "status" and m["state"] in ("starting", "idle"))
        c.until(lambda m: m["type"] == "status" and m.get("message") == "worker_ready")
        c.send(type="set_model", model="GaussianNB", params={})
        c.send(type="train")
        done = c.until(lambda m: m["type"] == "done")
        assert done["reason"] == "fitted"


def test_edit_points_live_retrain(client: Any) -> None:
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        c = Conn(ws)
        c.until(lambda m: m["type"] == "ready")
        c.send(type="set_model", model="KNeighborsClassifier", params={})
        c.send(type="load_dataset", spec={"kind": "preset", "preset": "blobs", "settings": {}}, rid=4)
        ds = c.until(lambda m: m["type"] == "dataset" and m.get("rid") == 4)
        c.send(type="train")
        c.until(lambda m: m["type"] == "done")
        n = ds["n_rows"]
        c.send(type="edit_points", add=[[0.0, 0.0, 1]], move=[], delete=[int(nd(ds["ids"])[0])])
        ds2 = c.until(lambda m: m["type"] == "dataset")
        assert ds2["n_rows"] == n
        c.until(lambda m: m["type"] == "done")


def test_stats_and_generator(client: Any) -> None:
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        c = Conn(ws)
        c.until(lambda m: m["type"] == "ready")
        gen = {"rows": 300, "seed": 42, "mode": "rules",
               "columns": [{"name": "x1", "type": "numeric", "dist": "normal", "mean": 0, "std": 1},
                           {"name": "x2", "type": "numeric", "dist": "uniform", "min": -3, "max": 3},
                           {"name": "color", "type": "category", "values": [{"v": "red", "w": 40}, {"v": "blue", "w": 60}]}],
               "target": {"name": "label", "source": "rule", "rule": "line", "on": ["x1", "x2"], "noise": 0.1}}
        c.send(type="preview_generator", generator=gen, rid=5)
        pv = c.until(lambda m: m["type"] == "preview" and m.get("rid") == 5)
        assert nd(pv["xy"]).shape == (300, 2) and len(pv["rows"]) == 5
        c.send(type="preview_generator", generator=gen, rid=6)
        pv2 = c.until(lambda m: m["type"] == "preview" and m.get("rid") == 6)
        assert np.array_equal(nd(pv["xy"]), nd(pv2["xy"])), "same seed must give the same data"
        c.send(type="load_dataset", spec={"kind": "generator", "generator": gen}, rid=8)
        c.until(lambda m: m["type"] == "dataset" and m.get("rid") == 8)
        c.send(type="dataset_stats", rid=9)
        st = c.until(lambda m: m["type"] == "stats")
        assert st["stats"]["categorical"][0]["name"] == "color"
        c.send(type="parse_csv", csv="a,b,t\n1,2,x\n3,4,y\n5,6,x\n", rid=10)
        tb = c.until(lambda m: m["type"] == "table")
        assert tb["table"]["target"] == "t" and len(tb["table"]["rows"]) == 3


def test_new_dataset_while_paused_then_resume_trains(client: Any) -> None:
    # pause, load other data, press the (still "Continue") button: training must start afresh
    with client.websocket_connect(f"/ws?token={TOKEN}") as ws:
        c = Conn(ws)
        c.until(lambda m: m["type"] == "ready")
        c.send(type="set_model", model="MLPClassifier", params={})
        c.send(type="load_dataset", spec={"kind": "preset", "preset": "moons", "settings": {}}, rid=1)
        c.until(lambda m: m["type"] == "dataset" and m.get("rid") == 1)
        c.send(type="train")
        c.until(lambda m: m["type"] == "frame")
        c.send(type="pause")
        c.until(lambda m: m["type"] == "status" and m["state"] == "paused")
        c.send(type="load_dataset", spec={"kind": "preset", "preset": "circles", "settings": {}}, rid=2)
        c.until(lambda m: m["type"] == "status" and m["state"] == "idle")
        ds = c.until(lambda m: m["type"] == "dataset" and m.get("rid") == 2)
        c.send(type="resume")
        f = c.until(lambda m: m["type"] == "frame")
        assert f["job"] > 0 and ds["name"]
