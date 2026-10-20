"""Wire protocol. Pydantic models are the single source of truth; `scripts/gen_ts.py`
turns them into `src/engine/protocol.gen.ts`.

Transport: msgpack over a WebSocket. Numeric arrays travel as `NDArray`
({"$nd": dtype, "shape": [...], "data": <bytes>}); the UI worker turns them into
typed arrays (Float32Array, Int32Array, Uint8Array) without copying element by element.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ParamValue = Any  # bool | int | float | str | None | list[int] | list[float]


class NDArray(BaseModel):
    nd: Literal["f4", "i4", "u1"] = Field(alias="$nd")
    shape: list[int]
    data: bytes


# ---------------------------------------------------------------- client -> engine


class DatasetSpec(BaseModel):
    kind: Literal["preset", "generator", "table", "csv"]
    preset: str | None = None
    settings: dict[str, float] = {}
    generator: dict[str, Any] | None = None
    table: dict[str, Any] | None = None  # {columns:[{name,type,catKind}], rows:[[...]], target}
    csv: str | None = None
    name: str | None = None


class Hello(BaseModel):
    type: Literal["hello"]
    lang: Literal["en", "ru"] = "en"


class LoadDataset(BaseModel):
    type: Literal["load_dataset"]
    spec: DatasetSpec
    rid: int | None = None


class PreviewPreset(BaseModel):
    type: Literal["preview_preset"]
    preset: str
    settings: dict[str, float] = {}
    rid: int | None = None


class PreviewGenerator(BaseModel):
    type: Literal["preview_generator"]
    generator: dict[str, Any]
    rid: int | None = None


class ParseCsv(BaseModel):
    type: Literal["parse_csv"]
    csv: str
    rid: int | None = None


class SetModel(BaseModel):
    type: Literal["set_model"]
    model: str
    params: dict[str, ParamValue] = {}


class SetParams(BaseModel):
    type: Literal["set_params"]
    params: dict[str, ParamValue]
    retrain: bool = False


class Train(BaseModel):
    type: Literal["train"]
    fast: bool = False  # run at max speed (live retrain after a data edit)
    paused: bool = False


class Pause(BaseModel):
    type: Literal["pause"]


class Resume(BaseModel):
    type: Literal["resume"]


class Step(BaseModel):
    type: Literal["step"]


class Reset(BaseModel):
    type: Literal["reset"]


class Cancel(BaseModel):
    type: Literal["cancel"]


class SetSpeed(BaseModel):
    type: Literal["set_speed"]
    speed: float


class ViewRect(BaseModel):
    x0: float
    x1: float
    y0: float
    y1: float
    w: int = 160
    h: int = 100


class SetView(BaseModel):
    type: Literal["set_view"]
    view: ViewRect | None


class Explain(BaseModel):
    type: Literal["explain"]
    point_id: int
    rid: int | None = None


class Probe(BaseModel):
    type: Literal["probe"]
    point_id: int | None


class EditPoints(BaseModel):
    type: Literal["edit_points"]
    add: list[list[float]] = []  # [x, y, label] in plane coords
    move: list[list[float]] = []  # [id, x, y]
    delete: list[int] = []
    retrain: bool = True


class DatasetStats(BaseModel):
    type: Literal["dataset_stats"]
    rid: int | None = None


class RemoveRows(BaseModel):
    type: Literal["remove_rows"]
    ids: list[int]


class SetScale(BaseModel):
    type: Literal["set_scale"]
    scale: bool


class SetSplitSeed(BaseModel):
    type: Literal["set_split_seed"]
    seed: int


ClientMessage = (
    Hello | LoadDataset | PreviewPreset | PreviewGenerator | ParseCsv | SetModel | SetParams | Train
    | Pause | Resume | Step | Reset | Cancel | SetSpeed | SetView | Explain | Probe | EditPoints
    | DatasetStats | RemoveRows | SetScale | SetSplitSeed
)


# ---------------------------------------------------------------- engine -> client


class Ready(BaseModel):
    type: Literal["ready"]
    sklearn: str
    models: list[dict[str, Any]]  # schema per model
    presets: list[dict[str, Any]]
    worker: bool
    session: dict[str, Any]


class DatasetInfo(BaseModel):
    """Points in plane coordinates plus everything the UI needs to draw them."""

    type: Literal["dataset"]
    name: str
    kind: str
    preset: str | None
    settings: dict[str, float]
    n_rows: int
    n_features: int
    feature_names: list[str]
    target: str | None
    task: Literal["classification", "regression", "none"]
    classes: list[str]
    plane: dict[str, Any]  # {mode: raw|pca|xy, axes:[str,str], bounds:[x0,x1,y0,y1], editable}
    ids: Any  # NDArray i4
    xy: Any  # NDArray f4 (n,2)
    label: Any  # NDArray f4 (class index or target value)
    test: Any  # NDArray u1
    compat: dict[str, Any]
    table: dict[str, Any]
    scale: bool
    split_seed: int
    label_kind: Literal["class", "value"]
    encoded: list[str]
    n_train: int
    n_test: int
    version: int
    model: str
    rid: int | None = None


class Frame(BaseModel):
    type: Literal["frame"]
    job: int
    iteration: int
    total: int
    loss: float | None
    loss_test: float | None
    metric_train: float | None
    metric_test: float | None
    extra: dict[str, Any]
    viz: dict[str, Any]
    final: bool = False
    viz_ms: float
    t: float


class Done(BaseModel):
    type: Literal["done"]
    job: int
    reason: Literal["converged", "max_iter", "fitted", "cancelled", "early_stop"]
    iteration: int
    seconds: float
    summary: dict[str, Any]


class Status(BaseModel):
    type: Literal["status"]
    state: Literal["idle", "training", "paused", "fitting", "starting"]
    job: int | None
    progress: float | None = None
    message: str | None = None
    iteration: int | None = None
    queued: bool | None = None
    reset: bool | None = None
    cancelled: bool | None = None
    engine_ms: float | None = None


class Error(BaseModel):
    type: Literal["error"]
    code: str
    message: dict[str, str]  # {"en": ..., "ru": ...}
    rid: int | None = None


class Stats(BaseModel):
    type: Literal["stats"]
    stats: dict[str, Any]
    rid: int | None = None


class ExplainSteps(BaseModel):
    type: Literal["explain_steps"]
    point_id: int
    family: str
    data: dict[str, Any]
    rid: int | None = None


class Preview(BaseModel):
    type: Literal["preview"]
    preset: str | None
    xy: Any
    label: Any
    rows: list[list[Any]] = []
    columns: list[str] = []
    classes: list[str] = []
    extra: dict[str, Any] = {}
    rid: int | None = None


class Table(BaseModel):
    type: Literal["table"]
    table: dict[str, Any]
    rid: int | None = None


class ProbeResult(BaseModel):
    type: Literal["probe"]
    point_id: int
    data: dict[str, Any]


EngineMessage = Ready | DatasetInfo | Frame | Done | Status | Error | Stats | ExplainSteps | Preview | Table | ProbeResult
