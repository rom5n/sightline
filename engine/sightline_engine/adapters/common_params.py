"""Parameter definitions shared by several adapters."""
from __future__ import annotations

from typing import Any

from .base import P

SEED_HINT = ("Fixes the randomness so every run gives the same result. None = Sightline picks a fixed seed per run.",
             "Фиксирует случайность, чтобы каждый запуск давал одинаковый результат. None — Sightline выбирает зерно на каждый запуск.")


def random_state(section: str = "repro") -> P:
    return P("random_state", "mixed", None, ("Random seed", "Зерно случайности"), SEED_HINT,
             lo=0, hi=9999, step=1, options=[None], section=section)


def fit_intercept(default: bool = True) -> P:
    return P("fit_intercept", "bool", default, ("Fit intercept", "Свободный член"),
             ("Lets the line or border shift away from the origin (0, 0). Almost always on.",
              "Позволяет линии или границе не проходить через начало координат (0, 0). Почти всегда включено."))


def tol(default: float, section: str = "stop") -> P:
    return P("tol", "float", default, ("Tolerance", "Допуск"),
             ("Stop when an improvement is smaller than this.",
              "Остановиться, когда улучшение меньше этого числа."),
             lo=1e-10, hi=1.0, log=True, section=section)


def copy(name: str) -> P:
    return P(name, "bool", True, ("Copy input data", "Копировать данные"),
             ("Only affects memory use, never the result.", "Влияет только на расход памяти, не на результат."),
             section="other", disabled=("Sightline always copies the data", "Sightline всегда копирует данные"))


def warm_start(section: str = "repro") -> P:
    return P("warm_start", "bool", False, ("Warm start", "Тёплый старт"),
             ("Continue from the previous fit instead of starting over.",
              "Продолжать с прошлого обучения, а не начинать заново."),
             section=section,
             disabled=("Sightline uses it itself to show training step by step",
                       "Sightline использует его сам, чтобы показывать обучение по шагам"))


def class_weight(section: str | None = None) -> P:
    return P("class_weight", "enum", None, ("Class weight", "Веса классов"),
             ("'balanced' counts rare classes more, so the model can't ignore them.",
              "«balanced» усиливает редкие классы, чтобы модель их не игнорировала."),
             options=[None, "balanced"], section=section)


def positive(section: str | None = None) -> P:
    return P("positive", "bool", False, ("Only positive weights", "Только положительные веса"),
             ("Forces every weight to be ≥ 0. Useful when a feature can only push the prediction up.",
              "Все веса будут ≥ 0. Полезно, когда признак может только увеличивать прогноз."),
             section=section)


def max_iter(default: int, lo: int = 1, hi: int = 1000, section: str | None = None,
             label: tuple[str, str] = ("Max iterations", "Макс. итераций"),
             hint: tuple[str, str] = ("How many optimisation steps it may take. Stops early when the border stops moving.",
                                      "Сколько шагов оптимизации разрешено. Обучение остановится раньше, если граница перестанет двигаться.")) -> P:
    return P("max_iter", "int", default, label, hint, lo=lo, hi=hi, step=1, section=section)


TREE_CRITERION_HINT = ("How a split is scored. The three usually give very similar trees.",
                       "Как оценивается разбиение. Все три обычно дают очень похожие деревья.")


def criterion() -> P:
    return P("criterion", "enum", "gini", ("Split quality", "Качество разбиения"), TREE_CRITERION_HINT,
             options=["gini", "entropy", "log_loss"])


def max_depth(default: Any = None, initial: Any = None, hi: int = 20, main: bool = True) -> P:
    p = P("max_depth", "int", default, ("Max depth", "Макс. глубина"),
          ("How many questions in a row it may ask. Deeper → finer borders, but it starts memorising noise.",
           "Сколько вопросов подряд можно задать. Глубже — тоньше границы, но дерево начинает запоминать шум."),
          lo=1, hi=hi, step=1, none_label=("No limit (None, the default)", "Без ограничения (None, по умолчанию)"))
    if initial is not None:
        p.initial = initial
    return p


def min_samples_split(default: Any = 2, section: str | None = None) -> P:
    return P("min_samples_split", "int", default, ("Min. to split", "Мин. для разбиения"),
             ("A node needs at least this many points before it may be split.",
              "Узел можно разбить, только если в нём не меньше столько точек."),
             lo=2, hi=50, step=1, section=section)


def min_samples_leaf(default: Any = 1, section: str | None = None, hi: int = 50) -> P:
    return P("min_samples_leaf", "int", default, ("Min. per leaf", "Мин. в листе"),
             ("Every leaf must keep at least this many points. Raise it to smooth the borders.",
              "В каждом листе должно остаться не меньше столько точек. Увеличьте, чтобы сгладить границы."),
             lo=1, hi=hi, step=1, section=section)


def tree_shape_advanced(include_splitter: bool, include_max_features: bool, include_criterion: bool = False,
                        include_split_leaf: bool = False) -> list[P]:
    out: list[P] = []
    if include_criterion:
        out.append(P("criterion", "enum", "gini", ("Split quality", "Качество разбиения"), TREE_CRITERION_HINT,
                     options=["gini", "entropy", "log_loss"], section="shape"))
    if include_split_leaf:
        out.append(min_samples_split(section="shape"))
    if include_splitter:
        out.append(P("splitter", "enum", "best", ("Split choice", "Выбор разбиения"),
                     ("'best' tries every cut; 'random' picks the best of random cuts — faster, rougher.",
                      "«best» перебирает все разрезы; «random» берёт лучший из случайных — быстрее, но грубее."),
                     options=["best", "random"], section="shape"))
    if include_max_features:
        out.append(P("max_features", "mixed", None, ("Features per split", "Признаков на разбиение"),
                     ("How many features each split may look at. None = all of them.",
                      "Сколько признаков смотреть при каждом разбиении. None — все."),
                     options=[None, "sqrt", "log2"], lo=0.05, hi=1.0, step=0.05, section="shape"))
    out += [
        P("max_leaf_nodes", "int", None, ("Max leaves", "Макс. листьев"),
          ("Upper limit on the number of leaves. None = no limit.", "Верхний предел числа листьев. None — без предела."),
          lo=2, hi=200, step=1, none_label=("No limit", "Без предела"), section="shape"),
        P("min_impurity_decrease", "float", 0.0, ("Min. impurity drop", "Мин. снижение неоднородности"),
          ("A split must reduce impurity by at least this much.", "Разбиение должно снижать неоднородность хотя бы на столько."),
          lo=0.0, hi=0.2, step=0.001, section="shape"),
        P("min_weight_fraction_leaf", "float", 0.0, ("Min. leaf weight share", "Мин. доля веса в листе"),
          ("Like min per leaf, but as a share of all points.", "Как «мин. в листе», но в долях от всех точек."),
          lo=0.0, hi=0.5, step=0.01, section="shape"),
        P("ccp_alpha", "float", 0.0, ("Pruning strength", "Сила обрезки"),
          ("Cuts back branches that help too little. 0 = no pruning.",
           "Срезает ветви, которые помогают слишком мало. 0 — без обрезки."),
          lo=0.0, hi=0.05, step=0.0005, section="shape"),
        P("monotonic_cst", "fixed", None, ("Monotonic constraints", "Монотонные ограничения"),
          ("Forces predictions to only rise (or fall) with a feature.",
           "Заставляет прогноз только расти (или падать) с признаком."),
          section="shape", disabled=("Needs one value per feature; not editable here",
                                     "Нужно значение для каждого признака; здесь не редактируется")),
    ]
    return out
