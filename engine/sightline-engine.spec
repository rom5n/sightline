# PyInstaller one-folder build of the Sightline engine (no Python needed on the target machine).
# Build: npm run build:engine
import sys

from PyInstaller.utils.hooks import collect_submodules

# the executable icon (Linux executables carry none)
ICON = {"win32": "../src-tauri/icons/icon.ico", "darwin": "../src-tauri/icons/icon.icns"}.get(sys.platform)

hidden = (
    collect_submodules("uvicorn")
    + collect_submodules("sightline_engine")
    + collect_submodules("sklearn.utils")
    + ["websockets", "websockets.legacy", "websockets.legacy.server", "msgpack", "sklearn.neighbors._partition_nodes",
       "sklearn.tree._utils", "sklearn.metrics._pairwise_distances_reduction", "scipy.special.cython_special"]
)

a = Analysis(
    ["run_engine.py"],
    pathex=["."],
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "plotly", "graphviz", "ipywidgets", "IPython", "pytest", "mypy", "PyInstaller", "pandas.tests", "numpy.tests",
              "scipy.tests", "sklearn.tests", "sklearn.datasets.tests", "PIL", "notebook", "jupyter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="sightline-engine", console=False, upx=False,
          icon=ICON)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="sightline-engine")
