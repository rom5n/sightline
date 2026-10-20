"""PyInstaller entry point for the engine sidecar."""
import multiprocessing

from sightline_engine.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
