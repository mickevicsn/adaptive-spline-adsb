from __future__ import annotations

import webbrowser
from pathlib import Path

from src.adsb_viewer import ViewerConfig, create_dash_app

# ============================================================
# EDIT THESE ONLY
# ============================================================

TRACK_OUTPUT_DIR = Path("track_output")

HOST = "127.0.0.1"
PORT = 8050
DEBUG = True

TITLE = "ADS-B 3D Flight Viewer"

# ============================================================
# MAIN
# ============================================================

def main() -> None:
    config = ViewerConfig(title=TITLE)
    app = create_dash_app(config=config, track_output_dir=TRACK_OUTPUT_DIR)

    url = f"http://{HOST}:{PORT}"
    print(f"Open: {url}")
    print(f"Flight source folder: {TRACK_OUTPUT_DIR.resolve()}")
    webbrowser.open(url)

    app.run(host=HOST, port=PORT, debug=DEBUG)


if __name__ == "__main__":
    main()
