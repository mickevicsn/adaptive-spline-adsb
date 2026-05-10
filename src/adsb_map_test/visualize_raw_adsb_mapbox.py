# visualize_raw_adsb_mapbox_hardcoded.py

import json
from pathlib import Path

import pandas as pd


# ============================================================
# HARD-CODED SETTINGS — EDIT THESE ONLY
# ============================================================

INPUT_FILE = r"C:\Users\YOUR_NAME\Desktop\adsb_rows.xlsx"
# Or:
# INPUT_FILE = r"C:\Users\YOUR_NAME\Desktop\adsb_rows.csv"

SHEET_NAME = 0

ICAO = "4D22BF"

OUTPUT_DIR = Path(r"C:\Users\YOUR_NAME\Desktop\adsb_map_output")

MAPBOX_TOKEN = "YOUR_MAPBOX_TOKEN"

# Riga Airport approximate map center
RIGA_AIRPORT_LAT = 56.923599
RIGA_AIRPORT_LON = 23.971100

# Your current raw ADS-B shape
POSITION_DF = 17
POSITION_TYPE_CODE = 11


# ============================================================
# DATA LOADING
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "time": "unix_time",
        "timestamp": "unix_time",
        "unix": "unix_time",
        "icao24": "icao",
        "icao_address": "icao",
        "typecode": "type_code",
        "tc": "type_code",
        "alt": "altitude_ft",
        "altitude": "altitude_ft",
        "baro_altitude": "altitude_ft",
        "latitude": "lat",
        "longitude": "lon",
        "long": "lon",
    }

    rename_map = {}

    for col in df.columns:
        original = col
        clean = str(col).strip().lower()
        rename_map[original] = aliases.get(clean, clean)

    return df.rename(columns=rename_map)


def load_input_file() -> pd.DataFrame:
    path = Path(INPUT_FILE)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=SHEET_NAME)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("INPUT_FILE must be .xlsx, .xls, or .csv")

    df = normalize_columns(df)

    print("Loaded columns:")
    for c in df.columns:
        print("  ", c)

    return df


# ============================================================
# ADS-B FILTERING
# ============================================================

def extract_position_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "unix_time",
        "icao",
        "df",
        "type_code",
        "lat",
        "lon",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    d = df.copy()

    d["icao"] = d["icao"].astype(str).str.upper()
    d["unix_time"] = pd.to_numeric(d["unix_time"], errors="coerce")
    d["df"] = pd.to_numeric(d["df"], errors="coerce")
    d["type_code"] = pd.to_numeric(d["type_code"], errors="coerce")
    d["lat"] = pd.to_numeric(d["lat"], errors="coerce")
    d["lon"] = pd.to_numeric(d["lon"], errors="coerce")

    if "altitude_ft" in d.columns:
        d["altitude_ft"] = pd.to_numeric(d["altitude_ft"], errors="coerce")
    else:
        d["altitude_ft"] = pd.NA

    pos = d[
        (d["icao"] == ICAO.upper())
        & (d["df"] == POSITION_DF)
        & (d["type_code"] == POSITION_TYPE_CODE)
        & d["unix_time"].notna()
        & d["lat"].notna()
        & d["lon"].notna()
    ].copy()

    pos = pos[
        pos["lat"].between(-90, 90)
        & pos["lon"].between(-180, 180)
    ].copy()

    if "row_id" in pos.columns:
        pos = pos.sort_values(["unix_time", "row_id"])
    else:
        pos = pos.sort_values(["unix_time"])

    # Remove exact duplicate decoded position messages
    pos = pos.drop_duplicates(
        subset=["unix_time", "lat", "lon", "altitude_ft"],
        keep="first",
    ).copy()

    return pos


# ============================================================
# GEOJSON EXPORT
# ============================================================

def altitude_ft_to_m(value):
    if pd.isna(value):
        return 0.0
    return float(value) * 0.3048


def make_points_geojson(pos: pd.DataFrame) -> dict:
    features = []

    for _, r in pos.iterrows():
        alt_ft = r.get("altitude_ft", pd.NA)
        alt_m = altitude_ft_to_m(alt_ft)

        props = {
            "icao": str(r["icao"]),
            "unix_time": float(r["unix_time"]),
            "altitude_ft": None if pd.isna(alt_ft) else float(alt_ft),
            "altitude_m": alt_m,
        }

        if "row_id" in pos.columns and pd.notna(r.get("row_id")):
            try:
                props["row_id"] = int(r["row_id"])
            except Exception:
                props["row_id"] = str(r["row_id"])

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                # GeoJSON order is lon, lat, altitude
                "coordinates": [
                    float(r["lon"]),
                    float(r["lat"]),
                    alt_m,
                ],
            },
            "properties": props,
        })

    return {
        "type": "FeatureCollection",
        "features": features,
    }


def make_track_geojson(pos: pd.DataFrame) -> dict:
    coordinates = []

    for _, r in pos.iterrows():
        alt_m = altitude_ft_to_m(r.get("altitude_ft", pd.NA))

        coordinates.append([
            float(r["lon"]),
            float(r["lat"]),
            alt_m,
        ])

    return {
        "type": "Feature",
        "properties": {
            "name": f"raw_adsb_track_{ICAO}",
            "icao": ICAO,
            "count": len(coordinates),
        },
        "geometry": {
            "type": "LineString",
            "coordinates": coordinates,
        },
    }


# ============================================================
# MAPBOX HTML EXPORT
# ============================================================

def write_html():
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Raw ADS-B Mapbox Viewer - {ICAO}</title>
  <meta name="viewport" content="initial-scale=1,maximum-scale=1,user-scalable=no" />

  <script src="https://api.mapbox.com/mapbox-gl-js/v3.20.0/mapbox-gl.js"></script>
  <link href="https://api.mapbox.com/mapbox-gl-js/v3.20.0/mapbox-gl.css" rel="stylesheet" />

  <style>
    body {{
      margin: 0;
      padding: 0;
      font-family: Arial, sans-serif;
    }}

    #map {{
      position: absolute;
      inset: 0;
    }}

    #panel {{
      position: absolute;
      top: 12px;
      left: 12px;
      z-index: 10;
      background: rgba(255, 255, 255, 0.94);
      padding: 12px;
      border-radius: 10px;
      max-width: 360px;
      box-shadow: 0 2px 12px rgba(0,0,0,0.25);
      font-size: 13px;
    }}

    #panel h3 {{
      margin: 0 0 8px 0;
      font-size: 16px;
    }}

    button {{
      margin-top: 8px;
      margin-right: 4px;
      padding: 6px 10px;
      cursor: pointer;
    }}
  </style>
</head>

<body>
<div id="map"></div>

<div id="panel">
  <h3>Raw ADS-B viewer: {ICAO}</h3>
  <div><b>Blue line</b> = raw DF17/type_code 11 position sequence</div>
  <div><b>Red dots</b> = raw decoded position messages</div>
  <div><b>Center</b> = Riga Airport</div>
  <button id="fit">Fit to track</button>
  <button id="riga">Back to Riga Airport</button>
</div>

<script>
mapboxgl.accessToken = "{MAPBOX_TOKEN}";

const RIGA_AIRPORT = [{RIGA_AIRPORT_LON}, {RIGA_AIRPORT_LAT}];

const map = new mapboxgl.Map({{
  container: "map",
  style: "mapbox://styles/mapbox/satellite-streets-v12",
  center: RIGA_AIRPORT,
  zoom: 13.5,
  pitch: 60,
  bearing: -25,
  antialias: true
}});

map.addControl(new mapboxgl.NavigationControl());

function computeBoundsFromPoints(featureCollection) {{
  const bounds = new mapboxgl.LngLatBounds();

  featureCollection.features.forEach(f => {{
    const c = f.geometry.coordinates;
    bounds.extend([c[0], c[1]]);
  }});

  return bounds;
}}

map.on("load", async () => {{
  const rawPoints = await fetch("raw_points.geojson").then(r => r.json());
  const rawTrack = await fetch("raw_track.geojson").then(r => r.json());

  map.addSource("raw-track", {{
    type: "geojson",
    data: rawTrack
  }});

  map.addLayer({{
    id: "raw-track-line",
    type: "line",
    source: "raw-track",
    layout: {{
      "line-join": "round",
      "line-cap": "round"
    }},
    paint: {{
      "line-color": "#00aaff",
      "line-width": 5,
      "line-opacity": 0.85
    }}
  }});

  map.addSource("raw-points", {{
    type: "geojson",
    data: rawPoints
  }});

  map.addLayer({{
    id: "raw-position-points",
    type: "circle",
    source: "raw-points",
    paint: {{
      "circle-radius": 5,
      "circle-color": "#ff3333",
      "circle-stroke-color": "#ffffff",
      "circle-stroke-width": 1.5,
      "circle-opacity": 0.95
    }}
  }});

  map.on("click", "raw-position-points", (e) => {{
    const f = e.features[0];
    const p = f.properties;
    const c = f.geometry.coordinates;

    new mapboxgl.Popup()
      .setLngLat([c[0], c[1]])
      .setHTML(`
        <b>Raw ADS-B position</b><br/>
        ICAO: ${{p.icao}}<br/>
        row_id: ${{p.row_id ?? "n/a"}}<br/>
        unix_time: ${{p.unix_time}}<br/>
        altitude_ft: ${{p.altitude_ft ?? "n/a"}}<br/>
        altitude_m: ${{Number(p.altitude_m).toFixed(1)}}<br/>
        lat: ${{c[1].toFixed(7)}}<br/>
        lon: ${{c[0].toFixed(7)}}
      `)
      .addTo(map);
  }});

  map.on("mouseenter", "raw-position-points", () => {{
    map.getCanvas().style.cursor = "pointer";
  }});

  map.on("mouseleave", "raw-position-points", () => {{
    map.getCanvas().style.cursor = "";
  }});

  document.getElementById("fit").onclick = () => {{
    const bounds = computeBoundsFromPoints(rawPoints);
    if (!bounds.isEmpty()) {{
      map.fitBounds(bounds, {{
        padding: 90,
        pitch: 60,
        bearing: -25,
        duration: 1200
      }});
    }}
  }};

  document.getElementById("riga").onclick = () => {{
    map.flyTo({{
      center: RIGA_AIRPORT,
      zoom: 13.5,
      pitch: 60,
      bearing: -25,
      duration: 1200
    }});
  }};

  const bounds = computeBoundsFromPoints(rawPoints);
  if (!bounds.isEmpty()) {{
    map.fitBounds(bounds, {{
      padding: 90,
      pitch: 60,
      bearing: -25,
      duration: 1200
    }});
  }}
}});
</script>
</body>
</html>
"""

    (OUTPUT_DIR / "index.html").write_text(html, encoding="utf-8")


# ============================================================
# MAIN — NO ARGS
# ============================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_input_file()
    pos = extract_position_rows(df)

    if len(pos) == 0:
        raise RuntimeError(
            f"No raw position rows found for ICAO={ICAO}. "
            f"Expected df={POSITION_DF}, type_code={POSITION_TYPE_CODE}, valid lat/lon."
        )

    points_geojson = make_points_geojson(pos)
    track_geojson = make_track_geojson(pos)

    raw_points_path = OUTPUT_DIR / "raw_points.geojson"
    raw_track_path = OUTPUT_DIR / "raw_track.geojson"

    raw_points_path.write_text(
        json.dumps(points_geojson, indent=2),
        encoding="utf-8",
    )

    raw_track_path.write_text(
        json.dumps(track_geojson, indent=2),
        encoding="utf-8",
    )

    write_html()

    print()
    print("DONE")
    print("================================")
    print(f"Input file: {INPUT_FILE}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"ICAO: {ICAO}")
    print(f"Raw position rows: {len(pos)}")
    print()
    print("Files written:")
    print(f"  {OUTPUT_DIR / 'raw_points.geojson'}")
    print(f"  {OUTPUT_DIR / 'raw_track.geojson'}")
    print(f"  {OUTPUT_DIR / 'index.html'}")
    print()
    print("Start local web server:")
    print(f"  cd {OUTPUT_DIR}")
    print("  python -m http.server 8000")
    print()
    print("Open in browser:")
    print("  http://localhost:8000")


if __name__ == "__main__":
    main()