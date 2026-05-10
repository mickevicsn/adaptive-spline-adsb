from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import dash
from dash import html
from flask import Response, jsonify, request

from .config import ViewerConfig
from .payload_builder import TrackPayloadBuilder


PACKAGE_DIR = Path(__file__).parent
TEMPLATE_DIR = PACKAGE_DIR / "templates"
ASSETS_DIR = PACKAGE_DIR / "assets"

PREFERRED_JSON_NAMES = (
    "input.json",
    "track.json",
    "debug_track.json",
)

METHOD_PLACEHOLDERS: tuple[dict[str, str], ...] = (
    {
        "methodId": "raw_adsb",
        "label": "Raw ADS-B",
        "description": "Reported ADS-B positions and GS+track-derived vectors.",
    },
    {
        "methodId": "kalman_rts",
        "label": "Kalman RTS",
        "description": "RTS-smoothed reconstruction. Raw observations are shown as disconnected points; reconstructed trajectory is shown as a line.",
    },
    {
        "methodId": "v_spline",
        "label": "V-Spline",
        "description": "Velocity-aware spline reconstruction. Raw observations are shown as disconnected points; V-Spline trajectory is shown as a line.",
    },
    {
        "methodId": "b_spline",
        "label": "B-spline",
        "description": "B-spline reconstruction. Raw observations are shown as disconnected points; B-spline trajectory is shown as a line.",
    },
    {
        "methodId": "hermite_spline",
        "label": "Hermite spline",
        "description": "Hermite-spline reconstruction. Raw observations are shown as disconnected points; Hermite spline trajectory is shown as a line.",
    },
)


class MethodNotAvailableError(RuntimeError):
    pass


def _safe_read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _strip_debug_suffix(name: str) -> str:
    for suffix in ("_debug_track", "-debug-track", "_track", "-track"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def _flight_sort_key(record: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("startTimeUtc") or ""),
        str(record.get("icao") or ""),
        str(record.get("flightId") or record.get("id") or ""),
    )


def _method_placeholder(method_id: str) -> dict[str, str] | None:
    for method in METHOD_PLACEHOLDERS:
        if method["methodId"] == method_id:
            return dict(method)
    return None


def _normalise_method(
    *,
    root: Path,
    flight_dir: Path | None,
    method: dict[str, Any],
) -> dict[str, Any]:
    method_id = str(method.get("methodId") or method.get("id") or "").strip()
    if not method_id:
        raise ValueError("Method entry is missing methodId")

    placeholder = _method_placeholder(method_id) or {}
    label = str(method.get("label") or method.get("name") or placeholder.get("label") or method_id)
    description = str(method.get("description") or placeholder.get("description") or "")

    file_value = method.get("file") or method.get("jsonFile") or method.get("path")
    file_label = str(file_value) if file_value else f"methods/{method_id}.json"
    detailed_file_value = method.get("detailedFile") or method.get("detailFile") or method.get("debugFile")
    detailed_file_label = str(detailed_file_value) if detailed_file_value else ""

    candidates: list[Path] = []
    if file_value:
        file_path = Path(str(file_value))
        if file_path.is_absolute():
            candidates.append(file_path)
        else:
            if flight_dir is not None:
                candidates.append(flight_dir / file_path)
            candidates.append(root / file_path)
    elif flight_dir is not None:
        candidates.append(flight_dir / "methods" / f"{method_id}.json")

    resolved_path = next((candidate for candidate in candidates if candidate.is_file()), None)

    status = str(method.get("status") or method.get("methodStatus") or "").strip()
    if not status:
        status = "available" if resolved_path is not None else "coming_soon"

    return {
        "methodId": method_id,
        "id": method_id,
        "label": label,
        "description": description,
        "file": file_label,
        "detailedFile": detailed_file_label,
        "available": resolved_path is not None,
        "placeholder": resolved_path is None,
        "status": status,
    }


def _add_method_if_missing(methods: list[dict[str, Any]], method: dict[str, Any]) -> None:
    method_id = method.get("methodId") or method.get("id")
    if not method_id:
        return
    if any(existing.get("methodId") == method_id for existing in methods):
        return
    methods.append(method)


def _normalise_methods(
    *,
    root: Path,
    flight_dir: Path | None,
    declared_methods: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []

    for method in declared_methods:
        try:
            _add_method_if_missing(
                methods,
                _normalise_method(root=root, flight_dir=flight_dir, method=method),
            )
        except Exception:
            continue

    # Auto-discover one-JSON-per-method files, so adding a future method JSON is
    # enough to make it selectable even before flights.json is hand-edited.
    if flight_dir is not None:
        methods_dir = flight_dir / "methods"
        for method_path in sorted(methods_dir.glob("*.json")) if methods_dir.is_dir() else []:
            method_id = method_path.stem
            _add_method_if_missing(
                methods,
                _normalise_method(
                    root=root,
                    flight_dir=flight_dir,
                    method={
                        "methodId": method_id,
                        "file": f"methods/{method_path.name}",
                    },
                ),
            )

    # Keep supported reconstruction methods visible in the chooser, even when
    # a method JSON has not been generated for the selected flight yet.
    for placeholder in METHOD_PLACEHOLDERS:
        _add_method_if_missing(
            methods,
            _normalise_method(root=root, flight_dir=flight_dir, method=placeholder),
        )

    if not methods:
        methods.append(
            _normalise_method(
                root=root,
                flight_dir=flight_dir,
                method={"methodId": "raw_adsb", "file": "methods/raw_adsb.json"},
            )
        )

    return methods


def _flight_dir_from_index_entry(root: Path, entry: dict[str, Any]) -> Path | None:
    flight_id = str(entry.get("flightId") or entry.get("id") or "").strip()
    metadata_file = entry.get("flightMetadataFile") or entry.get("metadataFile")

    if metadata_file:
        metadata_path = Path(str(metadata_file))
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
        return metadata_path.parent

    if flight_id:
        candidate = root / "flights" / flight_id
        if candidate.exists():
            return candidate

    return None


def _load_flight_metadata(flight_dir: Path | None) -> dict[str, Any]:
    if flight_dir is None:
        return {}

    metadata_path = flight_dir / "flight.json"
    if not metadata_path.is_file():
        return {}

    try:
        return _safe_read_json(metadata_path)
    except Exception:
        return {}


def _normalise_flight_record(
    *,
    root: Path,
    index: int,
    entry: dict[str, Any],
) -> dict[str, Any]:
    flight_id = str(entry.get("flightId") or entry.get("id") or f"flight_{index}").strip()
    flight_dir = _flight_dir_from_index_entry(root, entry)
    metadata = _load_flight_metadata(flight_dir)

    declared_methods = entry.get("methods") or []
    if not isinstance(declared_methods, list):
        declared_methods = []

    methods = _normalise_methods(root=root, flight_dir=flight_dir, declared_methods=declared_methods)

    requested_default = str(entry.get("defaultMethod") or metadata.get("defaultMethod") or "raw_adsb")
    if not any(method["methodId"] == requested_default and method["available"] for method in methods):
        first_available = next((method["methodId"] for method in methods if method["available"]), None)
        default_method = first_available or requested_default
    else:
        default_method = requested_default

    start_time = str(entry.get("startTimeUtc") or metadata.get("startTimeUtc") or "")
    end_time = str(entry.get("endTimeUtc") or metadata.get("endTimeUtc") or "")
    icao = str(entry.get("icao") or metadata.get("icao") or flight_id)
    callsign = str(entry.get("callsign") or metadata.get("callsign") or "")
    origin = str(entry.get("origin") or metadata.get("origin") or "")
    destination = str(entry.get("destination") or metadata.get("destination") or "")
    label = str(entry.get("label") or metadata.get("label") or _make_flight_label(icao, start_time, origin, destination))

    return {
        "id": flight_id,
        "flightId": flight_id,
        "icao": icao,
        "callsign": callsign,
        "label": label,
        "startTimeUtc": start_time,
        "endTimeUtc": end_time,
        "origin": origin,
        "destination": destination,
        "flightMetadataFile": entry.get("flightMetadataFile") or (
            _relative_to_root(root, flight_dir / "flight.json") if flight_dir else ""
        ),
        "defaultMethod": default_method,
        "methods": methods,
        "sourceKind": "indexed",
        "index": index,
    }


def _relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _make_flight_label(icao: str, start_time_utc: str, origin: str, destination: str) -> str:
    date_part = start_time_utc[:10] if start_time_utc else "unknown-date"
    route = f" · {origin} → {destination}" if origin or destination else ""
    return f"{icao} · {date_part}{route}"


def _discover_indexed_flights(root: Path) -> list[dict[str, Any]]:
    index_path = root / "flights.json"
    if not index_path.is_file():
        return []

    index_payload = _safe_read_json(index_path)
    entries = index_payload.get("flights") or []
    if not isinstance(entries, list):
        raise ValueError("track_output/flights.json must contain a flights array")

    records = [
        _normalise_flight_record(root=root, index=index, entry=entry)
        for index, entry in enumerate(entries)
        if isinstance(entry, dict)
    ]
    return sorted(records, key=_flight_sort_key)


def _legacy_json_for_folder(folder: Path) -> Path | None:
    for preferred_name in PREFERRED_JSON_NAMES:
        candidate = folder / preferred_name
        if candidate.is_file():
            return candidate

    exact_name = folder / f"{folder.name}.json"
    if exact_name.is_file():
        return exact_name

    json_files = sorted(p for p in folder.glob("*.json") if p.is_file())
    if not json_files:
        return None

    for candidate in json_files:
        if "input" in candidate.stem.lower():
            return candidate

    return json_files[0]


def _legacy_record(index: int, flight_id: str, icao: str, json_path: Path, root: Path) -> dict[str, Any]:
    method = _normalise_method(
        root=root,
        flight_dir=None,
        method={
            "methodId": "raw_adsb",
            "label": "Raw ADS-B",
            "file": _relative_to_root(root, json_path),
        },
    )
    methods = [method]
    for placeholder in METHOD_PLACEHOLDERS:
        if placeholder["methodId"] == "raw_adsb":
            continue
        methods.append(_normalise_method(root=root, flight_dir=None, method=placeholder))

    return {
        "id": flight_id,
        "flightId": flight_id,
        "icao": icao,
        "callsign": "",
        "label": icao,
        "startTimeUtc": "",
        "endTimeUtc": "",
        "origin": "",
        "destination": "",
        "flightMetadataFile": "",
        "defaultMethod": "raw_adsb",
        "methods": methods,
        "sourceKind": "legacy",
        "index": index,
    }


def _discover_legacy_flights(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "flights"):
        json_path = _legacy_json_for_folder(folder)
        if json_path is None:
            continue
        flight_id = f"legacy-folder:{folder.name}"
        records.append(_legacy_record(len(records), flight_id, folder.name, json_path, root))

    for json_path in sorted(p for p in root.glob("*.json") if p.is_file() and p.name != "flights.json"):
        icao = _strip_debug_suffix(json_path.stem)
        flight_id = f"legacy-file:{json_path.stem}"
        records.append(_legacy_record(len(records), flight_id, icao, json_path, root))

    return records


def discover_flights(track_output_dir: str | Path) -> list[dict[str, Any]]:
    """Discover selectable flight/method combinations.

    Preferred layout:
        track_output/flights.json
        track_output/flights/<flight_id>/flight.json
        track_output/flights/<flight_id>/methods/<method_id>.json

    Legacy layouts are still accepted so older packages keep running.
    """
    root = Path(track_output_dir)
    if not root.exists():
        return []

    indexed = _discover_indexed_flights(root)
    if indexed:
        return indexed

    return _discover_legacy_flights(root)


def _find_flight(flights: list[dict[str, Any]], flight_id: str | None) -> dict[str, Any]:
    if not flights:
        raise FileNotFoundError("No flights were found in track_output.")

    if flight_id:
        selected = next((flight for flight in flights if flight["id"] == flight_id or flight["flightId"] == flight_id), None)
        if selected is not None:
            return selected

    return flights[0]


def _find_method(flight: dict[str, Any], method_id: str | None) -> dict[str, Any]:
    methods = flight.get("methods") or []
    if not methods:
        raise FileNotFoundError(f"Flight {flight['flightId']} has no methods.")

    requested = method_id or flight.get("defaultMethod") or "raw_adsb"
    selected = next((method for method in methods if method["methodId"] == requested), None)

    if selected is None and method_id is None:
        selected = next((method for method in methods if method.get("available")), methods[0])

    if selected is None:
        placeholder = _method_placeholder(requested)
        if placeholder is not None:
            raise MethodNotAvailableError(
                f"Method '{requested}' is not available yet. Add methods/{requested}.json and list it in flights.json to enable it."
            )
        raise FileNotFoundError(f"Method '{requested}' is not listed for flight {flight['flightId']}.")

    if not selected.get("available"):
        raise MethodNotAvailableError(
            f"Method '{selected['methodId']}' is not available yet for flight {flight['flightId']}. "
            f"Expected JSON: {selected.get('file') or 'methods/<method>.json'}"
        )

    return selected


def _resolve_method_path(root: Path, flight: dict[str, Any], method: dict[str, Any]) -> Path:
    file_value = method.get("file")
    if not file_value:
        raise FileNotFoundError(f"Method {method['methodId']} has no JSON file configured.")

    file_path = Path(str(file_value))
    candidates: list[Path] = []

    if file_path.is_absolute():
        candidates.append(file_path)
    else:
        candidate_flight_dir = root / "flights" / str(flight["flightId"])
        candidates.append(candidate_flight_dir / file_path)
        candidates.append(root / file_path)

    json_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if json_path is None:
        raise FileNotFoundError(
            f"JSON file for {flight['flightId']} / {method['methodId']} was not found: {file_value}"
        )

    return json_path


def _resolve_selection(
    track_output_dir: str | Path,
    flight_id: str | None,
    method_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any], Path, list[dict[str, Any]]]:
    root = Path(track_output_dir)
    flights = discover_flights(root)
    if not flights:
        raise FileNotFoundError(
            "No flight data found. Expected track_output/flights.json with "
            "flights/<flight_id>/methods/raw_adsb.json."
        )

    selected_flight = _find_flight(flights, flight_id)
    selected_method = _find_method(selected_flight, method_id)
    json_path = _resolve_method_path(root, selected_flight, selected_method)

    return selected_flight, selected_method, json_path, flights


def _embedded_flights(payload: dict[str, Any]) -> list[dict[str, Any]]:
    track_id = payload.get("track_id") or "track"
    return [
        {
            "id": "embedded",
            "flightId": "embedded",
            "icao": str(track_id),
            "callsign": "",
            "label": str(track_id),
            "startTimeUtc": "",
            "endTimeUtc": "",
            "origin": "",
            "destination": "",
            "flightMetadataFile": "",
            "defaultMethod": "raw_adsb",
            "methods": [
                {
                    "methodId": "raw_adsb",
                    "id": "raw_adsb",
                    "label": "Raw ADS-B",
                    "description": "Embedded payload.",
                    "file": "",
                    "available": True,
                    "placeholder": False,
                    "status": "available",
                }
            ],
            "sourceKind": "embedded",
            "index": 0,
        }
    ]


def create_dash_app(
    payload: dict[str, Any] | None = None,
    *,
    config: ViewerConfig | None = None,
    track_output_dir: str | Path = "track_output",
) -> dash.Dash:
    """Create the Dash app and expose the viewer plus flight/method APIs."""

    config = config or ViewerConfig()
    app_title = str((payload or {}).get("title") or config.title or "ADS-B 3D Flight Viewer")

    app = dash.Dash(
        __name__,
        assets_folder=str(ASSETS_DIR),
        title=app_title,
    )
    server = app.server

    @server.route("/viewer")
    def viewer_page() -> Response:
        template = (TEMPLATE_DIR / "viewer.html").read_text(encoding="utf-8")
        template = template.replace("__TITLE__", app_title)
        return Response(template, mimetype="text/html")

    @server.route("/api/flights")
    def flights_api():
        if payload is not None:
            return jsonify({"flights": _embedded_flights(payload)})

        return jsonify({"flights": discover_flights(track_output_dir)})

    @server.route("/api/payload")
    def payload_api():
        if payload is not None:
            flights = _embedded_flights(payload)
            enriched = dict(payload)
            enriched.setdefault("selectedFlightId", "embedded")
            enriched.setdefault("selectedMethodId", "raw_adsb")
            enriched.setdefault("selectedFlight", flights[0])
            enriched.setdefault("selectedMethod", flights[0]["methods"][0])
            enriched.setdefault("availableFlights", flights)
            enriched.setdefault("availableMethods", flights[0]["methods"])
            return jsonify(enriched)

        flight_id = request.args.get("flight")
        method_id = request.args.get("method")

        try:
            selected_flight, selected_method, json_path, flights = _resolve_selection(
                track_output_dir,
                flight_id,
                method_id,
            )
            flight_config = replace(
                config,
                title=f"{config.title} — {selected_flight['label']} — {selected_method['label']}",
            )
            built_payload = TrackPayloadBuilder(flight_config).build_payload(
                json_path,
                method_id=selected_method["methodId"],
                method_label=selected_method.get("label") or selected_method["methodId"],
            )
            built_payload["selectedFlightId"] = selected_flight["flightId"]
            built_payload["selectedMethodId"] = selected_method["methodId"]
            built_payload["selectedFlight"] = selected_flight
            built_payload["selectedMethod"] = selected_method
            built_payload["availableFlights"] = flights
            built_payload["availableMethods"] = selected_flight.get("methods") or []
            return jsonify(built_payload)
        except MethodNotAvailableError as exc:
            response = jsonify({"error": str(exc), "kind": "method_not_available"})
            response.status_code = 501
            return response
        except Exception as exc:
            response = jsonify({"error": str(exc)})
            response.status_code = 404
            return response

    app.layout = html.Div(
        className="viewer-frame-shell",
        children=html.Iframe(
            src="/viewer",
            className="viewer-frame",
            style={
                "border": "0",
                "display": "block",
            },
        ),
    )

    return app
