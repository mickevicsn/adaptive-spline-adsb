import { AnimationController } from "./animation_controller.js";
import { CameraController } from "./camera_controller.js";
import { CesiumFlightScene } from "./cesium_scene.js";
import { ControlPanel } from "./controls.js";
import { DashboardUI } from "./dashboard_ui.js";
import { ViewerState } from "./state.js";

async function loadFlights() {
  const response = await fetch("/api/flights");

  if (!response.ok) {
    throw new Error(`Failed to load flight list: ${response.status} ${response.statusText}`);
  }

  const data = await response.json();
  return data.flights || [];
}

async function loadPayload(flightId, methodId) {
  const url = new URL("/api/payload", window.location.origin);
  if (flightId) {
    url.searchParams.set("flight", flightId);
  }
  if (methodId) {
    url.searchParams.set("method", methodId);
  }

  const response = await fetch(url);

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const errorPayload = await response.json();
      if (errorPayload?.error) {
        detail = errorPayload.error;
      }
    } catch (_) {
      // Keep the HTTP status if the server did not send JSON.
    }
    throw new Error(`Failed to load payload: ${detail}`);
  }

  return response.json();
}

function installResponsiveOverlayLayout() {
  const root = document.getElementById("viewerRoot");
  const topbar = document.querySelector(".topbar");
  const titleCard = document.querySelector(".title-card");

  if (!root || !topbar || !titleCard) {
    return;
  }

  const updateLayoutVars = () => {
    const rootRect = root.getBoundingClientRect();
    const topbarRect = topbar.getBoundingClientRect();
    const titleRect = titleCard.getBoundingClientRect();

    root.style.setProperty("--topbar-bottom", `${Math.round(topbarRect.bottom - rootRect.top)}px`);
    root.style.setProperty("--titlebar-bottom", `${Math.round(titleRect.bottom - rootRect.top)}px`);
  };

  const ro = new ResizeObserver(updateLayoutVars);
  ro.observe(topbar);
  ro.observe(titleCard);
  window.addEventListener("resize", updateLayoutVars);
  requestAnimationFrame(updateLayoutVars);
}

function getFlightId(flight) {
  return flight?.id || flight?.flightId || "";
}

function getMethodId(method) {
  return method?.methodId || method?.id || "";
}

function getMethods(flight) {
  return Array.isArray(flight?.methods) ? flight.methods : [];
}

function flightDisplayName(flight) {
  return flight?.label || flight?.icao || flight?.name || getFlightId(flight) || "Flight";
}

function methodDisplayName(method) {
  return method?.label || method?.name || getMethodId(method) || "Method";
}

function availableMethods(flight) {
  return getMethods(flight).filter(method => method.available !== false && !method.placeholder);
}

function defaultMethodForFlight(flight) {
  const methods = getMethods(flight);
  const defaultMethodId = flight?.defaultMethod || "raw_adsb";
  return (
    methods.find(method => getMethodId(method) === defaultMethodId && method.available !== false && !method.placeholder) ||
    availableMethods(flight)[0] ||
    methods[0] ||
    null
  );
}

function selectionFromUrl(flights) {
  const params = new URLSearchParams(window.location.search);
  const requestedFlightId = params.get("flight");
  const requestedMethodId = params.get("method");

  if (!requestedFlightId) {
    return null;
  }

  const flight = flights.find(candidate => getFlightId(candidate) === requestedFlightId || candidate.flightId === requestedFlightId);
  if (!flight) {
    return null;
  }

  const methods = getMethods(flight);
  let method = null;

  if (requestedMethodId) {
    method = methods.find(candidate => getMethodId(candidate) === requestedMethodId) || null;
    if (method && (method.available === false || method.placeholder)) {
      return null;
    }
  }

  if (!method) {
    method = defaultMethodForFlight(flight);
  }

  if (!method || method.available === false || method.placeholder) {
    return null;
  }

  return {
    flightId: getFlightId(flight),
    methodId: getMethodId(method),
    flight,
    method,
  };
}

function formatFlightMeta(flight) {
  const pieces = [];
  if (flight?.icao) pieces.push(`ICAO ${flight.icao}`);
  if (flight?.callsign) pieces.push(`Callsign ${flight.callsign}`);
  if (flight?.startTimeUtc) pieces.push(flight.startTimeUtc);
  if (flight?.origin || flight?.destination) pieces.push(`${flight.origin || "?"} → ${flight.destination || "?"}`);
  return pieces.join(" · ") || "Metadata may be completed in flight.json";
}

function formatMethodMeta(method) {
  if (method.available === false || method.placeholder) {
    return method.description || `Add ${method.file || "methods/<method>.json"} to enable this method.`;
  }
  return method.description || method.file || "Available method JSON";
}

function showFlightChooser(flights, options = {}) {
  const overlay = document.getElementById("flightOverlay");
  const title = document.getElementById("flightOverlayTitle");
  const message = document.getElementById("flightOverlayMessage");
  const list = document.getElementById("flightList");

  if (!overlay || !title || !message || !list) {
    return Promise.reject(new Error("Flight chooser overlay was not found in viewer.html."));
  }

  title.textContent = options.title || "Select flight dataset and reconstruction method";
  message.textContent = options.message || "Choose a flight dataset and then select one reconstruction method.";
  list.innerHTML = "";

  return new Promise(resolve => {
    if (!flights.length) {
      const empty = document.createElement("div");
      empty.className = "flight-empty";
      empty.textContent = "No flights were found. Expected track_output/flights.json and at least one methods/raw_adsb.json file inside a flight folder.";
      list.appendChild(empty);
      overlay.classList.remove("hidden");
      return;
    }

    for (const flight of flights) {
      const card = document.createElement("section");
      card.className = "flight-card";
      if (getFlightId(flight) === options.currentFlightId) {
        card.classList.add("flight-card-active");
      }

      const header = document.createElement("div");
      header.className = "flight-card-header";

      const name = document.createElement("span");
      name.className = "flight-card-name";
      name.textContent = flightDisplayName(flight);

      header.appendChild(name);

      const meta = document.createElement("span");
      meta.className = "flight-card-meta";
      meta.textContent = formatFlightMeta(flight);

      const methodsWrap = document.createElement("div");
      methodsWrap.className = "method-list";

      const methods = getMethods(flight);
      if (!methods.length) {
        const emptyMethod = document.createElement("div");
        emptyMethod.className = "method-empty";
        emptyMethod.textContent = "No methods are listed for this flight.";
        methodsWrap.appendChild(emptyMethod);
      }

      for (const method of methods) {
        const methodButton = document.createElement("button");
        methodButton.type = "button";
        methodButton.className = "method-card";
        const methodId = getMethodId(method);
        const methodAvailable = method.available !== false && !method.placeholder;
        methodButton.disabled = !methodAvailable;

        if (getFlightId(flight) === options.currentFlightId && methodId === options.currentMethodId) {
          methodButton.classList.add("method-card-active");
        }

        const methodName = document.createElement("span");
        methodName.className = "method-card-name";
        methodName.textContent = methodDisplayName(method);

        const methodMeta = document.createElement("span");
        methodMeta.className = "method-card-meta";
        methodMeta.textContent = formatMethodMeta(method);

        methodButton.appendChild(methodName);
        methodButton.appendChild(methodMeta);

        if (methodAvailable) {
          methodButton.addEventListener("click", () => {
            overlay.classList.add("hidden");
            resolve({
              flightId: getFlightId(flight),
              methodId,
            });
          });
        }

        methodsWrap.appendChild(methodButton);
      }

      card.appendChild(header);
      card.appendChild(meta);
      card.appendChild(methodsWrap);
      list.appendChild(card);
    }

    overlay.classList.remove("hidden");
  });
}

function openFlightInViewer(flightId, methodId) {
  const url = new URL("/viewer", window.location.origin);
  url.searchParams.set("flight", flightId);
  if (methodId) {
    url.searchParams.set("method", methodId);
  }
  window.location.replace(url.toString());
}

function showError(error) {
  console.error(error);

  const box = document.createElement("div");
  box.className = "error-box";
  box.textContent = error?.stack || error?.message || String(error);
  document.body.appendChild(box);
}

async function main() {
  installResponsiveOverlayLayout();

  const flights = await loadFlights();
  if (!flights.length) {
    await showFlightChooser(flights, {
      title: "No flights found",
      message: "Create track_output/flights.json and at least one methods/raw_adsb.json file, then reload the viewer.",
    });
    return;
  }

  let selection = selectionFromUrl(flights);
  if (!selection) {
    selection = await showFlightChooser(flights, {
      title: "Select flight dataset and reconstruction method",
      message: "Choose the flight dataset and the reconstruction method to view.",
    });
  }

  const payload = await loadPayload(selection.flightId, selection.methodId);

  document.title = payload.title || "ADS-B 3D Flight Viewer";
  document.getElementById("title").textContent = payload.title || "ADS-B 3D Flight Viewer";

  const state = new ViewerState(payload);

  const scene = new CesiumFlightScene("cesiumContainer", state);
  scene.initialize();

  const camera = new CameraController(scene.viewer, state);
  camera.installKeyboardHandlers();
  camera.installWheelHandler(document.getElementById("cesiumContainer"));

  const controls = new ControlPanel(state, scene);
  controls.mount();

  const hud = new DashboardUI(state, {
    onReset: async () => {
      state.playing = false;
      const selected = await showFlightChooser(state.payload.availableFlights || flights, {
        title: "Select flight dataset and reconstruction method",
        message: "Choose a flight dataset and method to reset and reload the viewer.",
        currentFlightId: state.payload.selectedFlightId,
        currentMethodId: state.payload.selectedMethodId,
      });
      openFlightInViewer(selected.flightId, selected.methodId);
    },
  });
  hud.mount();

  const animation = new AnimationController(state, scene, camera, hud);
  animation.start();

  setInterval(() => {
    hud.syncButtons();
  }, 250);
}

main().catch(showError);
