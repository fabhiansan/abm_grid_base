// --- Projections ---
proj4.defs(
  "EPSG:32749",
  "+proj=utm +zone=49 +south +datum=WGS84 +units=m +no_defs"
);
proj4.defs(
  "EPSG:32750",
  "+proj=utm +zone=50 +south +datum=WGS84 +units=m +no_defs"
);

// --- Helper Functions ---
function convertToLatLng(x, y, sourceCRS) {
  /* ... (keep existing implementation) ... */
  if (!sourceCRS || !proj4.defs[sourceCRS]) {
    console.warn(`Source CRS '${sourceCRS}' not defined...`);
    if (Math.abs(x) <= 180 && Math.abs(y) <= 90) {
      return [y, x];
    }
    console.error(`Cannot project...`);
    return null;
  }
  try {
    if (!isFinite(x) || !isFinite(y)) return null;
    const [lng, lat] = proj4(sourceCRS, "EPSG:4326", [
      Number(x),
      Number(y),
    ]);
    if (!isFinite(lat) || !isFinite(lng)) return null;
    return [lat, lng];
  } catch (error) {
    console.error(`Error converting...`, error);
    return null;
  }
}
function getAgentColor(type) {
  /* ... (keep existing implementation) ... */
  switch (String(type).toLowerCase()) {
    case "adult":
      return "#ff0000";
    case "child":
      return "#00ff00";
    case "elder":
      return "#0000ff";
    case "teen":
      return "#ffA500";
    default:
      return "#999999";
  }
}
function updateInfoDisplay() {
  let simStatusText = "Idle";
  if (isSimulationInitialized) simStatusText = "Initialized";
  if (isSimulationRunning) simStatusText = "Running";
  if (isSimulationCompleted) simStatusText = "Completed";

  let sirenText = "";
  if (sirenConfig) {
      sirenText = ` | Siren @ Step: ${sirenConfig.activation_step} (Pos: ${sirenConfig.x},${sirenConfig.y}, Radius: ${sirenConfig.radius_cells})`;
      // Add visual cue if siren is active
      if (isSimulationRunning && currentStep >= sirenConfig.activation_step) {
          sirenText += " (ACTIVE)";
          infoDisplay.style.color = "#dc3545"; // Example: Make text red when siren active
      } else {
           infoDisplay.style.color = ""; // Reset color if not active
      }
  } else {
       sirenText = " | Siren: Disabled";
       infoDisplay.style.color = ""; // Reset color if no siren
  }

  infoDisplay.textContent = `Step: ${currentStep} | Agents: ${totalAgents} | Status: ${simStatusText}${sirenText}`;
}
function setStatusMessage(message, isError = false) {
  /* ... (keep existing implementation) ... */
  status.textContent = message;
  if (isError) {
    status.classList.add("error-message");
  } else {
    status.classList.remove("error-message");
  }
}
function updateButtonStates() {
  /* ... (keep existing implementation) ... */
  initBtn.disabled = isSimulationInitialized;
  resetBtn.disabled = !isSimulationInitialized;
  playPauseBtn.disabled =
    !isSimulationInitialized || isSimulationCompleted;
  runStepBtn.disabled =
    !isSimulationInitialized || isSimulationCompleted || isPolling;
  runStepsBtn.disabled =
    !isSimulationInitialized || isSimulationCompleted || isPolling;
  updateConfigBtn.disabled = isSimulationInitialized;
  exportBtn.disabled = !isSimulationInitialized;
}
async function apiFetch(endpoint, options = {}) {
  const url = endpoint; // Use the endpoint directly without adding /api
  setStatusMessage(`Sending request to ${url}...`);
  try {
    const response = await fetch(url, options); // Fetch the correct URL
    // Check if response is ok AND content type is JSON before parsing
    const contentType = response.headers.get("content-type");
    if (!response.ok) {
      const errorText = await response.text();
      console.error(
        `API Error: Status ${response.status}, URL: ${url}, Response:`,
        errorText
      );
      throw new Error(`HTTP error ${response.status}`);
    }
    // Only parse if content type indicates JSON to avoid "Unexpected token '<'"
    if (contentType && contentType.includes("application/json")) {
      const result = await response.json();
      setStatusMessage(`${options.method || "GET"} ${url} successful.`);
      return result;
    } else {
      // Handle cases where response is OK but not JSON (if applicable)
      console.warn(
        `Received non-JSON response for ${url}, Content-Type: ${contentType}`
      );
      // Return something sensible or throw an error depending on expected behavior
      return {
        status: "ok",
        message: "Received non-JSON response",
        contentType: contentType,
      };
    }
  } catch (error) {
    console.error(`Error fetching ${url}:`, error);
    setStatusMessage(`Error: ${error.message}`, true);
    throw error; // Re-throw the error to be caught by the caller
  }
}