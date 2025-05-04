// --- Data Fetching & Visualization ---
async function fetchMetadata() {
  /* ... (keep existing implementation) ... */
  try {
    const response = await fetch("/data/metadata");
    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ error: `HTTP error ${response.status}` }));
      throw new Error(errorData.error || `HTTP error ${response.status}`);
    }
    const statusData = await response.json();
    const simState = statusData.simulation_state || {};
    currentStep = simState.current_step || 0;
    isSimulationRunning = simState.is_running || false;
    isSimulationCompleted = simState.is_completed || false;
    isSimulationInitialized = simState.is_running || simState.is_completed;
    sirenConfig = simState.siren_config || null; // Store siren config object from state
    totalAgents = statusData.agents_count || 0;
    updateInfoDisplay();
    updateButtonStates();
    if (!status.classList.contains("error-message")) {
      setStatusMessage(
        isSimulationRunning
          ? "Simulation running..."
          : isSimulationCompleted
          ? "Simulation completed."
          : isSimulationInitialized
          ? "Simulation Initialized."
          : "Simulation Idle."
      );
    }
    return statusData;
  } catch (error) {
    console.error("Error loading metadata:", error);
    setStatusMessage(`Error loading metadata: ${error.message}`, true);
    playPauseBtn.disabled = true;
    updateButtonStates();
    return null;
  }
}
async function fetchCurrentGeojsonData() {
  /* ... (keep existing implementation) ... */
  try {
    const response = await fetch("/data/current_geojson");
    const rawResponseText = await response.text(); // Read the response as text
    console.log("Raw response text from /data/current_geojson:", rawResponseText); // Log raw text
    if (!response.ok) {
      let errorData = { error: `HTTP error ${response.status}` };
      try {
        errorData = JSON.parse(rawResponseText); // Attempt to parse if it's JSON
      } catch (e) {
        console.warn("Failed to parse error response as JSON:", e);
      }
      throw new Error(errorData.error || `HTTP error ${response.status}`);
    }
    const responseData = JSON.parse(rawResponseText); // Parse the text as JSON
    console.log("Parsed response data:", JSON.stringify(responseData, null, 2)); // Log the full response data

    const geojsonData = responseData.geojson; // Extract the GeoJSON object

    console.log("Extracted GeoJSON data:", JSON.stringify(geojsonData, null, 2)); // Log the extracted GeoJSON

    if (!geojsonData || geojsonData.type !== "FeatureCollection") {
      throw new Error("Invalid GeoJSON data received (not a FeatureCollection)");
    }
    currentCRS = geojsonData.crs?.properties?.name;
    if (currentCRS && !proj4.defs[currentCRS]) {
      console.warn(`CRS ${currentCRS} defined...`);
    }
    return geojsonData; // Return the extracted GeoJSON data
  } catch (error) {
    console.error("Error loading GeoJSON:", error);
    setStatusMessage(`Error loading agent data: ${error.message}`, true);
    if (isPolling) {
      togglePolling();
    }
    return null;
  }
}
async function fetchGridCosts() {
  /* ... (keep existing implementation) ... */
  // REMOVED: if (gridCostData) return gridCostData; // Force fetch every time
  try {
    setStatusMessage("Fetching grid cost data...");
    gridCostData = await apiFetch("/api/grid/costs");
    // --- ADD LOGGING HERE ---
    console.log("Received gridCostData:", JSON.stringify(gridCostData)); // Log the full structure
    if (gridCostData && gridCostData.distance_to_road) {
      console.log(
        "Sample distance_to_road data (first 5 rows):",
        gridCostData.distance_to_road.slice(0, 5)
      );
    }
    // --- END LOGGING ---
    if (
      !gridCostData ||
      !gridCostData.distance_to_road ||
      !gridCostData.distance_to_shelter
    ) {
      throw new Error("Incomplete cost data received");
    }
    setStatusMessage("Grid cost data loaded.");
    return gridCostData;
  } catch (error) {
    setStatusMessage(`Error fetching grid costs: ${error.message}`, true);
    gridCostData = null; // Clear cache on error
    return null;
  }
}

// --- Rendering Functions ---
async function renderGridLayer() {
  /* ... (keep existing implementation) ... */
  if (gridLayer) map.removeLayer(gridLayer);
  try {
    setStatusMessage("Fetching grid data...");
    const gridGeoJson = await apiFetch("/api/grid/geojson");
    if (!gridGeoJson || !gridGeoJson.features) {
      throw new Error("Invalid grid GeoJSON received");
    }
    const gridCRS = gridGeoJson.crs?.properties?.name || currentCRS;
    gridLayer = L.featureGroup([], { pane: "gridPane" });
    L.geoJSON(gridGeoJson, {
      filter: function (feature) {
        return feature.properties && feature.properties.type === "road";
      },
      pointToLayer: function (feature, latlng) {
        return L.circleMarker(latlng, {
          radius: 3,
          weight: 1,
          opacity: 0.8,
          fillOpacity: 0.6,
          fillColor: "#888888",
          color: "#555555",
        });
      },
      coordsToLatLng: function (coords) {
        const latLng = convertToLatLng(coords[0], coords[1], gridCRS);
        return latLng ? L.latLng(latLng[0], latLng[1]) : null;
      },
    }).addTo(gridLayer);
    L.geoJSON(gridGeoJson, {
      filter: function (feature) {
        return (
          feature.properties && feature.properties.type === "shelter"
        );
      },
      pointToLayer: function (feature, latlng) {
        return L.circleMarker(latlng, {
          pane: "shelterPane",
          radius: 7,
          fillColor: "#006400",
          color: "#004d00",
          weight: 1,
          opacity: 0.9,
          fillOpacity: 0.7,
        }).bindPopup(`Shelter ID: ${feature.properties.id}`);
      },
      coordsToLatLng: function (coords) {
        const latLng = convertToLatLng(coords[0], coords[1], gridCRS);
        return latLng ? L.latLng(latLng[0], latLng[1]) : null;
      },
    }).addTo(map);
    gridLayer.addTo(map);
    setStatusMessage("Grid data rendered.");
  } catch (error) {
    setStatusMessage(`Error rendering grid: ${error.message}`, true);
  }
}
function renderTsunamiLayer(tsunamiGeoJson) {
  /* ... (keep existing implementation) ... */
  if (!map) return;
  if (tsunamiLayer) map.removeLayer(tsunamiLayer);
  if (
    !tsunamiGeoJson ||
    !tsunamiGeoJson.features ||
    tsunamiGeoJson.features.length === 0
  ) {
    console.log("No active tsunami data...");
    tsunamiLayer = null;
    return;
  }
  const tsunamiCRS = tsunamiGeoJson.crs?.properties?.name || currentCRS;
  tsunamiLayer = L.geoJSON(tsunamiGeoJson, {
    pane: "tsunamiPane",
    pointToLayer: function (feature, latlng) {
      const height = feature.properties.height || 1;
      return L.rectangle(latlng.toBounds(5), {
        fillColor: "blue",
        color: "#0000AA",
        weight: 0,
        fillOpacity: 0.4 + height / 20,
      });
    },
    coordsToLatLng: function (coords) {
      const latLng = convertToLatLng(coords[0], coords[1], tsunamiCRS);
      return latLng ? L.latLng(latLng[0], latLng[1]) : null;
    },
  }).addTo(map);
  console.log(
    `Rendered ${tsunamiGeoJson.features.length} tsunami points.`
  );
}
function updateAgentVisualization(geojsonData) {
  if (!map || !geojsonData || !geojsonData.features) return;
  if (agentLayer) {
    map.removeLayer(agentLayer);
  }
  agentLayer = L.featureGroup([], { pane: "agentPane" });
  let validPointCount = 0;
  let invalidPointCount = 0;

  geojsonData.features.forEach((feature) => {
    // Expect Point geometry now from the updated /api/export/geojson
    if (
      feature.geometry &&
      feature.geometry.type === "Point" &&
      feature.geometry.coordinates
    ) {
      const agentType = feature.properties.agent_type;
      const agentId = feature.properties.id; // Get agent ID
      const coordPair = feature.geometry.coordinates; // Direct coordinates for Point

      const latLng = convertToLatLng(
        coordPair[0],
        coordPair[1],
        currentCRS
      );
      if (latLng) {
        const marker = L.circleMarker(latLng, {
          radius: 6,
          fillColor: getAgentColor(agentType),
          color: "#000",
          weight: 1,
          opacity: 1,
          fillOpacity: 0.8,
        });
        // Store the agent ID from properties
        marker.options.agentId = agentId; // Store the ID
        marker.on("click", showAgentInfo); // Add click listener
        // marker.bindTooltip(`ID: ${agentId} Type: ${agentType}`); // Optional: Add ID to tooltip
        marker.addTo(agentLayer);
        validPointCount++;
      } else {
        console.warn(
          `Failed to convert coordinates for agent ${agentId}:`,
          coordPair
        );
        invalidPointCount++;
      }
    } else {
      // Log warning if geometry is not Point (or missing)
      console.warn(
        "Skipping feature with unexpected/missing geometry:",
        feature.geometry?.type
      );
      invalidPointCount++;
    }
  }); // End forEach feature

  agentLayer.addTo(map);
  if (!status.classList.contains("error-message")) {
    setStatusMessage(
      `Showing ${validPointCount} agents. ${
        invalidPointCount > 0
          ? `(${invalidPointCount} invalid/skipped)`
          : ""
      }`
    );
  }
  // Fit bounds only on initial load or reset
  // if (currentStep <= 1 && agentLayer.getLayers().length > 0) {
  //     const bounds = agentLayer.getBounds();
  //     if (bounds.isValid()) { map.fitBounds(bounds); }
  // }
}

// --- New Function to Show Agent Info ---
async function showAgentInfo(e) {
  const marker = e.target;
  const agentId = marker.options.agentId; // Retrieve stored agent ID
  console.log(
    "Clicked agent marker. Agent ID from marker options:",
    agentId
  ); // Log the ID

  // More robust check for valid ID
  if (agentId === undefined || agentId === null || agentId === "") {
    marker
      .bindPopup("Agent ID not available for this marker.")
      .openPopup();
    console.error(
      "Agent ID is undefined, null, or empty for the clicked marker."
    );
    return;
  }

  const fetchUrl = `api/agent/${agentId}`; // Use relative path
  // Removed duplicate line below
  console.log("Attempting to fetch:", fetchUrl);
  marker.bindPopup(`Loading info for Agent ${agentId}...`).openPopup(); // Show loading message with ID

  try {
    // Use the apiFetch helper function which handles the endpoint correctly
    const agentData = await apiFetch(fetchUrl); // Use apiFetch
    console.log("Received agent data:", agentData);

    // Format the agent data for display
    let infoString = `<strong>Agent ID: ${agentData.id}</strong><br>`;
    infoString += `Type: ${agentData.agent_type}<br>`;
    infoString += `Status: ${agentData.is_alive ? "Alive" : "Dead"}<br>`;
    infoString += `Position: (${agentData.x}, ${agentData.y})<br>`;
    infoString += `On Road: ${agentData.is_on_road}<br>`;
    infoString += `In Shelter: ${agentData.is_in_shelter}<br>`;
    infoString += `Knowledge: ${agentData.knowledge_level}<br>`;
    infoString += `Household Size: ${agentData.household_size}<br>`;
    infoString += `Decided to Evacuate: ${agentData.has_decided_to_evacuate}<br>`;
    infoString += `Trigger Time: ${
      agentData.evacuation_trigger_time ?? "N/A"
    }<br>`;
    infoString += `Milling Remaining: ${agentData.milling_steps_remaining}<br>`;
    // Add more fields as needed

    marker.bindPopup(infoString).openPopup();
  } catch (error) {
    marker.bindPopup(`Error loading info: ${error.message}`).openPopup();
  }
}

// --- Cost Grid Rendering (Replaces Heatmap) ---
function renderCostGridLayer(mode) {
  if (!map) return;
  // Clear previous cost layer
  if (costLayer) {
    map.removeLayer(costLayer);
    costLayer = null;
  }

  if (mode === "none" || !gridCostData) {
    console.log("Cost map display set to none or data not loaded.");
    return; // Don't render if mode is 'none' or data is missing
  }

  let valueArray; // Use generic name: cost or elevation
  let colorInterpolation; // Function to map value to color

  // Select data array and color function based on mode
  if (mode === "road") {
    valueArray = gridCostData.distance_to_road;
    // Lime (low) -> Yellow (mid) -> Red (high) for distance cost
    colorInterpolation = (normVal) => {
      let r, g;
      // Make transition to yellow/red happen sooner (e.g., around 0.3 instead of 0.5)
      const transitionPoint = 0.3; // Adjust this value (0.0 to 1.0)
      if (normVal < transitionPoint) {
        // Interpolate Lime (0, 255, 0) to Yellow (255, 255, 0)
        r = Math.round(255 * (normVal / transitionPoint));
        g = 255;
      } else {
        // Interpolate Yellow (255, 255, 0) to Red (255, 0, 0)
        r = 255;
        g = Math.round(
          255 * (1 - (normVal - transitionPoint) / (1 - transitionPoint))
        );
      }
      // Ensure g stays within 0-255
      g = Math.max(0, Math.min(255, g));
      return `rgb(${r}, ${g}, 0)`;
    };
  } else if (mode === "shelter") {
    valueArray = gridCostData.distance_to_shelter;
    // Use same adjusted Lime -> Red gradient for shelter distance
    colorInterpolation = (normVal) => {
      let r, g;
      const transitionPoint = 0.3; // Keep consistent with 'road'
      if (normVal < transitionPoint) {
        r = Math.round(255 * (normVal / transitionPoint));
        g = 255;
      } else {
        r = 255;
        g = Math.round(
          255 * (1 - (normVal - transitionPoint) / (1 - transitionPoint))
        );
      }
      g = Math.max(0, Math.min(255, g));
      return `rgb(${r}, ${g}, 0)`;
    };
  } else if (mode === "dtm") {
    valueArray = gridCostData.environment_layers?.dtm; // Access the DTM layer
    // Grayscale for DTM: Black (low) to White (high)
    colorInterpolation = (normVal) => {
      const intensity = Math.round(255 * normVal);
      return `rgb(${intensity}, ${intensity}, ${intensity})`;
    };
  } else {
    console.error(`Unknown cost map mode: ${mode}`);
    return;
  }

  // Ensure the selected data array exists
  if (!valueArray) {
    console.error(`Data for mode '${mode}' not found in gridCostData.`);
    setStatusMessage(`Data for mode '${mode}' not available.`, true);
    return;
  }

  const { nrows, ncols, xllcorner, yllcorner, cellsize } = gridCostData;
  const gridCRS = currentCRS || "EPSG:32749"; // Assume CRS if needed

  // Find min/max for normalization from the selected valueArray
  let maxValue = -Infinity;
  let minValue = Infinity;
  let validValueCount = 0;
  for (let r = 0; r < nrows; r++) {
    for (let c = 0; c < ncols; c++) {
      // Use optional chaining for safety
      const value = valueArray[r]?.[c];
      if (value !== null && value !== undefined) {
        validValueCount++;
        if (value > maxValue) maxValue = value;
        if (value < minValue) minValue = value;
      }
    }
  }
  // Handle edge cases for normalization range
  if (minValue === Infinity) minValue = 0; // No valid values found
  if (maxValue === -Infinity) maxValue = minValue; // No valid values or all same
  const valueRange = Math.max(1e-6, maxValue - minValue); // Use epsilon

  console.log(
    `Rendering grid for '${mode}'. Min: ${minValue}, Max: ${maxValue}, Range: ${valueRange}, Valid Cells: ${validValueCount}`
  );

  // Create a feature group for the cost rectangles, assign to costPane
  costLayer = L.featureGroup([], { pane: "costPane" });

  // Iterate and create rectangles
  for (let r = 0; r < nrows; r++) {
    for (let c = 0; c < ncols; c++) {
      const value = valueArray[r]?.[c]; // Use value instead of cost

      // Only draw cells with a valid value
      if (value !== null && value !== undefined) {
        // Calculate cell bounds
        const utmX1 = xllcorner + c * cellsize;
        const utmY1 = yllcorner + (nrows - 1 - r) * cellsize; // Bottom-left Y
        const utmX2 = utmX1 + cellsize;
        const utmY2 = utmY1 + cellsize; // Top Y

        const ll = convertToLatLng(utmX1, utmY1, gridCRS);
        const ur = convertToLatLng(utmX2, utmY2, gridCRS);

        if (ll && ur) {
          const bounds = L.latLngBounds([
            [ll[0], ll[1]],
            [ur[0], ur[1]],
          ]);
          // Normalize value to 0-1 range
          const normalizedValue = (value - minValue) / valueRange;

          // Log values for non-road cells for debugging
          if (mode === "road" && value > 0) {
            console.log(
              `  Cell(${r},${c}): Cost=${value}, Norm=${normalizedValue.toFixed(
                3
              )}`
            );
          }

          // Get color using the selected interpolation function
          const color = colorInterpolation(normalizedValue);

          L.rectangle(bounds, {
            color: color,
            weight: 0,
            fillColor: color,
            fillOpacity: 0.45, // Adjust opacity
          }).addTo(costLayer);
        }
      }
    }
  }
  costLayer.addTo(map); // Add the layer group to the map
  setStatusMessage(`Rendered grid for '${mode}'.`);
}

// --- New Function to Render Siren Layer ---
function renderSirenLayer() {
    if (!map) return;
    // Clear previous siren layer
    if (sirenLayer) {
        map.removeLayer(sirenLayer);
        sirenLayer = null;
    }

    // Check if siren config and grid metadata are available
    if (sirenConfig && gridCostData) {
        const { xllcorner, yllcorner, cellsize, nrows } = gridCostData; // Use nrows for y conversion if needed
        const { x: sirenGridX, y: sirenGridY, radius_cells } = sirenConfig;

        // Calculate UTM coordinates of the siren center
        // Note: Y-axis calculation might need adjustment based on grid origin (bottom-left vs top-left)
        // Assuming grid origin is bottom-left as is common for ASC
        const sirenUtmX = xllcorner + (sirenGridX + 0.5) * cellsize; // Center of the cell
        const sirenUtmY = yllcorner + (sirenGridY + 0.5) * cellsize; // Center of the cell

        // Convert UTM center to LatLng
        const sirenLatLng = convertToLatLng(sirenUtmX, sirenUtmY, currentCRS);

        if (sirenLatLng) {
            // Calculate radius in meters
            const sirenRadiusMeters = radius_cells * cellsize;

            // Create the circle layer
            sirenLayer = L.circle(sirenLatLng, {
                radius: sirenRadiusMeters,
                color: 'red',       // Outline color
                fillColor: '#f03',  // Fill color
                fillOpacity: 0.2, // Semi-transparent fill
                weight: 1           // Outline weight
            }).bindTooltip(`Siren Radius: ${radius_cells} cells (${sirenRadiusMeters.toFixed(0)}m)`);

            // Add to map
            sirenLayer.addTo(map);
            console.log(`Siren visualized at [${sirenLatLng}] with radius ${sirenRadiusMeters}m.`);
        } else {
            console.error("Could not convert siren UTM coordinates to LatLng.");
        }
    } else {
        console.log("Siren config or grid cost data not available for visualization.");
    }
}

// --- Main Refresh Function ---
async function refreshDataAndMap() {
  setStatusMessage("Fetching latest data...");
  const metadata = await fetchMetadata(); // Updates global state and buttons
  if (metadata && isSimulationInitialized) {
    // Fetch agent and tsunami data concurrently
    const [agentGeoJson, tsunamiGeoJson] = await Promise.all([
      fetchCurrentGeojsonData(),
      apiFetch("/api/tsunami/geojson"), // Fetch tsunami data
    ]);

    if (agentGeoJson) {
      updateAgentVisualization(agentGeoJson); // Update agent layer
    }
    renderTsunamiLayer(tsunamiGeoJson); // Render tsunami layer
    // Cost grid is NOT redrawn on every refresh automatically
    // It's only redrawn when the radio button selection changes or on init/reset
  } else {
    // If not initialized, ensure other layers are also cleared
    if (tsunamiLayer) map.removeLayer(tsunamiLayer);
    tsunamiLayer = null;
    if (costLayer) map.removeLayer(costLayer); // Clear cost grid
    costLayer = null;
  }
  // If simulation completed while polling, stop
  if (isSimulationCompleted && isPolling) {
    togglePolling();
    setStatusMessage("Simulation completed.");
  }
}