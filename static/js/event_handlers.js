// --- Event Listeners ---
initBtn.addEventListener("click", async () => {
  setStatusMessage("Resetting simulation state before init...", false);
  initBtn.disabled = true;
  playPauseBtn.disabled = true;
  resetBtn.disabled = true;
  gridCostData = null; // Clear cached cost data on init
  if (costLayer) map.removeLayer(costLayer);
  costLayer = null; // Clear cost layer

  try {
    try {
      await apiFetch("/api/reset", { method: "POST" });
      setStatusMessage("Previous state reset. Now initializing...");
    } catch (resetError) {
      console.warn("Reset before init failed:", resetError);
      setStatusMessage("Reset failed. Attempting initialization...");
    }

    const result = await apiFetch("/init_simulation", { method: "POST" });
    setStatusMessage(`Simulation initialized: ${result.message || "OK"}`);
    isSimulationInitialized = true;

    // Fetch grid and cost data *after* successful init
    await fetchGridCosts(); // Fetch and cache cost data
    await renderGridLayer(); // Render static grid
    await refreshDataAndMap(); // Fetch initial agent/tsunami state and map

    if (agentLayer && map && agentLayer.getLayers().length > 0) {
      const bounds = agentLayer.getBounds();
      if (bounds.isValid()) map.fitBounds(bounds);
    }
    // Draw initial cost grid based on selection AFTER data is fetched
    renderCostGridLayer(
      document.querySelector('input[name="costmap"]:checked').value
    );
    // Render siren layer after init and data fetch
    renderSirenLayer();
  } catch (error) {
    initBtn.disabled = false;
  } finally {
    updateButtonStates();
  }
});

playPauseBtn.addEventListener("click", togglePolling);

resetBtn.addEventListener("click", async () => {
  /* ... (keep existing implementation, clear cost data/layer) ... */
  if (!confirm("Are you sure?")) return;
  setStatusMessage("Resetting simulation...", false);
  if (isPolling) togglePolling();
  try {
    const result = await apiFetch("/api/reset", { method: "POST" });
    setStatusMessage(`Simulation reset: ${result.message || "OK"}`);
    isSimulationInitialized = false;
    isSimulationRunning = false;
    isSimulationCompleted = false;
    currentStep = 0;
    totalAgents = 0;
    sirenConfig = null; // Clear siren config on reset
    if (agentLayer) map.removeLayer(agentLayer);
    if (gridLayer) map.removeLayer(gridLayer);
    if (tsunamiLayer) map.removeLayer(tsunamiLayer);
    if (costLayer) map.removeLayer(costLayer);
    agentLayer = null;
    gridLayer = null;
    tsunamiLayer = null;
    costLayer = null;
    gridCostData = null;
    if (sirenLayer) map.removeLayer(sirenLayer); // Clear siren layer on reset
    sirenLayer = null;
    await fetchMetadata();
  } catch (error) {
    /* Handled */
  } finally {
    updateButtonStates();
  }
});

runStepBtn.addEventListener("click", async () => {
  /* ... (keep existing implementation) ... */
  setStatusMessage("Running single step...", false);
  runStepBtn.disabled = true;
  runStepsBtn.disabled = true;
  try {
    const result = await apiFetch("/api/step", { method: "POST" });
    setStatusMessage(
      `Step executed. Current step: ${
        result?.simulation_state?.current_step || "N/A"
      }`
    );
    await refreshDataAndMap();
  } catch (error) {
    /* Handled */
  } finally {
    updateButtonStates();
  }
});

runStepsBtn.addEventListener("click", async () => {
  /* ... (keep existing implementation) ... */
  const steps = parseInt(runStepsInput.value);
  if (isNaN(steps) || steps < 1) {
    setStatusMessage("Invalid number of steps.", true);
    return;
  }
  setStatusMessage(`Running ${steps} steps...`, false);
  runStepBtn.disabled = true;
  runStepsBtn.disabled = true;
  try {
    const result = await apiFetch(`/api/run/${steps}`, {
      method: "POST",
      timeout: 60000,
    });
    setStatusMessage(
      `${result?.steps_executed || 0} steps executed. Current step: ${
        result?.simulation_state?.current_step || "N/A"
      }`
    );
    await refreshDataAndMap();
  } catch (error) {
    /* Handled */
  } finally {
    updateButtonStates();
  }
});

healthBtn.addEventListener("click", async () => {
  /* ... (keep existing implementation) ... */
  try {
    const result = await apiFetch("/api/health");
    setStatusMessage(`API Health: ${result.status} - ${result.message}`);
  } catch (error) {
    /* Handled */
  }
});

getConfigBtn.addEventListener("click", async () => {
  try {
    const result = await apiFetch("/api/config");
    // Populate textarea, new input fields, and checkbox
    configTextarea.value = JSON.stringify(result, null, 2);
    document.getElementById("tsunami_delay").value =
      result.tsunami_delay ?? 100; // Use default if null/undefined
    // Check if agent_reaction_delay input exists before setting value
    const agentReactionDelayInput = document.getElementById("agent_reaction_delay");
    if (agentReactionDelayInput && result.hasOwnProperty('agent_reaction_delay')) {
         agentReactionDelayInput.value = result.agent_reaction_delay ?? 50; // Use default if null/undefined
    }
    // Populate new fields
    document.getElementById("tsunami_speed_time").value = result.tsunami_speed_time ?? 60;
    document.getElementById("data_collection_interval").value = result.data_collection_interval ?? 30;
    document.getElementById("milling_time_min").value = result.milling_time_min ?? 5;
    document.getElementById("milling_time_max").value = result.milling_time_max ?? 20;
    document.getElementById("siren_effectiveness_probability").value = result.siren_effectiveness_probability ?? 0.8;
    document.getElementById("knowledge_level_min").value = result.knowledge_level_min ?? 10;
    document.getElementById("knowledge_level_max").value = result.knowledge_level_max ?? 90;
    document.getElementById("household_size_min").value = result.household_size_min ?? 1;
    document.getElementById("household_size_max").value = result.household_size_max ?? 5;

    useDtmCheckbox.checked = result.use_dtm_for_cost === true;
    configDisplay.textContent = "Current config loaded into fields."; // Update display message
    setStatusMessage("Current config loaded.");
  } catch (error) {
    configDisplay.textContent = `Error loading config: ${error.message}`;
    setStatusMessage(`Error loading config: ${error.message}`, true);
  }
});

updateConfigBtn.addEventListener("click", async () => {
  setStatusMessage("Fetching current config to update...", false);
  let currentConfig;
  try {
    currentConfig = await apiFetch("/api/config"); // Fetch current config first
  } catch (error) {
    setStatusMessage(
      "Failed to fetch current config. Cannot update.",
      true
    );
    return;
  }

  // Read values from specific input fields
  const tsunamiDelayInput = document.getElementById("tsunami_delay");
  const tsunamiSpeedTimeInput = document.getElementById("tsunami_speed_time");
  const dataCollectionIntervalInput = document.getElementById("data_collection_interval");
  const millingTimeMinInput = document.getElementById("milling_time_min");
  const millingTimeMaxInput = document.getElementById("milling_time_max");
  const sirenEffectivenessInput = document.getElementById("siren_effectiveness_probability");
  const knowledgeLevelMinInput = document.getElementById("knowledge_level_min");
  const knowledgeLevelMaxInput = document.getElementById("knowledge_level_max");
  const householdSizeMinInput = document.getElementById("household_size_min");
  const householdSizeMaxInput = document.getElementById("household_size_max");
  // Removed reference to agent_reaction_delay input

  // Parse values from inputs, providing defaults based on fetched config or hardcoded defaults
  const tsunamiDelay =
    parseInt(tsunamiDelayInput.value) ||
    currentConfig.tsunami_delay ||
    100;
  const tsunamiSpeedTime = parseInt(tsunamiSpeedTimeInput.value) || currentConfig.tsunami_speed_time || 60;
  const dataCollectionInterval = parseInt(dataCollectionIntervalInput.value) || currentConfig.data_collection_interval || 30;
  const millingTimeMin = parseInt(millingTimeMinInput.value) || currentConfig.milling_time_min || 5;
  const millingTimeMax = parseInt(millingTimeMaxInput.value) || currentConfig.milling_time_max || 20;
  const sirenEffectiveness = parseFloat(sirenEffectivenessInput.value) || currentConfig.siren_effectiveness_probability || 0.8;
  const knowledgeLevelMin = parseInt(knowledgeLevelMinInput.value) || currentConfig.knowledge_level_min || 10;
  const knowledgeLevelMax = parseInt(knowledgeLevelMaxInput.value) || currentConfig.knowledge_level_max || 90;
  const householdSizeMin = parseInt(householdSizeMinInput.value) || currentConfig.household_size_min || 1;
  const householdSizeMax = parseInt(householdSizeMaxInput.value) || currentConfig.household_size_max || 5;
  // Removed parsing of agent_reaction_delay
  const useDtm = useDtmCheckbox.checked;

  // Update the fetched config object
  currentConfig.tsunami_delay = tsunamiDelay;
  currentConfig.tsunami_speed_time = tsunamiSpeedTime;
  currentConfig.data_collection_interval = dataCollectionInterval;
  currentConfig.milling_time_min = millingTimeMin;
  currentConfig.milling_time_max = millingTimeMax;
  currentConfig.siren_effectiveness_probability = sirenEffectiveness;
  currentConfig.knowledge_level_min = knowledgeLevelMin; // Add new value
  currentConfig.knowledge_level_max = knowledgeLevelMax; // Add new value
  currentConfig.household_size_min = householdSizeMin; // Add new value
  currentConfig.household_size_max = householdSizeMax; // Add new value
  // Removed assignment of agent_reaction_delay to currentConfig
  currentConfig.use_dtm_for_cost = useDtm;

  // Update the fetched config object *before* sending
  configTextarea.value = JSON.stringify(currentConfig, null, 2);

  setStatusMessage("Sending updated config...", false);
  try {
    const result = await apiFetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentConfig), // Send the updated object
    });
    setStatusMessage(`Config updated: ${result.message || "OK"}`);
    // Optionally refresh the display again to confirm, or rely on the textarea update above
    // getConfigBtn.click();
    configDisplay.textContent = "Config updated successfully.";
  } catch (error) {
    /* Error message already set by apiFetch */
  }
});

exportBtn.addEventListener("click", async () => {
  /* ... (keep existing implementation) ... */
  try {
    const result = await apiFetch("/api/export");
    exportDisplay.textContent = JSON.stringify(result, null, 2); // Show raw data too

    // Process and render charts
    if (result.death_data && result.shelter_data) {
      const steps = result.death_data.map((d) => d.step);
      renderDeathChart(steps, result.death_data);
      renderShelterChart(steps, result.shelter_data);
      setStatusMessage("Data exported and charts updated.");
    } else {
      setStatusMessage("Data exported, but chart data missing.", true);
    }
  } catch (error) {
    exportDisplay.textContent = `Error exporting/charting data: ${error.message}`;
    setStatusMessage(
      `Error exporting/charting data: ${error.message}`,
      true
    );
  }
});

// Listener for cost map radio buttons
costmapRadios.forEach((radio) => {
  radio.addEventListener("change", async (event) => {
    const selectedMode = event.target.value;
    if (selectedMode !== "none") {
      // Always fetch fresh cost data when switching TO a cost map view
      setStatusMessage("Fetching latest cost data for visualization...");
      await fetchGridCosts(); // Fetch fresh data
      if (!gridCostData) {
        setStatusMessage(
          `Failed to load cost data for mode '${selectedMode}'.`,
          true
        );
        return; // Don't render if fetch failed
      }
    }
    // Render (or clear if mode is 'none')
    renderCostGridLayer(selectedMode);
  });
});

// Listener for Agent Outcomes Export Button
const exportOutcomesBtn = document.getElementById("export-agent-outcomes");
const outcomesDisplay = document.getElementById("agent-outcomes-display");
exportOutcomesBtn.addEventListener("click", async () => {
    setStatusMessage("Fetching agent outcomes...", false);
    outcomesDisplay.textContent = "Loading outcomes...";
    try {
        const result = await apiFetch("/api/export/agent_outcomes"); // Call the new endpoint
        if (result && result.outcomes) {
            outcomesDisplay.textContent = JSON.stringify(result.outcomes, null, 2);
            setStatusMessage(`Agent outcomes loaded (${result.outcomes.length} agents).`);
        } else {
            outcomesDisplay.textContent = `Error: ${result.message || 'Invalid data received'}`;
            setStatusMessage(`Failed to load outcomes: ${result.message || 'Invalid data'}`, true);
        }
    } catch (error) {
        outcomesDisplay.textContent = `Error fetching outcomes: ${error.message}`;
        setStatusMessage(`Error fetching outcomes: ${error.message}`, true);
    }
});