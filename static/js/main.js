// --- Initial Load ---
async function init() {
  setStatusMessage("Initializing map...", false);
  initMap();
  await fetchMetadata(); // Fetch initial state
  // Don't fetch/render costs until simulation is initialized
  if (!status.classList.contains("error-message")) {
    setStatusMessage("Ready. Initialize simulation.");
  }
  updateButtonStates(); // Ensure buttons reflect initial state
}

// Start the application
init();