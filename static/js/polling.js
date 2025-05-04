// --- Polling Control ---
function togglePolling() {
  /* ... (keep existing implementation) ... */
  isPolling = !isPolling;
  playPauseBtn.textContent = isPolling
    ? "Pause Polling"
    : "Resume Polling";
  updateButtonStates();
  if (isPolling) {
    refreshDataAndMap();
    pollingInterval = setInterval(refreshDataAndMap, pollingFrequency);
    setStatusMessage("Polling for live data...");
  } else {
    clearInterval(pollingInterval);
    pollingInterval = null;
    if (!status.classList.contains("error-message")) {
      setStatusMessage("Polling paused.");
    }
  }
}