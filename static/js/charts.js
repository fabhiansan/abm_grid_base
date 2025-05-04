// --- Chart Rendering Functions ---
function renderDeathChart(steps, deathCounts) {
  if (!deathChartCanvas) return;
  const ctx = deathChartCanvas.getContext("2d");

  // Extract total deaths per step
  const totalDeaths = deathCounts.map((d) => d.dead_agents.total);

  // Destroy existing chart instance before creating a new one
  if (deathChart) {
    deathChart.destroy();
  }

  // Create a new chart instance
  deathChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: steps,
      datasets: [
        {
          label: "Total Dead Agents",
          data: totalDeaths,
          borderColor: "rgb(220, 53, 69)", // Red
          tension: 0.1,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Count" } },
        x: { title: { display: true, text: "Step" } },
      },
      plugins: { legend: { display: true } },
    },
  });
}

function renderShelterChart(steps, shelterData) {
  if (!shelterChartCanvas) return;
  const ctx = shelterChartCanvas.getContext("2d");

  // Process shelter data for stacked bar chart
  const shelterIDs = new Set();
  shelterData.forEach((stepData) => {
    Object.keys(stepData.shelters).forEach((key) => shelterIDs.add(key));
  });
  const sortedShelterIDs = Array.from(shelterIDs).sort();

  const datasets = sortedShelterIDs.map((shelterId) => {
    const data = steps.map((step) => {
      const stepEntry = shelterData.find((sd) => sd.step === step);
      const shelterCounts = stepEntry?.shelters?.[shelterId];
      return shelterCounts
        ? shelterCounts.child +
            shelterCounts.teen +
            shelterCounts.adult +
            shelterCounts.elder
        : 0;
    });
    // Basic color cycling - could be improved
    const colorIndex = sortedShelterIDs.indexOf(shelterId) % 5; // Example: cycle through 5 colors
    const colors = [
      "rgba(40, 167, 69, 0.7)",
      "rgba(23, 162, 184, 0.7)",
      "rgba(255, 193, 7, 0.7)",
      "rgba(108, 117, 125, 0.7)",
      "rgba(52, 58, 64, 0.7)",
    ];

    return {
      label: shelterId,
      data: data,
      backgroundColor: colors[colorIndex],
    };
  });

  if (shelterChart) {
    shelterChart.data.labels = steps;
    shelterChart.data.datasets = datasets;
    shelterChart.update();
  } else {
    shelterChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: steps,
        datasets: datasets,
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { stacked: true, title: { display: true, text: "Step" } },
          y: {
            stacked: true,
            beginAtZero: true,
            title: { display: true, text: "Agents in Shelter" },
          },
        },
      plugins: { legend: { display: true } },
      },
    });
  }
}