// --- Map Initialization ---
function initMap() {
  map = L.map("map").setView([-8.4095, 115.1889], 8);
  // Create panes for layering
  map.createPane("costPane"); // Pane for cost grid layer
  map.getPane("costPane").style.zIndex = 425; // Below grid features
  map.createPane("gridPane");
  map.getPane("gridPane").style.zIndex = 450; // Roads below agents
  map.createPane("tsunamiPane");
  map.getPane("tsunamiPane").style.zIndex = 475; // Tsunami below agents but above grid
  map.createPane("agentPane");
  map.getPane("agentPane").style.zIndex = 500; // Agents
  map.createPane("shelterPane");
  map.getPane("shelterPane").style.zIndex = 550; // Shelters on top

  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  }).addTo(map);
  L.control.scale().addTo(map);
  addLegend();
  // Removed heatmap init and map move listener
  return map;
}
function addLegend() {
  const legend = L.control({ position: "bottomright" });
  legend.onAdd = function (map) {
    const div = L.DomUtil.create("div", "legend");
    div.innerHTML = `
              <strong>Legend</strong><br>
              <i class="road-icon" style="background: #888"></i> Road<br>
              <i class="shelter-icon" style="background: #006400"></i> Shelter<br>
              <i class="tsunami-icon" style="background: blue;"></i> Tsunami<br>
              <div class="cost-gradient" style="background: linear-gradient(to right, lime, yellow, red);"></div> <!-- Updated Gradient -->
              <span style="float: left; font-size: 0.8em;">Low</span>
              <span style="float: right; font-size: 0.8em;">High</span>
              <div style="clear: both;"></div>
              <hr style="margin: 5px 0;">
              <strong>Agent Types</strong><br>
              <i style="background: #ff0000"></i> Adult<br>
              <i style="background: #00ff00"></i> Child<br>
              <i style="background: #0000ff"></i> Elder<br>
              <i style="background: #ffA500"></i> Teen<br>
              <i style="background: #999999"></i> Unknown/Other
          `;
    return div;
  };
  legend.addTo(map);
}