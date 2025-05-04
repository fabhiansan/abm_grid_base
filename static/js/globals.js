// --- Global Variables ---
let map = null;
let currentStep = 0;
let totalAgents = 0;
let isPolling = false;
let isSimulationInitialized = false;
let isSimulationRunning = false;
let isSimulationCompleted = false;
let pollingInterval = null;
const pollingFrequency = 1500; // ms
let sirenConfig = null; // Store full siren config object { activation_step, x, y, radius_cells }
let agentLayer = null;
let gridLayer = null;
let tsunamiLayer = null;
let costLayer = null; // Layer for cost grid rectangles
let currentCRS = null;
let gridCostData = null; // To store fetched cost data
let deathChart = null; // Chart instance variables
let shelterChart = null;
let sirenLayer = null; // Layer for siren visualization

// --- DOM Elements ---
const initBtn = document.getElementById("init-simulation");
const playPauseBtn = document.getElementById("play-pause");
const resetBtn = document.getElementById("reset-simulation");
const runStepBtn = document.getElementById("run-single-step");
const runStepsInput = document.getElementById("run-steps-input");
const runStepsBtn = document.getElementById("run-multiple-steps");
const healthBtn = document.getElementById("check-health");
const getConfigBtn = document.getElementById("get-config");
const updateConfigBtn = document.getElementById("update-config");
const configTextarea = document.getElementById("config-textarea");
const configDisplay = document.getElementById("config-display");
const useDtmCheckbox = document.getElementById("use-dtm-checkbox"); // Get checkbox element
const exportBtn = document.getElementById("export-results");
const exportDisplay = document.getElementById("export-display");
const infoDisplay = document.getElementById("info-display");
const status = document.getElementById("status");
const costmapRadios = document.querySelectorAll('input[name="costmap"]');
const deathChartCanvas = document.getElementById("deathChartCanvas");
const shelterChartCanvas = document.getElementById("shelterChartCanvas");
// Removed heatmapCanvas