use crate::game::agent::{Agent, AgentType};
use crate::game::game::Model;
use crate::game::grid::{load_grid_from_ascii, Grid, Terrain}; // Added Terrain import
use crate::{
    export_agent_statistics, export_agents_to_geojson, load_population_and_create_agents,
    read_tsunami_data, DeadAgentTypeData, ShelterAgentTypeData, TSUNAMI_DELAY, TSUNAMI_SPEED_TIME,
};

use actix_cors::Cors;
use actix_web::{
    get, post, web, App, HttpResponse, HttpServer, Responder, middleware::Logger,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
// use log::{info, warn}; // Keep commented unless needed


// Configuration for simulation
#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationConfig {
    pub location: String,
    pub grid_path: String,
    pub population_path: String,
    pub tsunami_data_path: String,
    pub output_path: String,
    pub max_steps: Option<u32>,
    pub dtm_path: Option<String>, // Optional path for DTM data
    pub use_dtm_for_cost: bool, // Flag to enable/disable DTM in cost calculation
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            location: "sample".to_string(),
            grid_path: "./data_sample/sample_grid.asc".to_string(),
            population_path: "./data_sample/sample_agents.asc".to_string(),
            tsunami_data_path: "./data_sample/tsunami_ascii_sample".to_string(),
            output_path: "./output".to_string(),
            max_steps: None,
            // Default to the newly generated sample DTM path
            dtm_path: Some("./data_sample/sample_dtm.asc".to_string()),
            use_dtm_for_cost: true, // Default to using DTM if available
        }
    }
}

impl SimulationConfig {
    // Get location-specific paths
    pub fn get_location_paths(&self) -> (String, String, String) {
        match self.location.as_str() {
            "pacitan" => (
                format!("./data_pacitan/jalantes_pacitan.asc"),
                format!("./data_pacitan/agent_pacitan.asc"),
                format!("./data_pacitan/tsunami_ascii_pacitan")
            ),
            "sample" => (
                format!("./data_sample/sample_grid.asc"),
                format!("./data_sample/sample_agents.asc"),
                format!("./data_sample/tsunami_ascii_sample")
            ),
            _ => (
                format!("./data_jembrana/jalantes_jembrana.asc"),
                format!("./data_jembrana/agen_jembrana.asc"),
                format!("./data_jembrana/tsunami_ascii_jembrana")
            ),
        }
    }
}

// Current state of simulation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimulationState {
    pub current_step: u32,
    pub is_tsunami: bool,
    pub tsunami_index: usize,
    pub dead_agents: usize,
    pub is_running: bool,
    pub is_completed: bool,
}

// Result of simulation step
#[derive(Serialize, Deserialize)]
pub struct StepResult {
    pub step: u32,
    pub dead_agents: usize,
    pub dead_agent_types: DeadAgentTypeData,
    pub shelter_data: HashMap<String, ShelterAgentTypeData>,
}

// Application state
pub struct AppState {
    pub config: SimulationConfig,
    pub state: SimulationState,
    pub model: Option<Model>,
    pub death_json_counter: Vec<serde_json::Value>,
    pub shelter_json_counter: Vec<serde_json::Value>,
}

// --- API Handlers ---

#[get("/health")]
async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(json!({"status": "ok", "message": "Service is running"}))
}

#[get("/config")]
async fn get_config(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let app_state = data.lock().unwrap();
    HttpResponse::Ok().json(&app_state.config)
}

#[post("/config")]
async fn update_config(data: web::Data<Arc<Mutex<AppState>>>, config: web::Json<SimulationConfig>) -> impl Responder {
    let mut app_state = data.lock().unwrap();
    app_state.config = config.into_inner();
    HttpResponse::Ok().json(json!({"status": "ok", "message": "Configuration updated"}))
}

#[post("/init")]
async fn init_simulation(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let mut app_state = data.lock().unwrap();
    if app_state.state.is_running {
        return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Simulation is already running or initialized. Reset first."}));
    }
    app_state.state = SimulationState::default();
    app_state.death_json_counter = Vec::new();
    app_state.shelter_json_counter = Vec::new();
    app_state.model = None;
    let config = app_state.config.clone();
    let (grid_path, population_path, tsunami_data_path) = config.get_location_paths();
    drop(app_state); // Release lock before I/O

    println!("Initializing simulation for location: {}", config.location);
    println!("Using grid path: {}", grid_path);
    println!("Using population path: {}", population_path);
    println!("Using tsunami data path: {}", tsunami_data_path);

    // Load base grid (terrain, shelters)
    let (mut grid, mut agents) = match load_grid_from_ascii(&grid_path) {
         Ok(data) => data,
         Err(e) => {
            eprintln!("Failed to load grid from '{}': {}", grid_path, e);
            return HttpResponse::InternalServerError().json(json!({"status": "error", "message": format!("Failed to load grid file '{}': {}", grid_path, e)}));
         }
    };

    // Load population agents
    let mut next_agent_id = agents.len(); // Start ID after any agents potentially loaded from grid file (though we removed that)
    if let Err(e) = load_population_and_create_agents(&population_path, grid.width, grid.height, &mut grid, &mut agents, &mut next_agent_id) {
        eprintln!("Failed to load population from '{}': {}", population_path, e);
        return HttpResponse::InternalServerError().json(json!({"status": "error", "message": format!("Failed to load or process population file '{}': {}", population_path, e)}));
    }

    // Load tsunami data
     let tsunami_data = match read_tsunami_data(&tsunami_data_path, grid.ncol, grid.nrow) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("Failed to read tsunami data from '{}': {}", tsunami_data_path, e);
            return HttpResponse::InternalServerError().json(json!({"status": "error", "message": format!("Failed to read tsunami data from directory '{}': {}", tsunami_data_path, e)}));
        }
    };
    let tsunami_len = tsunami_data.len();
    grid.tsunami_data = tsunami_data;

    // Load Optional DTM Layer
    if let Some(dtm_path) = &config.dtm_path {
        println!("Loading DTM data from: {}", dtm_path);
        match crate::game::grid::load_float_asc_layer(dtm_path) {
            Ok((dtm_data, dtm_ncols, dtm_nrows, _, _, _)) => {
                if dtm_ncols == grid.width && dtm_nrows == grid.height {
                    grid.environment_layers.insert("dtm".to_string(), dtm_data);
                    println!("  Successfully loaded DTM data.");
                    // Distances will be recomputed after model creation
                } else {
                    eprintln!("Error: DTM dimensions ({}x{}) do not match grid dimensions ({}x{}). DTM not loaded.", dtm_ncols, dtm_nrows, grid.width, grid.height);
                }
            }
            Err(e) => { eprintln!("Error loading DTM data from '{}': {}. Proceeding without DTM.", dtm_path, e); }
        }
    } else {
        println!("No DTM path provided in config. Skipping DTM loading.");
    }

    // Create Model
    let mut model = Model { grid, agents, dead_agents: 0, dead_agent_types: Vec::new() };

    // Recompute distances if DTM was loaded, passing the config flag
    if config.dtm_path.is_some() && model.grid.environment_layers.contains_key("dtm") {
         println!("Recomputing distances using config setting use_dtm_for_cost: {}", config.use_dtm_for_cost);
         model.grid.compute_distance_to_shelters(config.use_dtm_for_cost);
         model.grid.compute_distance_to_road(config.use_dtm_for_cost);
    } else {
        // If DTM wasn't loaded or path not specified, compute without DTM flag
        println!("Recomputing distances without DTM (DTM not loaded or path not specified).");
        model.grid.compute_distance_to_shelters(false);
        model.grid.compute_distance_to_road(false);
    }

    // Re-acquire lock and update state
    let mut app_state = data.lock().unwrap();
    app_state.model = Some(model); // Store the final model
    app_state.state.is_running = true;
    app_state.state.is_completed = false;
    let final_agent_count = app_state.model.as_ref().unwrap().agents.len();
    if let Err(e) = export_agent_statistics(&app_state.model.as_ref().unwrap().agents) { eprintln!("Warning: Failed to export agent statistics: {}", e); }

    HttpResponse::Ok().json(json!({"status": "ok", "message": "Simulation initialized", "details": {"tsunami_data_length": tsunami_len, "agents_count": final_agent_count}}))
}

#[post("/step")]
async fn run_step(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let mut app_state = data.lock().unwrap();
    if app_state.model.is_none() { return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Simulation not initialized"})); }
    if app_state.state.is_completed { return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Simulation already completed"})); }

    let current_step = app_state.state.current_step;
    let is_tsunami = app_state.state.is_tsunami;
    let tsunami_index = app_state.state.tsunami_index;
    let model = app_state.model.as_mut().unwrap();
    model.step(current_step, is_tsunami, tsunami_index);

    // Collect results
    let mut dead_agent_counts = DeadAgentTypeData::default(); for agent_type in &model.dead_agent_types { match agent_type { AgentType::Child => dead_agent_counts.child += 1, AgentType::Teen => dead_agent_counts.teen += 1, AgentType::Adult => dead_agent_counts.adult += 1, AgentType::Elder => dead_agent_counts.elder += 1, } dead_agent_counts.total += 1; }
    let dead_agents_count = model.dead_agents;
    let shelter_info: HashMap<String, ShelterAgentTypeData> = model.grid.shelters.iter().map(|&(_, _, id)| { let key = format!("shelter_{}_{}", id, current_step); let count = model.grid.shelter_agents.get(&id).map(|agents_in_shelter| { let mut data = ShelterAgentTypeData::default(); for agent_tuple in agents_in_shelter { match agent_tuple.1 { AgentType::Child => data.child += 1, AgentType::Teen => data.teen += 1, AgentType::Adult => data.adult += 1, AgentType::Elder => data.elder += 1, } } data }).unwrap_or_default(); (key, count) }).collect();

    // Check state changes
    let mut new_tsunami_index = tsunami_index;
    let mut should_complete_simulation = false;
    let should_update_tsunami = current_step >= TSUNAMI_DELAY;
    let tsunami_data_len = model.grid.tsunami_data.len();
    let should_advance_tsunami_index = should_update_tsunami && current_step > 0 && current_step % TSUNAMI_SPEED_TIME == 0;
    drop(model); // Release model borrow

    // Update state
     if should_update_tsunami { app_state.state.is_tsunami = true; if should_advance_tsunami_index { if tsunami_index + 1 >= tsunami_data_len { should_complete_simulation = true; } else { new_tsunami_index = tsunami_index + 1; } } }
     let step_result = StepResult { step: current_step, dead_agents: dead_agents_count, dead_agent_types: dead_agent_counts, shelter_data: shelter_info.clone() };
     app_state.death_json_counter.push(json!({"step": current_step, "dead_agents": dead_agent_counts}));
     app_state.shelter_json_counter.push(json!(shelter_info));
     app_state.state.tsunami_index = new_tsunami_index;
     app_state.state.dead_agents = dead_agents_count;
     if should_complete_simulation { app_state.state.is_completed = true; app_state.state.is_running = false; } else if let Some(max_steps) = app_state.config.max_steps { if current_step + 1 >= max_steps { app_state.state.is_completed = true; app_state.state.is_running = false; } }
     if !app_state.state.is_completed { app_state.state.current_step += 1; }

    let state_clone = app_state.state.clone();
    drop(app_state); // Release lock

    HttpResponse::Ok().json(json!({"status": "ok", "result": step_result, "simulation_state": state_clone}))
}

#[post("/run/{steps}")]
async fn run_steps(data: web::Data<Arc<Mutex<AppState>>>, steps_to_run: web::Path<u32>) -> impl Responder {
    let steps_to_run = steps_to_run.into_inner();
    if steps_to_run == 0 { return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Number of steps must be greater than 0"})); }

    let mut results = Vec::with_capacity(steps_to_run as usize);
    let mut steps_executed_count = 0;

    for _ in 0..steps_to_run {
        let mut app_state = data.lock().unwrap();
        if app_state.model.is_none() || app_state.state.is_completed { drop(app_state); break; }

        let current_step = app_state.state.current_step;
        let is_tsunami = app_state.state.is_tsunami;
        let tsunami_index = app_state.state.tsunami_index;
        let model = app_state.model.as_mut().unwrap();
        model.step(current_step, is_tsunami, tsunami_index);

        // Collect results
        let mut dead_agent_counts = DeadAgentTypeData::default(); for agent_type in &model.dead_agent_types { match agent_type { AgentType::Child => dead_agent_counts.child += 1, AgentType::Teen => dead_agent_counts.teen += 1, AgentType::Adult => dead_agent_counts.adult += 1, AgentType::Elder => dead_agent_counts.elder += 1, } dead_agent_counts.total += 1; }
        let dead_agents_count = model.dead_agents;
        let shelter_info: HashMap<String, ShelterAgentTypeData> = model.grid.shelters.iter().map(|&(_, _, id)| { let key = format!("shelter_{}_{}", id, current_step); let count = model.grid.shelter_agents.get(&id).map(|agents_in_shelter| { let mut data = ShelterAgentTypeData::default(); for agent_tuple in agents_in_shelter { match agent_tuple.1 { AgentType::Child => data.child += 1, AgentType::Teen => data.teen += 1, AgentType::Adult => data.adult += 1, AgentType::Elder => data.elder += 1, } } data }).unwrap_or_default(); (key, count) }).collect();

        // Check state changes
        let mut new_tsunami_index = tsunami_index;
        let mut should_complete_simulation = false;
        let should_update_tsunami = current_step >= TSUNAMI_DELAY;
        let tsunami_data_len = model.grid.tsunami_data.len();
        let should_advance_tsunami_index = should_update_tsunami && current_step > 0 && current_step % TSUNAMI_SPEED_TIME == 0;
        drop(model); // Release model borrow

        // Update state
        if should_update_tsunami { app_state.state.is_tsunami = true; if should_advance_tsunami_index { if tsunami_index + 1 >= tsunami_data_len { should_complete_simulation = true; } else { new_tsunami_index = tsunami_index + 1; } } }
        let step_result = StepResult { step: current_step, dead_agents: dead_agents_count, dead_agent_types: dead_agent_counts, shelter_data: shelter_info.clone() };
        app_state.death_json_counter.push(json!({"step": current_step, "dead_agents": dead_agent_counts}));
        app_state.shelter_json_counter.push(json!(shelter_info));
        app_state.state.tsunami_index = new_tsunami_index;
        app_state.state.dead_agents = dead_agents_count;
        if should_complete_simulation { app_state.state.is_completed = true; app_state.state.is_running = false; } else if let Some(max_steps) = app_state.config.max_steps { if current_step + 1 >= max_steps { app_state.state.is_completed = true; app_state.state.is_running = false; } }
        if !app_state.state.is_completed { app_state.state.current_step += 1; }

        results.push(json!({"step_executed": true, "step_result": step_result, "simulation_state": app_state.state.clone()}));
        steps_executed_count += 1;

        if app_state.state.is_completed { drop(app_state); break; }
        drop(app_state); // Release lock for this iteration
    }

    let final_state = data.lock().unwrap().state.clone();
    HttpResponse::Ok().json(json!({"status": "ok", "steps_requested": steps_to_run, "steps_executed": steps_executed_count, "simulation_state": final_state, "results": results}))
}

#[get("/status")]
async fn get_status(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let app_state = data.lock().unwrap();
    let is_initialized = app_state.model.is_some();
    HttpResponse::Ok().json(json!({"status": "ok", "is_initialized": is_initialized, "simulation_state": app_state.state, "agents_count": app_state.model.as_ref().map(|m| m.agents.len()), "dead_agents": app_state.model.as_ref().map(|m| m.dead_agents)}))
}

#[get("/export")]
async fn export_results(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let app_state = data.lock().unwrap();
    if app_state.death_json_counter.is_empty() && app_state.shelter_json_counter.is_empty() { return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Simulation not initialized or no steps run to export data"})); }
    let death_json_clone = app_state.death_json_counter.clone();
    let shelter_json_clone = app_state.shelter_json_counter.clone();
    drop(app_state);
    HttpResponse::Ok().json(json!({"status": "ok", "message": "Simulation results exported", "death_data": death_json_clone, "shelter_data": shelter_json_clone}))
}

#[post("/reset")]
async fn reset_simulation(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let mut app_state = data.lock().unwrap();
    app_state.model = None;
    app_state.state = SimulationState::default();
    app_state.death_json_counter = Vec::new();
    app_state.shelter_json_counter = Vec::new();
    println!("Simulation reset.");
    HttpResponse::Ok().json(json!({"status": "ok", "message": "Simulation reset"}))
}

#[get("/export/geojson")]
async fn export_agent_geojson(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    use serde_json::{json, Value};
    let app_state = data.lock().unwrap();
    if app_state.model.is_none() { return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Simulation not initialized (agent data unavailable)"})); }

    let model = app_state.model.as_ref().unwrap();
    let grid = &model.grid;
    let current_step = app_state.state.current_step;
    let location = &app_state.config.location;
    let epsg_code = match location.as_str() { "pacitan" | "sample" => "EPSG:32749", _ => "EPSG:32750" };

    let mut grouped_data: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    for agent in &model.agents {
        if agent.is_alive {
            let agent_type_str = agent.agent_type.to_string();
            let coordinates_list = grouped_data.entry(agent_type_str).or_insert_with(Vec::new);
            let x_utm = grid.xllcorner + (agent.x as f64 * grid.cellsize);
            let y_utm = grid.yllcorner + ((grid.height - 1 - agent.y) as f64 * grid.cellsize);
            coordinates_list.push(vec![x_utm, y_utm]);
        }
    }
    let features: Vec<Value> = grouped_data.into_iter().map(|(agent_type, coordinates)| { json!({"type": "Feature", "geometry": {"type": "MultiPoint", "coordinates": coordinates}, "properties": {"timestamp": current_step, "agent_type": agent_type}}) }).collect();
    let geojson = json!({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": epsg_code}}, "features": features});
    HttpResponse::Ok().content_type("application/geo+json").json(geojson)
}

#[get("/grid/geojson")]
async fn export_grid_geojson(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    use serde_json::{json, Value};
    let app_state = data.lock().unwrap();
    if app_state.model.is_none() { return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Simulation not initialized (grid data unavailable)"})); }

    let model = app_state.model.as_ref().unwrap();
    let grid = &model.grid;
    let location = &app_state.config.location;
    let epsg_code = match location.as_str() { "pacitan" | "sample" => "EPSG:32749", _ => "EPSG:32750" };
    let mut features: Vec<Value> = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let terrain = &grid.terrain[y as usize][x as usize];
            let x_utm = grid.xllcorner + (x as f64 * grid.cellsize);
            let y_utm = grid.yllcorner + ((grid.height - 1 - y) as f64 * grid.cellsize);
            let feature = match terrain {
                Terrain::Road => Some(json!({"type": "Feature", "geometry": {"type": "Point", "coordinates": [x_utm, y_utm]}, "properties": {"type": "road"}})),
                Terrain::Shelter(id) => Some(json!({"type": "Feature", "geometry": {"type": "Point", "coordinates": [x_utm, y_utm]}, "properties": {"type": "shelter", "id": id}})),
                Terrain::Land => None, // Add arm for Land - don't create a feature
                Terrain::Blocked => None,
            };
            if let Some(f) = feature { features.push(f); }
        }
    }
    let geojson = json!({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": epsg_code}}, "features": features});
    HttpResponse::Ok().content_type("application/geo+json").json(geojson)
}

#[get("/tsunami/geojson")]
async fn export_tsunami_geojson(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    use serde_json::{json, Value};
    let app_state = data.lock().unwrap();
    if app_state.model.is_none() { return HttpResponse::Ok().json(json!({"type": "FeatureCollection", "features": []})); } // Return empty if not initialized

    let model = app_state.model.as_ref().unwrap();
    let grid = &model.grid;
    let state = &app_state.state;
    let location = &app_state.config.location;
    let epsg_code = match location.as_str() { "pacitan" | "sample" => "EPSG:32749", _ => "EPSG:32750" };
    let mut features: Vec<Value> = Vec::new();

    if state.is_tsunami && state.tsunami_index < grid.tsunami_data.len() {
        let current_tsunami_frame = &grid.tsunami_data[state.tsunami_index];
        for y in 0..grid.height {
            for x in 0..grid.width {
                if let Some(row) = current_tsunami_frame.get(y as usize) {
                    if let Some(&height) = row.get(x as usize) {
                        if height > 0 {
                            let x_utm = grid.xllcorner + (x as f64 * grid.cellsize);
                            let y_utm = grid.yllcorner + ((grid.height - 1 - y) as f64 * grid.cellsize);
                            features.push(json!({"type": "Feature", "geometry": {"type": "Point", "coordinates": [x_utm, y_utm]}, "properties": {"type": "tsunami", "height": height}}));
                        }
                    } else { eprintln!("Warning: Tsunami frame index out of bounds at x={}", x); }
                } else { eprintln!("Warning: Tsunami frame index out of bounds at y={}", y); }
            }
        }
    }
    let geojson = json!({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": epsg_code}}, "features": features});
    HttpResponse::Ok().content_type("application/geo+json").json(geojson)
}

// Export precalculated grid costs (distance fields)
#[get("/grid/costs")]
async fn export_grid_costs(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    use serde_json::json;
    let app_state = data.lock().unwrap();
    if app_state.model.is_none() { return HttpResponse::BadRequest().json(json!({"status": "error", "message": "Simulation not initialized (grid cost data unavailable)"})); }

    let model = app_state.model.as_ref().unwrap();
    let grid = &model.grid;
    // Include environment_layers in the payload
    let costs_payload = json!({
        "ncols": grid.width, "nrows": grid.height,
        "xllcorner": grid.xllcorner, "yllcorner": grid.yllcorner, "cellsize": grid.cellsize,
        "distance_to_road": grid.distance_to_road,
        "distance_to_shelter": grid.distance_to_shelter,
        "environment_layers": grid.environment_layers // Add this line
    });
    drop(app_state); // Release lock
    HttpResponse::Ok().content_type("application/json").json(costs_payload)
}


// --- Server Setup ---
pub async fn start_api_server(port: u16) -> std::io::Result<()> {
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    println!("Starting API server on port {}", port);

    let app_state = Arc::new(Mutex::new(AppState {
        config: SimulationConfig::default(),
        state: SimulationState::default(),
        model: None,
        death_json_counter: Vec::new(),
        shelter_json_counter: Vec::new(),
    }));

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header();

        App::new()
            .wrap(Logger::default())
            .wrap(cors)
            .app_data(web::Data::new(app_state.clone()))
            .service(health_check)
            .service(get_config)
            .service(update_config)
            .service(init_simulation)
            .service(run_step)
            .service(run_steps)
            .service(get_status)
            .service(export_results)
            .service(export_agent_geojson)
            .service(export_grid_geojson)
            .service(export_tsunami_geojson)
            .service(export_grid_costs) // Add the new cost service
            .service(reset_simulation)
    })
    .bind(format!("0.0.0.0:{}", port))?
    .run()
    .await
}
