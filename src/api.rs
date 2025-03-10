use crate::game::agent::{Agent, AgentType};
use crate::game::game::Model;
use crate::game::grid::{load_grid_from_ascii, Grid};
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

// Configuration for simulation
#[derive(Serialize, Deserialize, Clone)]
pub struct SimulationConfig {
    pub location: String,
    pub grid_path: String,
    pub population_path: String,
    pub tsunami_data_path: String,
    pub output_path: String,
    pub max_steps: Option<u32>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            location: "pacitan".to_string(),
            grid_path: "./data_pacitan/jalantes_pacitan.asc".to_string(),
            population_path: "./data_pacitan/agent_pacitan.asc".to_string(),
            tsunami_data_path: "./data_pacitan/tsunami_ascii_pacitan".to_string(),
            output_path: "./output".to_string(),
            max_steps: None,
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

// Health check endpoint
#[get("/health")]
async fn health_check() -> impl Responder {
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "message": "Service is running"
    }))
}

// Get current configuration
#[get("/config")]
async fn get_config(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let app_state = data.lock().unwrap();
    HttpResponse::Ok().json(&app_state.config)
}

// Update configuration
#[post("/config")]
async fn update_config(
    data: web::Data<Arc<Mutex<AppState>>>,
    config: web::Json<SimulationConfig>,
) -> impl Responder {
    let mut app_state = data.lock().unwrap();
    app_state.config = config.into_inner();
    
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "message": "Configuration updated"
    }))
}

// Initialize simulation
#[post("/init")]
async fn init_simulation(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let mut app_state = data.lock().unwrap();
    
    // If simulation is already running, return error
    if app_state.state.is_running {
        return HttpResponse::BadRequest().json(json!({
            "status": "error",
            "message": "Simulation is already running"
        }));
    }
    
    // Reset simulation state
    app_state.state = SimulationState::default();
    app_state.death_json_counter = Vec::new();
    app_state.shelter_json_counter = Vec::new();
    
    // Get location-specific paths
    let (grid_path, population_path, tsunami_data_path) = app_state.config.get_location_paths();
    
    println!("Initializing simulation for location: {}", app_state.config.location);
    println!("Using grid path: {}", grid_path);
    println!("Using population path: {}", population_path);
    println!("Using tsunami data path: {}", tsunami_data_path);
    
    // Load grid and agents from file ASC
    let grid_result = load_grid_from_ascii(&grid_path);
    if grid_result.is_err() {
        return HttpResponse::InternalServerError().json(json!({
            "status": "error",
            "message": format!("Failed to load grid: {}", grid_result.err().unwrap())
        }));
    }
    
    let (mut grid, mut agents) = grid_result.unwrap();
    
    let mut next_agent_id = agents.len();
    
    // Load population and create agents
    let population_result = load_population_and_create_agents(
        &population_path,
        grid.width,
        grid.height,
        &mut grid,
        &mut agents,
        &mut next_agent_id,
    );
    
    if population_result.is_err() {
        return HttpResponse::InternalServerError().json(json!({
            "status": "error",
            "message": format!("Failed to populate grid: {}", population_result.err().unwrap())
        }));
    }
    
    // Export agent statistics
    let _ = export_agent_statistics(&agents);
    
    // Load tsunami data
    let tsunami_data_result = read_tsunami_data(&tsunami_data_path, grid.ncol, grid.nrow);
    if tsunami_data_result.is_err() {
        return HttpResponse::InternalServerError().json(json!({
            "status": "error",
            "message": format!("Failed to read tsunami data: {}", tsunami_data_result.err().unwrap())
        }));
    }
    
    let tsunami_data = tsunami_data_result.unwrap();
    let tsunami_len = tsunami_data.len();
    grid.tsunami_data = tsunami_data;
    
    // Create model
    let model = Model {
        grid,
        agents,
        dead_agents: 0,
        dead_agent_types: Vec::new(),
    };
    
    app_state.model = Some(model);
    app_state.state.is_running = true;
    
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "message": "Simulation initialized",
        "details": {
            "tsunami_data_length": tsunami_len,
            "agents_count": app_state.model.as_ref().unwrap().agents.len()
        }
    }))
}

// Run simulation step
#[post("/step")]
async fn run_step(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    // First, get a lock on app_state and check conditions
    let mut app_state = data.lock().unwrap();
    
    // Check if simulation is initialized
    if app_state.model.is_none() {
        return HttpResponse::BadRequest().json(json!({
            "status": "error",
            "message": "Simulation not initialized"
        }));
    }
    
    // Check if simulation is completed
    if app_state.state.is_completed {
        return HttpResponse::BadRequest().json(json!({
            "status": "error",
            "message": "Simulation already completed"
        }));
    }
    
    // Get the values we need without borrowing app_state
    let current_step = app_state.state.current_step;
    let is_tsunami = app_state.state.is_tsunami;
    let tsunami_index = app_state.state.tsunami_index;
    
    // Execute the step on the model and capture the results
    let model = app_state.model.as_mut().unwrap();
    model.step(current_step, is_tsunami, tsunami_index);
    
    // Collect stats for dead agents
    let mut dead_agent_counts = DeadAgentTypeData::default();
    for agent in &model.dead_agent_types {
        match agent {
            AgentType::Child => dead_agent_counts.child += 1,
            AgentType::Teen => dead_agent_counts.teen += 1,
            AgentType::Adult => dead_agent_counts.adult += 1,
            AgentType::Elder => dead_agent_counts.elder += 1,
        }
        dead_agent_counts.total += 1;
    }
    
    // Clone the data we need from the model
    let dead_agents = model.dead_agents;
    
    // Get shelter information
    let shelter_info: HashMap<String, ShelterAgentTypeData> = model
        .grid
        .shelters
        .iter()
        .map(|&(_, _, id)| {
            let key = format!("shelter_{}_{}", id, current_step as u32);
            let count = model
                .grid
                .shelter_agents
                .get(&id)
                .map(|agents| {
                    let mut shelter_agent_type_data = ShelterAgentTypeData::default();
                    for agent in agents {
                        match agent.1 {
                            AgentType::Child => shelter_agent_type_data.child += 1,
                            AgentType::Teen => shelter_agent_type_data.teen += 1,
                            AgentType::Adult => shelter_agent_type_data.adult += 1,
                            AgentType::Elder => shelter_agent_type_data.elder += 1,
                        }
                    }
                    shelter_agent_type_data
                })
                .unwrap_or(ShelterAgentTypeData::default());
            (key, count)
        })
        .collect();
    
    // Check tsunami conditions but extract data we need first
    let mut new_tsunami_index = tsunami_index;
    let mut is_completed = false;
    let should_update_tsunami = current_step > TSUNAMI_DELAY;
    
    // Extract tsunami len before dropping model borrow
    let tsunami_len = if should_update_tsunami && current_step % TSUNAMI_SPEED_TIME == 0 && current_step != 0 {
        Some(model.grid.tsunami_data.len())
    } else {
        None
    };
    
    // We're done with model, drop it to free the borrow
    drop(model);
    
    // Now we can modify app_state.state
    if should_update_tsunami {
        // Set tsunami flag
        app_state.state.is_tsunami = true;
        
        if let Some(len) = tsunami_len {
            if tsunami_index >= len - 1 {
                is_completed = true;
            } else {
                new_tsunami_index = tsunami_index + 1;
            }
        }
    }
    
    // Now create the step result before updating app_state
    let step_result = StepResult {
        step: current_step,
        dead_agents,
        dead_agent_types: dead_agent_counts,
        shelter_data: shelter_info.clone(),
    };
    
    // Update app_state
    app_state.death_json_counter.push(json!({
        "step": current_step,
        "dead_agents": {
            "child": dead_agent_counts.child,
            "teen": dead_agent_counts.teen,
            "adult": dead_agent_counts.adult,
            "elder": dead_agent_counts.elder,
            "total": dead_agent_counts.total
        }
    }));
    
    app_state.shelter_json_counter.push(json!(shelter_info));
    
    // Update tsunami index if needed
    app_state.state.tsunami_index = new_tsunami_index;
    
    // Update completion status if needed
    if is_completed {
        app_state.state.is_completed = true;
    }
    
    // Increment step
    app_state.state.current_step += 1;
    
    // Check if max steps reached
    if let Some(max_steps) = app_state.config.max_steps {
        if app_state.state.current_step >= max_steps {
            app_state.state.is_completed = true;
        }
    }
    
    // Clone the state to use after releasing the lock
    let state_clone = app_state.state.clone();
    
    // Release the lock by dropping app_state
    drop(app_state);
    
    // Return the result
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "result": step_result,
        "simulation_state": state_clone
    }))
}

// Run multiple simulation steps
#[post("/run/{steps}")]
async fn run_steps(
    data: web::Data<Arc<Mutex<AppState>>>,
    steps: web::Path<u32>,
) -> impl Responder {
    let steps = steps.into_inner();
    let mut results = Vec::with_capacity(steps as usize);
    
    for _ in 0..steps {
        // First check if we should continue
        let should_continue = {
            let app_state = data.lock().unwrap();
            app_state.model.is_some() && !app_state.state.is_completed
        };
        
        if !should_continue {
            break;
        }
        
        // Now perform the step
        let mut app_state = data.lock().unwrap();
        
        // Get the values we need without borrowing app_state
        let current_step = app_state.state.current_step;
        let is_tsunami = app_state.state.is_tsunami;
        let tsunami_index = app_state.state.tsunami_index;
        
        // Execute the step on the model and capture the results
        let model = app_state.model.as_mut().unwrap();
        model.step(current_step, is_tsunami, tsunami_index);
        
        // Collect stats for dead agents
        let mut dead_agent_counts = DeadAgentTypeData::default();
        for agent in &model.dead_agent_types {
            match agent {
                AgentType::Child => dead_agent_counts.child += 1,
                AgentType::Teen => dead_agent_counts.teen += 1,
                AgentType::Adult => dead_agent_counts.adult += 1,
                AgentType::Elder => dead_agent_counts.elder += 1,
            }
            dead_agent_counts.total += 1;
        }
        
        // Clone the data we need from the model
        let dead_agents = model.dead_agents;
        
        // Get shelter information
        let shelter_info: HashMap<String, ShelterAgentTypeData> = model
            .grid
            .shelters
            .iter()
            .map(|&(_, _, id)| {
                let key = format!("shelter_{}_{}", id, current_step as u32);
                let count = model
                    .grid
                    .shelter_agents
                    .get(&id)
                    .map(|agents| {
                        let mut shelter_agent_type_data = ShelterAgentTypeData::default();
                        for agent in agents {
                            match agent.1 {
                                AgentType::Child => shelter_agent_type_data.child += 1,
                                AgentType::Teen => shelter_agent_type_data.teen += 1,
                                AgentType::Adult => shelter_agent_type_data.adult += 1,
                                AgentType::Elder => shelter_agent_type_data.elder += 1,
                            }
                        }
                        shelter_agent_type_data
                    })
                    .unwrap_or(ShelterAgentTypeData::default());
                (key, count)
            })
            .collect();
        
        // Check tsunami conditions but extract data we need first
        let mut new_tsunami_index = tsunami_index;
        let mut is_completed = false;
        let should_update_tsunami = current_step > TSUNAMI_DELAY;
        
        // Extract tsunami len before dropping model borrow
        let tsunami_len = if should_update_tsunami && current_step % TSUNAMI_SPEED_TIME == 0 && current_step != 0 {
            Some(model.grid.tsunami_data.len())
        } else {
            None
        };
        
        // We're done with model, drop it to free the borrow
        drop(model);
        
        // Now we can modify app_state.state
        if should_update_tsunami {
            // Set tsunami flag
            app_state.state.is_tsunami = true;
            
            if let Some(len) = tsunami_len {
                if tsunami_index >= len - 1 {
                    is_completed = true;
                } else {
                    new_tsunami_index = tsunami_index + 1;
                }
            }
        }
        
        // Now create the step result before updating app_state
        let step_result = StepResult {
            step: current_step,
            dead_agents,
            dead_agent_types: dead_agent_counts,
            shelter_data: shelter_info.clone(),
        };
        
        // Update app_state
        app_state.death_json_counter.push(json!({
            "step": current_step,
            "dead_agents": {
                "child": dead_agent_counts.child,
                "teen": dead_agent_counts.teen,
                "adult": dead_agent_counts.adult,
                "elder": dead_agent_counts.elder,
                "total": dead_agent_counts.total
            }
        }));
        
        app_state.shelter_json_counter.push(json!(shelter_info));
        
        // Update tsunami index if needed
        app_state.state.tsunami_index = new_tsunami_index;
        
        // Update dead_agents count in the simulation state
        app_state.state.dead_agents = dead_agents;
        
        // Update completion status if needed
        if is_completed {
            app_state.state.is_completed = true;
        }
        
        // Increment step
        app_state.state.current_step += 1;
        
        // Check if max steps reached
        if let Some(max_steps) = app_state.config.max_steps {
            if app_state.state.current_step >= max_steps {
                app_state.state.is_completed = true;
            }
        }
        
        // Release the lock
        drop(app_state);
        
        results.push(json!({
            "step_executed": true,
            "step_result": step_result
        }));
    }
    
    // Get the final state outside of the loop
    let final_state = data.lock().unwrap().state.clone();
    
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "steps_executed": results.len(),
        "simulation_state": final_state,
        "results": results
    }))
}

// Get simulation status
#[get("/status")]
async fn get_status(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let app_state = data.lock().unwrap();
    
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "simulation_state": app_state.state,
        "agents_count": app_state.model.as_ref().map(|m| m.agents.len()).unwrap_or(0),
        "dead_agents": app_state.model.as_ref().map(|m| m.dead_agents).unwrap_or(0)
    }))
}

// Export simulation results
#[get("/export")]
async fn export_results(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    // Create a separate scope for mutable borrowing
    let result = {
        let app_state = data.lock().unwrap();
        
        // Check if simulation is initialized
        if app_state.model.is_none() {
            return HttpResponse::BadRequest().json(json!({
                "status": "error",
                "message": "Simulation not initialized"
            }));
        }
        
        // Create clones of the JSON data to avoid borrow issues
        let death_json_clone = app_state.death_json_counter.clone();
        let shelter_json_clone = app_state.shelter_json_counter.clone();
        
        (death_json_clone, shelter_json_clone)
    };
    
    // Use the cloned data outside the scope to avoid mutable borrow issues
    let (death_json, shelter_json) = result;
    
    // Return results without trying to save (which requires mutable access)
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "message": "Simulation results exported",
        "death_data": death_json,
        "shelter_data": shelter_json
    }))
}

// Reset simulation
#[post("/reset")]
async fn reset_simulation(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    let mut app_state = data.lock().unwrap();
    
    app_state.model = None;
    app_state.state = SimulationState::default();
    app_state.death_json_counter = Vec::new();
    app_state.shelter_json_counter = Vec::new();
    
    HttpResponse::Ok().json(json!({
        "status": "ok",
        "message": "Simulation reset"
    }))
}

// Export agent locations as GeoJSON
#[get("/export/geojson")]
async fn export_agent_geojson(data: web::Data<Arc<Mutex<AppState>>>) -> impl Responder {
    use serde_json::{json, Value};
    use std::collections::HashMap;
    
    let app_state = data.lock().unwrap();
    
    // Check if simulation is initialized
    if app_state.model.is_none() {
        return HttpResponse::BadRequest().json(json!({
            "status": "error",
            "message": "Simulation not initialized"
        }));
    }
    
    let model = app_state.model.as_ref().unwrap();
    let current_step = app_state.state.current_step;
    let location = &app_state.config.location;
    
    // Get the correct EPSG code based on location
    let epsg_code = if location == "pacitan" {
        "EPSG:32749" // UTM zone 49S for Pacitan
    } else {
        "EPSG:32750" // UTM zone 50S for Jembrana
    };
    
    // Group agents by type
    let mut grouped_data: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    
    for agent in &model.agents {
        let agent_type = agent.agent_type.to_string();
        let coordinates = grouped_data.entry(agent_type).or_insert_with(Vec::new);
        
        // Convert grid coordinates to UTM coordinates
        let x_utm = agent.x as f64 * model.grid.cellsize + model.grid.xllcorner;
        let y_utm = agent.y as f64 * (-1.0 * model.grid.cellsize) + model.grid.yllcorner + (model.grid.nrow as f64 * model.grid.cellsize) ;
        
        coordinates.push(vec![x_utm, y_utm]);
    }
    
    // Create GeoJSON features for each agent type
    let features: Vec<Value> = grouped_data
        .into_iter()
        .map(|(agent_type, coordinates)| {
            json!({
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": coordinates
                },
                "properties": {
                    "timestamp": current_step,
                    "agent_type": agent_type
                }
            })
        })
        .collect();
    
    // Create the GeoJSON FeatureCollection
    let geojson = json!({
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": epsg_code
            }
        },
        "features": features
    });
    
    HttpResponse::Ok()
        .content_type("application/geo+json")
        .json(geojson)
}

// Start API server
pub async fn start_api_server(port: u16) -> std::io::Result<()> {
    // Initialize logger
    env_logger::init_from_env(env_logger::Env::new().default_filter_or("info"));
    
    println!("Starting API server on port {}", port);
    
    // Create application state
    let app_state = Arc::new(Mutex::new(AppState {
        config: SimulationConfig::default(),
        state: SimulationState::default(),
        model: None,
        death_json_counter: Vec::new(),
        shelter_json_counter: Vec::new(),
    }));
    
    // Start HTTP server
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
            .service(reset_simulation)
    })
    .bind(format!("0.0.0.0:{}", port))?
    .run()
    .await
}
