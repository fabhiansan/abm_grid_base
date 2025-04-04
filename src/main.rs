mod game;
mod api;

use game::agent::{/* Removed Agent */ AgentType};
use game::game::Model;
use game::grid::{load_grid_from_ascii, load_float_asc_layer, Grid, Terrain};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead /* Removed Write */};
use std::path::{Path, PathBuf};
use std::env;
use rayon::prelude::*;
use std::fs; // Removed path as it's implicitly imported with fs

// Removed: const TSUNAMI_DELAY: u32 = 5;
const TSUNAMI_SPEED_TIME: u32 = 60; // Keep this for now, controls tsunami data index advancement speed

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ShelterAgentTypeData {
    pub child: u32,
    pub teen: u32,
    pub adult: u32,
    pub elder: u32,
    pub car: u32,
}

impl Default for ShelterAgentTypeData {
    fn default() -> Self {
        ShelterAgentTypeData {
            child: 0,
            teen: 0,
            adult: 0,
            elder: 0,
            car: 0,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ShelterData {
    pub step: u32,
    pub shelters: HashMap<String, ShelterAgentTypeData>,
    pub total_dead_agents: usize,
}

#[derive(Serialize, Deserialize, Default)]
pub struct SimulationData {
    pub records: Vec<ShelterData>,
}

#[derive(Serialize, Deserialize)]
struct AgentStatistics {
    total_agents: usize,
    agent_types: HashMap<String, usize>,
}

// --- Simulation Result Struct ---
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct SimulationResult {
    pub total_steps: u32,
    pub total_dead_agents: usize,
    pub final_shelter_counts: HashMap<String, ShelterAgentTypeData>, // Simplified for now
    pub message: String,
    // We might add more detailed results later, like the full timeseries data
}
// --- End Simulation Result Struct ---


pub fn export_agent_statistics(agents: &Vec<crate::game::agent::Agent>) -> std::io::Result<()> {
    let mut stats = AgentStatistics {
        total_agents: agents.len(),
        agent_types: HashMap::new(),
    };

    // Count agents by type
    for agent in agents {
        let agent_type = match agent.agent_type {
            crate::game::agent::AgentType::Child => "Child",
            crate::game::agent::AgentType::Teen => "Teen",
            crate::game::agent::AgentType::Adult => "Adult",
            crate::game::agent::AgentType::Elder => "Elder",
            // crate::game::agent::AgentType::Car => "Car",
        };
        *stats.agent_types.entry(agent_type.to_string()).or_insert(0) += 1;
    }

    // Write to JSON file
    let json = serde_json::to_string_pretty(&stats)?;
    // Consider making this filename configurable or part of the result
    std::fs::write("simulation_initial_stats.json", json)?;

    Ok(())
}

pub const DISTRIBUTION_WEIGHTS: [i32; 5] = [10, 20, 30, 15, 20];

/* // Function to write grid state - kept for potential debugging, but not essential for core logic
fn write_grid_to_ascii(filename: &str, model: &Model) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(filename)?;

    // Tulis header ASC dengan nilai dari grid
    writeln!(file, "ncols        {}", model.grid.width)?;
    writeln!(file, "nrows        {}", model.grid.height)?;
    writeln!(file, "xllcorner    {}", model.grid.xllcorner)?;
    writeln!(file, "yllcorner    {}", model.grid.yllcorner)?;
    writeln!(file, "cellsize     {}", model.grid.cellsize)?;
    writeln!(file, "NODATA_value  0")?; // Assuming 0 is NODATA for visualization

    // Tulis data grid: tiap baris dipisahkan spasi
    for y in 0..model.grid.height as usize {
        let mut row_tokens = Vec::with_capacity(model.grid.width as usize);
        for x in 0..model.grid.width as usize {
            let token = if !model.grid.agents_in_cell[y][x].is_empty() {
                let agent_id = model.grid.agents_in_cell[y][x][0];
                if let Some(agent) = model.agents.iter().find(|a| a.id == agent_id) {
                    match agent.agent_type {
                        AgentType::Child => "3",
                        AgentType::Teen => "4",
                        AgentType::Adult => "5",
                        AgentType::Elder => "6",
                    }
                    .to_string()
                } else {
                    "0".to_string() // Fallback if agent not found
                }
            } else {
                match model.grid.terrain[y][x] {
                    Terrain::Land => "0".to_string(),
                    Terrain::Blocked => "0".to_string(), // Represent Blocked as NODATA
                    Terrain::Road => "1".to_string(),
                    Terrain::Shelter(id) => format!("20{:02}", id), // Shelter visualization
                }
            };
            row_tokens.push(token);
        }
        let row_line = row_tokens.join(" ");
        writeln!(file, "{}", row_line)?;
    }
    Ok(())
}
*/

// Structure to store agent data for each step
#[derive(Clone, Serialize)] // Added Serialize for potential future use
struct AgentStepData {
    x: f64,
    y: f64,
    id: usize,
    agent_type: String,
    is_on_road: bool,
    speed: u32,
    step: u32,
}

// Structure to collect all agent data throughout simulation
pub struct AgentDataCollector {
    data: Vec<AgentStepData>,
    grid: Grid, // Keep a copy of grid metadata for coordinate conversion
}

impl AgentDataCollector {
    fn new(grid: Grid) -> Self {
        Self {
            data: Vec::new(),
            grid,
        }
    }

    fn collect_step(&mut self, model: &Model, step: u32) {
        for agent in &model.agents {
            if agent.is_alive {
                // Convert grid coordinates (agent.x, agent.y) to real-world coordinates
                let real_x = self.grid.xllcorner + (agent.x as f64 * self.grid.cellsize);
                // Note: Y-axis calculation for ASC grids often needs inversion
                let real_y = self.grid.yllcorner
                    + ((self.grid.nrow - 1 - agent.y) as f64 * self.grid.cellsize); // Adjusted Y calculation

                self.data.push(AgentStepData {
                    x: real_x,
                    y: real_y,
                    id: agent.id,
                    agent_type: format!("{:?}", agent.agent_type),
                    is_on_road: agent.is_on_road,
                    speed: agent.speed, // Assuming Agent struct has speed
                    step,
                });
            }
        }
    }
}

// Function to export agent movement data to GeoJSON
fn export_agents_to_geojson(collector: &AgentDataCollector, filename: &str) -> std::io::Result<()> {
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;

    // Group data by step and agent type for MultiPoint features per step
    let mut grouped_data: HashMap<(u32, String), Vec<Vec<f64>>> = HashMap::new();

    for agent_data in &collector.data {
        let key = (agent_data.step, agent_data.agent_type.clone());
        let coordinates_list = grouped_data.entry(key).or_insert_with(Vec::new);
        coordinates_list.push(vec![agent_data.x, agent_data.y]); // Use pre-calculated real coords
    }

    let features: Vec<Value> = grouped_data
        .into_iter()
        .map(|((step, agent_type), coordinates)| {
            json!({
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": coordinates // Array of [x, y] pairs
                },
                "properties": {
                    "timestamp": step,
                    "agent_type": agent_type
                }
            })
        })
        .collect();

    let geojson = json!({
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                // Assuming UTM Zone 49S for Pacitan/Jembrana - MAKE THIS CONFIGURABLE
                "name": "EPSG:32749"
            }
        },
        "features": features
    });

    let mut file = File::create(filename)?;
    file.write_all(serde_json::to_string_pretty(&geojson)?.as_bytes())?;

    Ok(())
}

// --- Refactored Simulation Instance Function ---
pub fn run_simulation_instance(config: &api::SimulationConfig) -> io::Result<SimulationResult> {
    println!("Starting simulation instance with config: {:?}", config);

    let (grid_path, population_path, tsunami_data_path) = config.get_location_paths();
    // Access dtm_path field directly, clone the Option<String>
    let dtm_path = config.dtm_path.clone().unwrap_or_default();

    println!("Using location: {}", config.location);
    println!("Grid path: {}", grid_path);
    println!("Population path: {}", population_path);
    println!("Tsunami data path: {}", tsunami_data_path);
    println!("DTM path: {}", dtm_path);
    println!("Tsunami Delay: {}", config.tsunami_delay);
    println!("Agent Reaction Delay: {}", config.agent_reaction_delay);


    // Load grid and initial agents
    let (mut grid, mut agents) =
        load_grid_from_ascii(&grid_path).expect("Failed to load grid");
    println!("Grid loaded: {}x{}", grid.width, grid.height);

    // Load population and add agents
    let mut next_agent_id = agents.len();
    load_population_and_create_agents(
        &population_path,
        grid.width,
        grid.height,
        &mut grid,
        &mut agents,
        &mut next_agent_id,
    ).expect("Failed to populate grid");
    println!("Population loaded, total agents: {}", agents.len());

    // Export initial agent statistics (optional)
    export_agent_statistics(&agents).expect("Failed to export initial agent statistics");

    // Load tsunami data
    println!("Loading tsunami data...");
    let tsunami_data = read_tsunami_data(&tsunami_data_path, grid.ncol, grid.nrow)?;
    let tsunami_len = tsunami_data.len();
    println!("Loaded {} tsunami inundation timesteps", tsunami_len);
    if tsunami_len == 0 {
         eprintln!("Warning: No tsunami data loaded. Tsunami impact will not be simulated.");
    }
    grid.tsunami_data = tsunami_data;


    // Load DTM data if path is provided and exists
    if !dtm_path.is_empty() {
        println!("Attempting to load DTM from: {}", dtm_path);
        match load_float_asc_layer(&dtm_path) {
            Ok((dtm_data, dtm_ncols, dtm_nrows, _, _, _)) => {
                if dtm_ncols == grid.ncol && dtm_nrows == grid.nrow {
                    grid.environment_layers.insert("dtm".to_string(), dtm_data);
                    println!("Successfully loaded DTM data.");

                    // Recalculate distance fields using DTM if configured (assuming true for now)
                    // TODO: Add a config flag `use_dtm_for_pathfinding`
                    let use_dtm_for_pathfinding = true;
                    if use_dtm_for_pathfinding {
                        println!("Recalculating distance fields using DTM...");
                        grid.compute_distance_to_shelters(true); // Pass true to use DTM
                        grid.compute_distance_to_road(true);     // Pass true to use DTM
                        println!("Distance fields recalculated.");
                    } else {
                        println!("DTM loaded, but pathfinding will use simple distance.");
                    }
                } else {
                    eprintln!("Error: DTM dimensions ({}x{}) do not match grid dimensions ({}x{}). DTM not used.",
                             dtm_ncols, dtm_nrows, grid.ncol, grid.nrow);
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to load DTM data from {}: {}. Proceeding without DTM.", dtm_path, e);
            }
        }
    } else {
        println!("No DTM path provided or DTM loading disabled. Proceeding without DTM.");
        // Ensure distance fields are computed without DTM if they weren't already
        if grid.distance_to_shelter.is_empty() { // Basic check if computation is needed
             println!("Computing distance fields without DTM...");
             grid.compute_distance_to_shelters(false);
             grid.compute_distance_to_road(false);
             println!("Distance fields computed.");
        }
    }


    // Initialize the model
    let mut model = Model {
        grid,
        agents,
        dead_agents: 0,
        dead_agent_types: Vec::new(),
    };

    // Initialize data collectors (optional, for detailed output)
    let mut collector = AgentDataCollector::new(model.grid.clone());
    let mut death_json_counter: Vec<serde_json::Value> = Vec::new();
    let mut shelter_json_counter: Vec<serde_json::Value> = Vec::new();


    // --- Simulation Loop ---
    let mut current_step = 0;
    let mut tsunami_data_index = 0; // Renamed from 'index' for clarity
    let mut is_playing = true;

    // Define a maximum step count to prevent infinite loops
    let max_steps = config.max_steps.unwrap_or(5000); // Default to 5000 if not set

    println!("Starting simulation loop (max_steps: {})...", max_steps);

    while is_playing && current_step < max_steps {
        // Determine if tsunami is active based on tsunami_delay
        let is_tsunami_active = current_step >= config.tsunami_delay;

        // Advance tsunami data index periodically after tsunami becomes active
        if is_tsunami_active && tsunami_len > 0 {
             // Advance index based on TSUNAMI_SPEED_TIME, but only if tsunami is active
             if current_step > config.tsunami_delay && (current_step - config.tsunami_delay) % TSUNAMI_SPEED_TIME == 0 {
                 if tsunami_data_index < tsunami_len - 1 {
                     tsunami_data_index += 1;
                     // println!("Advancing to tsunami index: {}", tsunami_data_index); // Can be verbose
                 } else {
                     // Optional: Stop simulation if last tsunami step is reached and no agents are moving?
                     // Or just keep using the last tsunami frame. For now, keep using last frame.
                     // println!("Reached final tsunami index {}. Using last data frame.", tsunami_data_index);
                 }
             }
        } else {
             tsunami_data_index = 0; // Ensure index is 0 before tsunami hits
        }


        // Perform simulation step
        // Pass the relevant delays and current tsunami index to the model step
        model.step(current_step, config.agent_reaction_delay, is_tsunami_active, tsunami_data_index);

        // --- Data Collection (every N steps, e.g., 30) ---
        if current_step % 30 == 0 {
            println!("Step: {}, Dead Agents: {}", current_step, model.dead_agents); // Progress indicator

            // Collect agent positions for GeoJSON
            collector.collect_step(&model, current_step);

            // Collect death counts
            let mut dead_agent_counts = DeadAgentTypeData::default();
            for agent_type in &model.dead_agent_types { // Iterate over types recorded at death
                match agent_type {
                    AgentType::Child => dead_agent_counts.child += 1,
                    AgentType::Teen => dead_agent_counts.teen += 1,
                    AgentType::Adult => dead_agent_counts.adult += 1,
                    AgentType::Elder => dead_agent_counts.elder += 1,
                }
            }
            dead_agent_counts.total = model.dead_agents as u32; // Use the model's counter

            death_json_counter.push(json!({
                "step": current_step,
                "dead_agents": dead_agent_counts // Use the struct directly
            }));

            // Collect shelter counts
            let shelter_info: HashMap<String, ShelterAgentTypeData> = model.grid.shelters.iter()
                .map(|&(_, _, id)| {
                    let key = format!("shelter_{}", id); // Key is just shelter ID
                    let count = model.grid.shelter_agents.get(&id)
                        .map(|agents_in_shelter| {
                            let mut type_data = ShelterAgentTypeData::default();
                            for agent_tuple in agents_in_shelter { // Iterate over tuples (agent_id, agent_type)
                                match agent_tuple.1 { // Access the AgentType from the tuple
                                    AgentType::Child => type_data.child += 1,
                                    AgentType::Teen => type_data.teen += 1,
                                    AgentType::Adult => type_data.adult += 1,
                                    AgentType::Elder => type_data.elder += 1,
                                }
                            }
                            type_data
                        })
                        .unwrap_or_default();
                    (key, count)
                })
                .collect();

             shelter_json_counter.push(json!({
                 "step": current_step,
                 "shelters": shelter_info
             }));

            // Optional: Write grid state periodically for debugging
            // let filename = format!("output/step_{}.asc", current_step);
            // if let Err(e) = write_grid_to_ascii(&filename, &model) {
            //     eprintln!("Error writing {}: {}", filename, e);
            // }
        }
        // --- End Data Collection ---


        // --- Check Simulation End Conditions ---
        // Example: Stop if all agents are dead or in shelters
        let active_agents = model.agents.iter().filter(|a| a.is_alive && !a.is_in_shelter).count();
        if active_agents == 0 && current_step > config.agent_reaction_delay { // Ensure agents had a chance to react
            println!("Simulation ended: No more active agents outside shelters at step {}.", current_step);
            is_playing = false;
        }
        // --- End Check Simulation End Conditions ---

        current_step += 1;
    } // End while loop

    if current_step == max_steps {
         println!("Simulation ended: Reached maximum step count ({}).", max_steps);
    }

    println!("Simulation loop finished. Total steps: {}", current_step);
    println!("Final dead agents: {}", model.dead_agents);


    // --- Final Data Export ---
    // Save final shelter counts (using the last collected data)
    let final_shelter_counts = if let Some(last_shelter_data) = shelter_json_counter.last() {
         // Attempt to deserialize the shelter part of the last JSON entry
         last_shelter_data.get("shelters")
             .and_then(|s| serde_json::from_value(s.clone()).ok())
             .unwrap_or_default()
    } else {
         // If no data collected, calculate final counts now
         model.grid.shelters.iter()
             .map(|&(_, _, id)| {
                 let key = format!("shelter_{}", id);
                 let count = model.grid.shelter_agents.get(&id)
                     .map(|agents_in_shelter| {
                         let mut type_data = ShelterAgentTypeData::default();
                         for agent_tuple in agents_in_shelter {
                             match agent_tuple.1 {
                                 AgentType::Child => type_data.child += 1,
                                 AgentType::Teen => type_data.teen += 1,
                                 AgentType::Adult => type_data.adult += 1,
                                 AgentType::Elder => type_data.elder += 1,
                             }
                         }
                         type_data
                     })
                     .unwrap_or_default();
                 (key, count)
             })
             .collect()
    };


    // Save detailed timeseries data (optional, could be large)
    // Consider making these filenames configurable
    fs::create_dir_all("output")?; // Ensure output directory exists
    if let Err(e) = fs::write("output/death_timeseries.json", serde_json::to_string_pretty(&death_json_counter)?) {
        eprintln!("Error saving death timeseries data: {}", e);
    }
     if let Err(e) = fs::write("output/shelter_timeseries.json", serde_json::to_string_pretty(&shelter_json_counter)?) {
        eprintln!("Error saving shelter timeseries data: {}", e);
    }

    // Export agent movement GeoJSON
    if let Err(e) = export_agents_to_geojson(&collector, "output/agent_movement.geojson") {
         eprintln!("Error exporting agent movement GeoJSON: {}", e);
    } else {
         println!("Agent movement data saved to output/agent_movement.geojson");
    }
    // --- End Final Data Export ---


    // Prepare and return the final result
    Ok(SimulationResult {
        total_steps: current_step,
        total_dead_agents: model.dead_agents,
        final_shelter_counts,
        message: format!("Simulation completed in {} steps.", current_step),
    })
}
// --- End Refactored Simulation Instance Function ---


#[derive(Serialize, Deserialize, Clone, Copy, Debug)] // Added Debug
pub struct DeadAgentTypeData {
    pub child: u32,
    pub teen: u32,
    pub adult: u32,
    pub elder: u32,
    pub car: u32, // Keep car for struct consistency, even if not used currently
    pub total: u32,
}

impl Default for DeadAgentTypeData {
    fn default() -> Self {
        DeadAgentTypeData {
            child: 0,
            teen: 0,
            adult: 0,
            elder: 0,
            car: 0,
            total: 0,
        }
    }
}

// Removed DeadAgentData struct as we collect timeseries differently now

pub fn load_population_and_create_agents(
    path: &str,
    ncols: u32,
    nrows: u32,
    grid: &mut Grid,
    agents: &mut Vec<crate::game::agent::Agent>,
    next_agent_id: &mut usize,
) -> std::io::Result<()> {
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header lines (assuming 6)
    for _ in 0..6 {
        lines.next();
    }

    let mut population: Vec<Vec<u32>> = Vec::with_capacity(nrows as usize);
    for line in lines {
        let line = line?;
        // Handle potential extra whitespace or non-numeric values gracefully
        let row: Vec<u32> = line.split_whitespace()
            .filter_map(|token| token.parse::<u32>().ok())
            .collect();
        // Basic check if row length matches expected ncols, adjust if necessary
        if row.len() >= ncols as usize {
             population.push(row.into_iter().take(ncols as usize).collect());
        } else if !row.is_empty() {
             // Handle potentially short rows, maybe pad with 0 or log warning
             eprintln!("Warning: Population file row shorter than expected ncols. Padding with 0.");
             let mut padded_row = row;
             padded_row.resize(ncols as usize, 0);
             population.push(padded_row);
        }
        // Skip empty lines
    }

    if population.len() != nrows as usize {
        eprintln!("Warning: Population file has {} rows, expected {}. Grid dimensions might be misaligned.", population.len(), nrows);
        // Decide how to handle: error out or proceed with caution? For now, proceed.
        // return Err(std::io::Error::new(
        //     std::io::ErrorKind::InvalidData,
        //     "Population data rows do not match grid height.",
        // ));
    }

    // Iterate population data and create agents
    for (y, row) in population.iter().enumerate() {
        // Ensure we don't go out of bounds if population rows < nrows
        if y >= grid.height as usize { break; }
        for (x, &pop) in row.iter().enumerate() {
             // Ensure we don't go out of bounds if population cols < ncols
             if x >= grid.width as usize { break; }

            // Create agents based on population density 'pop'
            // Current logic creates 1 agent if pop != 0. Adjust if 'pop' means number of agents.
            if pop > 0 {
                // Example: Create 'pop' number of agents per cell
                // for _ in 0..pop {
                // For now, stick to 1 agent per non-zero cell for simplicity
                for _ in 0..1 {
                    // Check terrain at (x, y) *before* creating agent
                    if y < grid.terrain.len() && x < grid.terrain[y].len() {
                        let is_on_road = grid.terrain[y][x] == Terrain::Road;
                        let agent_type = crate::game::agent::AgentType::random(); // Assign random type

                        let agent = crate::game::agent::Agent::new( // Removed mut
                            *next_agent_id,
                            x as u32,
                            y as u32,
                            agent_type,
                            is_on_road,
                        );
                        // agent.remaining_steps = agent.speed; // Agent::new likely handles this

                        grid.add_agent(x as u32, y as u32, agent.id);
                        agents.push(agent);
                        *next_agent_id += 1;
                    } else {
                         eprintln!("Warning: Skipping agent creation at ({}, {}), out of terrain bounds.", x, y);
                    }
                }
            }
        }
    }

    Ok(())
}


// Helper function to read a single tsunami data file
fn read_tsunami_data_file(path: &Path, ncols: u32, nrows: u32) -> io::Result<Vec<Vec<u32>>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    // Read header to find NODATA_value
    let mut nodata_value: Option<f64> = None;
    let mut header_count = 0;
    let mut header_lines = Vec::new(); // Store header lines if needed later

    for line_result in lines.by_ref() {
        header_count += 1;
        let line = line_result?;
        header_lines.push(line.clone()); // Store the line

        if line.to_lowercase().starts_with("nodata_value") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(val) = parts[1].parse::<f64>() {
                    nodata_value = Some(val);
                    println!("Found NODATA_value: {} in {:?}", val, path.file_name().unwrap_or_default());
                }
            }
        }
        if header_count >= 6 { // Assume header is max 6 lines
            break;
        }
    }

    // If NODATA_value wasn't found, try a default or raise an error
    let nodata = nodata_value.unwrap_or_else(|| {
        // Default based on path - refine this logic if needed
        if path.to_string_lossy().contains("pacitan") {
             println!("Warning: NODATA_value not found in {:?}, assuming 0.0 for Pacitan.", path.file_name().unwrap_or_default());
             0.0
        } else {
             println!("Warning: NODATA_value not found in {:?}, assuming -9999.0 as default.", path.file_name().unwrap_or_default());
             -9999.0
        }
    });


    // Read data rows
    let mut tsunami_data = Vec::with_capacity(nrows as usize);
    for line_result in lines {
        let line = line_result?;
        let row: Vec<u32> = line.split_whitespace()
            .filter_map(|token| token.parse::<f64>().ok()) // Parse as float first
            .map(|val| {
                // Check against NODATA, handle potential float comparison issues
                if (val - nodata).abs() < 1e-6 { // Use tolerance for float comparison
                    0 // No tsunami height
                } else {
                    val.max(0.0) as u32 // Tsunami height, ensure non-negative
                }
            })
            .collect();

        // Ensure row has correct length, pad if necessary
        if row.len() >= ncols as usize {
            tsunami_data.push(row.into_iter().take(ncols as usize).collect());
        } else if !row.is_empty() {
             eprintln!("Warning: Tsunami data row in {:?} shorter than expected ncols. Padding with 0.", path.file_name().unwrap_or_default());
             let mut padded_row = row;
             padded_row.resize(ncols as usize, 0);
             tsunami_data.push(padded_row);
        }
        // Skip empty lines implicitly
    }

    // Ensure correct number of rows, pad if necessary
    if tsunami_data.len() < nrows as usize {
         eprintln!("Warning: Tsunami data file {:?} has {} rows, expected {}. Padding with 0.", path.file_name().unwrap_or_default(), tsunami_data.len(), nrows);
         tsunami_data.resize_with(nrows as usize, || vec![0; ncols as usize]);
    } else if tsunami_data.len() > nrows as usize {
         eprintln!("Warning: Tsunami data file {:?} has {} rows, expected {}. Truncating.", path.file_name().unwrap_or_default(), tsunami_data.len(), nrows);
         tsunami_data.truncate(nrows as usize);
    }


    Ok(tsunami_data)
}

// Function to read all tsunami data files from a directory
fn read_tsunami_data(dir_path: &str, ncols: u32, nrows: u32) -> io::Result<Vec<Vec<Vec<u32>>>> {
    println!("Reading tsunami data from directory: {}", dir_path);
    let is_pacitan = dir_path.contains("pacitan");
    let is_sample = dir_path.contains("data_sample/tsunami_ascii_sample");

    // Define filter patterns
    let filter_pattern = |name: &str| -> bool {
        let lower_name = name.to_lowercase();
        // Exclude common auxiliary files explicitly
        if lower_name.ends_with(".aux.xml") || lower_name.ends_with(".tfw") || lower_name.ends_with(".prj") || lower_name.ends_with(".ovr") {
            return false;
        }
        // Include based on location pattern
        if is_pacitan {
            lower_name.starts_with("aav_rep_z_04_") && lower_name.ends_with(".asc")
        } else if is_sample {
            lower_name.starts_with("tsunami_") && lower_name.ends_with(".asc")
        } else { // Default (Jembrana-like)
            lower_name.ends_with("_processed.asc")
        }
    };

    // Get and filter file paths
    let mut tsunami_files: Vec<PathBuf> = fs::read_dir(dir_path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file() && // Ensure it's a file
            path.file_name()
                .and_then(|name| name.to_str())
                .map(filter_pattern)
                .unwrap_or(false)
        })
        .collect();

    println!("Found {} potential tsunami files matching pattern.", tsunami_files.len());

    // Sort files numerically based on the number in the filename
    tsunami_files.sort_by_key(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| {
                // Extract number based on pattern
                if is_pacitan {
                    name.trim_start_matches("aav_rep_z_04_").trim_end_matches(".asc")
                        .parse::<u32>().ok()
                } else if is_sample {
                     name.trim_start_matches("tsunami_").trim_end_matches(".asc")
                        .parse::<u32>().ok()
                } else { // Default (Jembrana)
                    name.trim_start_matches("z_07_").trim_end_matches("_processed.asc")
                        .parse::<u32>().ok()
                }
            })
            .unwrap_or(u32::MAX) // Put files that fail parsing at the end
    });

    // Print first few sorted files for verification
    // println!("First 5 sorted files:");
    // for file in tsunami_files.iter().take(5) {
    //     println!("  {:?}", file.file_name().unwrap_or_default());
    // }

    // Select files based on location-specific logic (e.g., sampling for Pacitan)
    let selected_files = if is_pacitan {
        // Select every 4th file for Pacitan (0, 28, 56, ...)
        tsunami_files.into_iter().step_by(4).collect::<Vec<_>>()
    } else {
        // Use all sorted files for other locations (or apply other limits if needed)
        // Example limit for Jembrana (if still desired):
        // if !is_sample && tsunami_files.len() > 68 {
        //     println!("Warning: Found more than 68 Jembrana files. Truncating.");
        //     tsunami_files.truncate(68);
        // }
        tsunami_files
    };

    println!("Selected {} files for processing.", selected_files.len());
    // if selected_files.len() <= 10 { // Print selection if it's short
    //      println!("Selected files:");
    //      for file in &selected_files {
    //          println!("  {:?}", file.file_name().unwrap_or_default());
    //      }
    // }


    // Process selected files in parallel
    let all_tsunami_data: Vec<_> = selected_files
        .par_iter() // Use parallel iterator
        .filter_map(|file_path| {
            match read_tsunami_data_file(file_path, ncols, nrows) {
                Ok(data) => Some(data),
                Err(e) => {
                    eprintln!("Error reading tsunami file {:?}: {}", file_path.file_name().unwrap_or_default(), e);
                    None // Skip files that cause errors
                }
            }
        })
        .collect();

    if all_tsunami_data.is_empty() && !selected_files.is_empty() {
         // If we selected files but couldn't read any, it's an error
         return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Failed to read any of the selected tsunami data files.",
        ));
    } else if all_tsunami_data.is_empty() {
         println!("Warning: No tsunami data files were found or processed successfully.");
         // Return empty vector, simulation will run without tsunami impact
    }


    println!("Successfully loaded {} tsunami data steps.", all_tsunami_data.len());
    if !all_tsunami_data.is_empty() {
        // Verify dimensions of the first loaded step
        let first_step = &all_tsunami_data[0];
        println!("First tsunami step dimensions: {} rows x {} cols", first_step.len(), first_step.get(0).map_or(0, |row| row.len()));
        if first_step.len() != nrows as usize || first_step.get(0).map_or(0, |row| row.len()) != ncols as usize {
             eprintln!("Warning: Loaded tsunami data dimensions do not match grid dimensions!");
        }
    }

    Ok(all_tsunami_data)
}


// --- Main Function ---
fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    // Check if the first argument is "api" to start the server
    if args.len() > 1 && args[1] == "api" {
        let port = args.get(2).and_then(|p| p.parse::<u16>().ok()).unwrap_or(8080);
        println!("Starting API server on port {}...", port);

        // Use Tokio runtime to run the async API server
        return tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(api::start_api_server(port)); // Assuming start_api_server is async

    } else {
        // --- Command-Line Execution (Example/Placeholder) ---
        // This part is NOT used by the Flask app but can be useful for direct testing
        // or running sensitivity analysis from the command line later.
        println!("Running simulation directly from command line (example)...");

        // Create a default config for the command-line run
        // We need to decide how to get tsunami_delay and agent_reaction_delay here.
        // For now, use some default values.
        let mut config = api::SimulationConfig::default(); // Load defaults (location, paths)
        config.tsunami_delay = 100; // Example default
        config.agent_reaction_delay = 50; // Example default
        config.max_steps = Some(1000); // Example max steps

        // You could add command-line argument parsing here to override defaults
        // e.g., using libraries like `clap`

        match run_simulation_instance(&config) {
            Ok(result) => {
                println!("Command-line simulation finished.");
                println!("Result: {:?}", result);
                // Optionally write result to a file
                let result_json = serde_json::to_string_pretty(&result)?;
                fs::write("output/cli_simulation_result.json", result_json)?;
                println!("Result saved to output/cli_simulation_result.json");
            }
            Err(e) => {
                eprintln!("Command-line simulation failed: {}", e);
                return Err(e);
            }
        }
    }

    Ok(())
}
// --- End Main Function ---
