mod game;
mod api;

use game::agent::{Agent, AgentType};
use game::game::Model;
use game::grid::{load_grid_from_ascii, Grid, Terrain};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::env;

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

// const TSUNAMI_DELAY: u32 = 100; 
// const TSUNAMI_DELAY: u32 = 30 * 60;
const TSUNAMI_DELAY: u32 = 5;
const TSUNAMI_SPEED_TIME: u32 = 2;

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
    std::fs::write("simulation_data.json", json)?;

    Ok(())
}

pub const DISTRIBUTION_WEIGHTS: [i32; 5] = [10, 20, 30, 15, 20];

fn write_grid_to_ascii(filename: &str, model: &Model) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(filename)?;

    // Tulis header ASC dengan nilai dari grid
    writeln!(file, "ncols        {}", model.grid.width)?;
    writeln!(file, "nrows        {}", model.grid.height)?;
    writeln!(file, "xllcorner    {}", model.grid.xllcorner)?;
    writeln!(file, "yllcorner    {}", model.grid.yllcorner)?;
    writeln!(file, "cellsize     {}", model.grid.cellsize)?;
    writeln!(file, "NODATA_value  0")?;

    // Tulis data grid: tiap baris dipisahkan spasi
    for y in 0..model.grid.height as usize {
        let mut row_tokens = Vec::with_capacity(model.grid.width as usize);
        for x in 0..model.grid.width as usize {
            let token = if !model.grid.agents_in_cell[y][x].is_empty() {
                // Get the first agent in the cell (if multiple agents exist)
                let agent_id = model.grid.agents_in_cell[y][x][0];
                // Find the agent with this ID
                if let Some(agent) = model.agents.iter().find(|a| a.id == agent_id) {
                    match agent.agent_type {
                        AgentType::Child => "3",
                        AgentType::Teen => "4",
                        AgentType::Adult => "5",
                        AgentType::Elder => "6",
                        // AgentType::Car => "7",
                    }
                    .to_string()
                } else {
                    // Fallback if agent not found (shouldn't happen)
                    "0".to_string()
                }
            } else {
                match model.grid.terrain[y][x] {
                    Terrain::Land => "0".to_string(), // Explicitly handle Land as 0
                    Terrain::Blocked => "0".to_string(), // Keep Blocked as 0 for now (or choose another NODATA value if needed)
                    Terrain::Road => "1".to_string(),
                    Terrain::Shelter(id) => format!("20{:02}", id),
                }
            };
            row_tokens.push(token);
        }
        let row_line = row_tokens.join(" ");
        writeln!(file, "{}", row_line)?;
    }
    Ok(())
}

use rayon::prelude::*;
use std::{fs, path};

// Structure to store agent data for each step
#[derive(Clone)]
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
    grid: Grid,
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
                let real_x = self.grid.xllcorner + (agent.x as f64 * self.grid.cellsize);
                let real_y = self.grid.yllcorner
                    + (-1.0 * agent.y as f64 * self.grid.cellsize)
                    + (self.grid.nrow as f64 * self.grid.cellsize);

                self.data.push(AgentStepData {
                    x: real_x,
                    y: real_y,
                    id: agent.id,
                    agent_type: format!("{:?}", agent.agent_type),
                    is_on_road: agent.is_on_road,
                    speed: agent.speed,
                    step,
                });
            }
        }
    }
}

fn export_agents_to_geojson(collector: &AgentDataCollector, filename: &str) -> std::io::Result<()> {
    use serde_json::{json, Value};
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Write;

    let mut grouped_data: HashMap<(u32, String), Vec<Vec<f64>>> = HashMap::new();

    for agent_data in &collector.data {
        let key = (agent_data.step, agent_data.agent_type.clone());
        let coordinates = grouped_data.entry(key).or_insert_with(Vec::new);

        // Convert grid coordinates to geographic coordinates
        let x_utm = agent_data.x as f64 * collector.grid.cellsize + collector.grid.xllcorner;
        let y_utm = agent_data.y as f64 * (-1.0 * collector.grid.cellsize) + collector.grid.yllcorner + (collector.grid.nrow as f64 * collector.grid.cellsize) ;

        coordinates.push(vec![x_utm, y_utm]);
    }

    let features: Vec<Value> = grouped_data
        .into_iter()
        .map(|((step, agent_type), coordinates)| {
            json!({
                "type": "Feature",
                "geometry": {
                    "type": "MultiPoint",
                    "coordinates": coordinates
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
                "name": "EPSG:32749"
            }
        },
        "features": features
    });

    let mut file = File::create(filename)?;
    file.write_all(serde_json::to_string_pretty(&geojson)?.as_bytes())?;

    Ok(())
}

fn read_tsunami_events(dir_path: &str, grid_width: u32, grid_height: u32) -> io::Result<Vec<(u32, u32, u32)>> {
    let mut events = Vec::new();
    let paths = fs::read_dir(dir_path)?;
    
    for (time_step, entry) in paths.enumerate() {
        let entry = entry?;
        let path = entry.path();
        
        if path.extension().and_then(|s| s.to_str()) == Some("asc") {
            let file = File::open(&path)?;
            let reader = io::BufReader::new(file);
            
            for (y, line) in reader.lines().enumerate() {
                let line = line?;
                for (x, value) in line.split_whitespace().enumerate() {
                    if let Ok(height) = value.parse::<u32>() {
                        if height > 0 {
                            events.push((time_step as u32, x as u32, y as u32));
                        }
                    }
                }
            }
        }
    }

    println!("{:?}", events[0]);
    
    Ok(events)
}

fn main() -> io::Result<()> {
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    // If first argument is "api", start API server
    if args.len() > 1 && args[1] == "api" {
        // Get port from arguments or use default
        let port = if args.len() > 2 {
            args[2].parse::<u16>().unwrap_or(8080)
        } else {
            8080
        };
        
        // Start API server
        return tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(api::start_api_server(port));
    }
    
    // Otherwise, run simulation directly
    
    // Get default config location
    let config = api::SimulationConfig::default();
    let (grid_path, population_path, tsunami_data_path) = config.get_location_paths();
    
    println!("Using location: {}", config.location);
    println!("Grid path: {}", grid_path);
    println!("Population path: {}", population_path);
    println!("Tsunami data path: {}", tsunami_data_path);

    // Muat grid dan agen dari file ASC
    let (mut grid, mut agents) =
        load_grid_from_ascii(&grid_path).expect("Failed to load grid");

    println!("grid width : {}, grid height {}", grid.width, grid.height);

    let mut next_agent_id = agents.len();

    let _ = load_population_and_create_agents(
        &population_path,
        grid.width,
        grid.height,
        &mut grid,
        &mut agents,
        &mut next_agent_id,
    )
    .expect("Failed to populate grid");

    export_agent_statistics(&agents).expect("Failed to export agent statistics");

    println!("Loading tsunami data...");
    let tsunami_data = read_tsunami_data(&tsunami_data_path, grid.ncol, grid.nrow)?;
    println!("Loaded {} tsunami inundation timesteps", tsunami_data.len());
    let tsunami_len = tsunami_data.len();
    grid.tsunami_data = tsunami_data; // Assign the loaded tsunami data to grid
    println!("Tsunami data length: {}", tsunami_len);

    // println!("Loading tsunami inundation data...");
    // let tsunami_data = read_tsunami_data(
    //     "./tsunami_ascii/",
    //     grid.width,
    //     grid.height,
    // ).expect("Failed to read tsunami inundation data");
    // println!("Loaded {} tsunami inundation timesteps", tsunami_data.len());
    // let tsunami_len = tsunami_data.len();
    // grid.tsunami_data = tsunami_data;

    let mut model = Model {
        grid,
        agents,
        dead_agents: 0,
        dead_agent_types: Vec::new(),
    };

    // Create collector with its own copy of the grid
    let mut collector = AgentDataCollector {
        data: Vec::new(),
        grid: model.grid.clone(),
    };

    let mut death_json_counter: Vec<serde_json::Value> = Vec::new();
    let mut shelter_json_counter: Vec<serde_json::Value> = Vec::new();
    let mut current_step = 0;
    let mut index = 0;
    let mut is_playing = true;
    let mut is_tsunami = false;

    while is_playing {

        model.step(current_step, is_tsunami, index);
        println!("Step : {} Tsunami Index : {}", current_step, index);

        if current_step % 30 == 0 {
            let mut dead_agent_counts = DeadAgentTypeData::default();

            // Count dead agents by type
            for agent in &model.dead_agent_types {
                match agent {
                    AgentType::Child => dead_agent_counts.child += 1,
                    AgentType::Teen => dead_agent_counts.teen += 1,
                    AgentType::Adult => dead_agent_counts.adult += 1,
                    AgentType::Elder => dead_agent_counts.elder += 1,
                }
                dead_agent_counts.total += 1;
            }

            death_json_counter.push(json!({
                "step": current_step,
                "dead_agents": {
                    "child": dead_agent_counts.child,
                    "teen": dead_agent_counts.teen,
                    "adult": dead_agent_counts.adult,
                    "elder": dead_agent_counts.elder,
                    "total": dead_agent_counts.total
                }
            }));

            // Add step information to shelter data
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
                                    // AgentType::Car => shelter_agent_type_data.car += 1,
                                }
                            }
                            shelter_agent_type_data
                        })
                        .unwrap_or(ShelterAgentTypeData::default());
                    (key, count)
                })
                .collect();

            shelter_json_counter.push(json!(shelter_info));

            // let filename = format!("output/step_{}.asc", current_step);
            // if let Err(e) = write_grid_to_ascii(&filename, &model) {
            //     eprintln!("Error writing {}: {}", filename, e);
            // } else {
            //     println!("Saved output to {}", filename);
            // }

            collector.collect_step(&model, current_step);
        }
        if current_step > TSUNAMI_DELAY {
            is_tsunami = true;

            if current_step % TSUNAMI_SPEED_TIME == 0 && current_step != 0 && is_tsunami {
                if index >= tsunami_len - 1 {
                    println!("Reached final tsunami index. Ending simulation.");
                    is_playing = false;
                    break;
                }
                index += 1;
                println!("Advancing to tsunami index: {}", index);
            }
        }
        current_step += 1;
    }

    // Save shelter data with current dead agents count
    if let Err(e) = model.save_shelter_data(&death_json_counter, &shelter_json_counter) {
        eprintln!("Error saving shelter data: {}", e);
    }

    export_agents_to_geojson(&collector, "output/step.geojson")?;
    Ok(())
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct DeadAgentTypeData {
    pub child: u32,
    pub teen: u32,
    pub adult: u32,
    pub elder: u32,
    pub car: u32,
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

#[derive(Serialize, Deserialize)]
pub struct DeadAgentData {
    pub step: u32,
    pub dead_agents: DeadAgentTypeData,
}

pub fn load_population_and_create_agents(
    path: &str,
    ncols: u32,
    nrows: u32,
    grid: &mut Grid,
    agents: &mut Vec<crate::game::agent::Agent>,
    next_agent_id: &mut usize,
) -> std::io::Result<()> {
    // Buka file dan baca isinya
    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);
    let mut lines = reader.lines();

    // Lewati 6 baris header
    for _ in 0..6 {
        lines.next();
    }

    // Baca data populasi ke dalam vector 2D
    let mut population: Vec<Vec<u32>> = Vec::with_capacity(nrows as usize);
    for line in lines {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < ncols as usize {
            continue;
        }
        let row: Vec<u32> = tokens
            .iter()
            .take(ncols as usize)
            .map(|token| token.parse::<u32>().unwrap_or(0))
            .collect();
        population.push(row);
    }

    if population.len() != nrows as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Dimensi data populasi tidak sesuai dengan grid.",
        ));
    }

    // grid.population field removed from Grid struct, so assignment is removed.
    // Iterasi data populasi dan tambahkan agen untuk setiap unit populasi
    for (y, row) in population.iter().enumerate() {
        for (x, &pop) in row.iter().enumerate() {
            // println!("pop {pop}");
            if pop != 0 {
                for _ in 0..1 {
                    let is_on_road = grid.terrain[y][x] == Terrain::Road;
                    let agent_type = crate::game::agent::AgentType::random();
    
                    let mut agent = crate::game::agent::Agent::new(
                        *next_agent_id,
                        x as u32,
                        y as u32,
                        agent_type,
                        is_on_road,
                    );
                    // Inisialisasi lebih lanjut untuk agen
                    agent.id = *next_agent_id;
                    agent.remaining_steps = agent.speed;
                    agent.is_on_road = is_on_road;
    
                    // Tambahkan agen ke grid dan vektor agen
                    grid.add_agent(x as u32, y as u32, agent.id);
                    agents.push(agent);
                    *next_agent_id += 1;
                }
            }
        }
    }

    Ok(())
}

pub fn load_population_from_ascii(path: &str, ncols: u32, nrows: u32) -> io::Result<Vec<Vec<u32>>> {
    let file = std::fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();
    
    // Lewati 6 baris header
    for _ in 0..6 {
        lines.next();
    }
    
    let mut population: Vec<Vec<u32>> = Vec::with_capacity(nrows as usize);
    
    for line in lines {
        let line = line?;
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < ncols as usize {
            continue;
        }
        let row: Vec<u32> = tokens
            .iter()
            .take(ncols as usize)
            .map(|token| token.parse::<u32>().unwrap_or(0))
            .collect();
        population.push(row);
    }
    
    if population.len() != nrows as usize {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            "Dimensi data populasi tidak sesuai dengan grid.",
        ));
    }
    
    Ok(population)
}

fn read_tsunami_data_file(path: &Path, ncols: u32, nrows: u32) -> io::Result<Vec<Vec<u32>>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    // Determine the location based on the file path
    let is_pacitan = path.to_string_lossy().contains("pacitan");
    
    // Look for NODATA_value in the header
    let mut nodata_value: Option<f64> = None;
    
    // Read the headers
    let mut header_lines = Vec::new();
    for _ in 0..6 {
        if let Some(Ok(line)) = lines.next() {
            // Try to extract NODATA_value from the header if present
            if line.contains("NODATA_value") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    if let Ok(val) = parts[parts.len() - 1].parse::<f64>() {
                        nodata_value = Some(val);
                    }
                }
            }
            header_lines.push(line);
        }
    }
    
    // If NODATA_value was not found in the header, set default based on the location
    if nodata_value.is_none() {
        nodata_value = if is_pacitan { Some(0.0) } else { Some(-9999.0) };
    }
    
    // Print debug information without too much spam
    if is_pacitan {
        println!("Processing Pacitan tsunami file: {:?} with NODATA_value: {:?}", 
                 path.file_name().unwrap_or_default(), nodata_value);
    }
    
    let mut tsunami_data = Vec::new();

    for line in lines {
        let line = line?;
        let row: Vec<u32> = line
            .split_whitespace()
            .take(ncols as usize)
            .filter_map(|token| {
                token.parse::<f64>().ok().map(|val| {
                    // If the value matches NODATA_value, it means NO tsunami (height 0)
                    // Otherwise keep the non-negative height value as is
                    if let Some(nodata) = nodata_value {
                        if (val - nodata).abs() < 0.001 {
                            0 // No tsunami
                        } else {
                            val.max(0.0) as u32 // Keep positive tsunami heights
                        }
                    } else {
                        val.max(0.0) as u32 // Fallback, ensure non-negative
                    }
                })
            })
            .collect();
        tsunami_data.push(row);
    }

    // Fill missing rows with zeros if needed
    while tsunami_data.len() < nrows as usize {
        tsunami_data.push(vec![0; ncols as usize]);
    }

    Ok(tsunami_data)
}

fn read_tsunami_data(dir_path: &str, ncols: u32, nrows: u32) -> io::Result<Vec<Vec<Vec<u32>>>> {
    // Determine which location/data type we're using based on the directory path
    let is_pacitan = dir_path.contains("pacitan");
    let is_sample = dir_path.contains("data_sample/tsunami_ascii_sample"); // Check for sample path

    // Define file patterns based on location/type
    let filter_pattern = if is_pacitan {
        // log::debug!("Using Pacitan filter pattern for path: {}", dir_path);
        // Pacitan files use pattern: aav_rep_z_04_XXXXXX.asc
        |name: &str| name.contains("aav_rep_z_04_") && name.ends_with(".asc") && !name.ends_with(".asc.aux.xml")
    } else if is_sample {
        // log::debug!("Using Sample filter pattern for path: {}", dir_path);
        // Sample files use pattern: tsunami_XXX.asc
        |name: &str| name.starts_with("tsunami_") && name.ends_with(".asc") && !name.ends_with(".asc.aux.xml")
    } else {
        // log::debug!("Using default (Jembrana-like) filter pattern for path: {}", dir_path);
        // Default (Jembrana) files use pattern: z_07_XXXXXX_processed.asc
        |name: &str| name.ends_with("_processed.asc") && !name.ends_with(".asc.aux.xml")
    };
    
    // Get tsunami files using the appropriate pattern
    let mut tsunami_files: Vec<PathBuf> = fs::read_dir(dir_path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(filter_pattern)
                .unwrap_or(false)
        })
        .collect();

    println!("Found {} potential tsunami files", tsunami_files.len());
    
    // Sort files by their numeric index (keep this sequential for correct ordering)
    tsunami_files.sort_by_key(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| {
                if is_pacitan {
                    // Extract number from Pacitan format: aav_rep_z_04_000000.asc
                    name.trim_start_matches("aav_rep_z_04_")
                        .trim_end_matches(".asc")
                        .parse::<u32>()
                        .ok()
                } else if is_sample {
                     // Extract number from Sample format: tsunami_000.asc
                     name.trim_start_matches("tsunami_")
                        .trim_end_matches(".asc")
                        .parse::<u32>()
                        .ok()
                } else {
                    // Extract number from Jembrana format: z_07_000000_processed.asc
                    name.trim_start_matches("z_07_")
                        .trim_end_matches("_processed.asc")
                        .parse::<u32>()
                        .ok()
                }
            })
            .unwrap_or(0)
    });

    // Print the first few sorted files for debugging
    println!("First 5 files after sorting:");
    for (i, file) in tsunami_files.iter().take(5).enumerate() {
        println!("{}: {:?}", i, file.file_name().unwrap_or_default());
    }
    
    // Select specific files based on location
    let selected_files = if is_pacitan {
        // For Pacitan, use a simpler approach by skipping 3 files at a time (each file is +7)
        // This will give us files at intervals of 0, 28, 56, 84, 112, etc.
        
        // Sort tsunami files by their numeric index
        let mut indexed_files: Vec<(u32, PathBuf)> = tsunami_files.iter()
            .filter_map(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .and_then(|name| {
                        name.trim_start_matches("aav_rep_z_04_")
                            .trim_end_matches(".asc")
                            .parse::<u32>()
                            .ok()
                            .map(|num| (num, path.clone()))
                    })
            })
            .collect();
        
        // Sort files by their index
        indexed_files.sort_by_key(|(idx, _)| *idx);
        
        // Select every 4th file (skipping 3 files each time)
        let mut selected = Vec::new();
        for i in (0..indexed_files.len()).step_by(4) {
            let (idx, file) = &indexed_files[i];
            println!("Selected tsunami file with index {}", idx);
            selected.push(file.clone());
        }
        
        println!("Selected {} tsunami files for Pacitan at intervals of 0, 28, 56, ...", selected.len());
        selected
    } else {
        // For Jembrana, use all files (up to 68 as before)
        if tsunami_files.len() > 68 {
            println!("total tsunami len: {}", tsunami_files.len());
            println!("Warning: Found more than 68 tsunami files. Truncating to the first 68.");
            tsunami_files.truncate(68);
        }
        tsunami_files
    };
    
    println!("Selected {} tsunami files for processing", selected_files.len());
    if selected_files.len() > 0 && selected_files.len() <= 5 {
        println!("Selected files:");
        for (i, file) in selected_files.iter().enumerate() {
            println!("{}: {:?}", i, file.file_name().unwrap_or_default());
        }
    }

    // Process tsunami data files in parallel
    let all_tsunami_data: Vec<_> = selected_files
        .par_iter()
        .filter_map(
            |file_path| match read_tsunami_data_file(file_path, ncols, nrows) {
                Ok(data) => Some(data),
                Err(e) => {
                    eprintln!("Error reading tsunami file {:?}: {}", file_path, e);
                    None
                }
            },
        )
        .collect();

    if all_tsunami_data.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No valid tsunami data files found",
        ));
    }

    println!("Loaded {} tsunami data files", all_tsunami_data.len());
    if !all_tsunami_data.is_empty() {
        println!("Tsunami grid dimensions: {}x{}", 
                all_tsunami_data[0].len(), 
                all_tsunami_data[0][0].len());
    }
    
    Ok(all_tsunami_data)
}
