mod game;

use game::agent::{Agent, AgentType};
use game::game::Model;
use game::grid::{load_grid_from_ascii, Grid, Terrain};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

        use std::fs::File;
use std::io::{self, BufRead, Write};

#[derive(Serialize, Deserialize)]
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

const TSUNAMI_DELAY: u32 = 30 * 60;
const TSUNAMI_SPEED_TIME: u32 = 28;

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
            crate::game::agent::AgentType::Car => "Car",
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
                        AgentType::Car => "7",
                    }
                    .to_string()
                } else {
                    // Fallback if agent not found (shouldn't happen)
                    "0".to_string()
                }
            } else {
                match model.grid.terrain[y][x] {
                    Terrain::Blocked => "0".to_string(),
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
        // let longitude = agent_data.x as f64 * collector.grid.cellsize + collector.grid.xllcorner;
        // let latitude = agent_data.y as f64 * collector.grid.cellsize + collector.grid.yllcorner;

        coordinates.push(vec![agent_data.x, agent_data.y]);
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
                "name": "EPSG:4326"
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
    
    Ok(events)
}

fn main() -> io::Result<()> {
    // Muat grid dan agen dari file ASC
    let (mut grid, mut agents) =
        load_grid_from_ascii("./19feb/jalantes.asc").expect("Failed to load grid");

    println!("grid width : {}, grid height {}", grid.width, grid.height);

    let mut next_agent_id = agents.len();

    let _ = load_population_and_create_agents(
        "./19feb/agenbener2m.asc",
        grid.width,
        grid.height,
        &mut grid,
        &mut agents,
        &mut next_agent_id,
    )
    .expect("Failed to populate grid");

    export_agent_statistics(&agents).expect("Failed to export agent statistics");

    println!("Loading tsunami data...");
    let tsunami_events = read_tsunami_events(
        "./data_pacitan/tsunami_pacitan/tsunami_pacitan_2/asc",
        grid.width,
        grid.height,
    ).expect("Failed to read tsunami data");
    
    let tsunami_len = tsunami_events.len();
    println!("Loaded {} tsunami events", tsunami_len);
    grid.tsunami_events = tsunami_events;

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

    while is_playing {
        let mut is_tsunami = false;

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
                                    AgentType::Car => shelter_agent_type_data.car += 1,
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
                if index >= tsunami_len {
                    is_playing = false;

                    break;
                }
                index += 1;
            }
        }
        current_step += 1;
        println!("Tsunami events memory: {:.1}KB", 
            model.grid.tsunami_events.len() as f64 * std::mem::size_of::<(u32,u32,u32)>() as f64 / 1024.0
        );
    }

    // Save shelter data with current dead agents count
    if let Err(e) = model.save_shelter_data(&death_json_counter, &shelter_json_counter) {
        eprintln!("Error saving shelter data: {}", e);
    }

    export_agents_to_geojson(&collector, "output/step.geojson")?;
    Ok(())
}

#[derive(Serialize, Deserialize, Default)]
pub struct DeadAgentTypeData {
    pub child: u32,
    pub teen: u32,
    pub adult: u32,
    pub elder: u32,
    pub car: u32,
    pub total: u32,
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

    // Simpan data populasi ke grid (jika grid memiliki field population)
    grid.population = population.clone();

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

// use std::fs::{self, File};
use std::path::{Path, PathBuf};

fn read_tsunami_data_file(path: &Path, ncols: u32, nrows: u32) -> io::Result<Vec<Vec<u32>>> {
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut lines = reader.lines();

    // Skip header lines
    for _ in 0..6 {
        lines.next();
    }

    let mut tsunami_data = Vec::new();

    for line in lines {
        let line = line?;
        let row: Vec<u32> = line
            .split_whitespace()
            // .iter()
            .take(ncols as usize)
            .filter_map(|token| token.parse::<f64>().ok().map(|val| val as u32))
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
    let mut tsunami_files: Vec<PathBuf> = fs::read_dir(dir_path)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                // .map(|name| name.starts_with("asc_tsunami_pacitan_") && name.ends_with(".asc"))
                .map(|name| name.ends_with(".asc"))
                .unwrap_or(false)
        })
        .collect();

    // Sort files by their numeric index (keep this sequential for correct ordering)
    tsunami_files.sort_by_key(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .and_then(|name| {
                name.trim_start_matches("aav_rep_z_04_")
                    .trim_end_matches(".asc")
                    .parse::<u32>()
                    .ok()
            })
            .unwrap_or(0)
    });

    // Process tsunami data files in parallel
    let all_tsunami_data: Vec<_> = tsunami_files
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

    Ok(all_tsunami_data)
}
