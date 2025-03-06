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

use std::{fs, path};
fn main() -> io::Result<()> {
    // Muat grid dan agen dari file ASC
    let (mut grid, mut agents) =
        // load_grid_from_ascii("./data_pacitan/jalandantes2m.asc").expect("Failed to load grid");
        load_grid_from_ascii("./Jembrana New/jalan/jalantes.asc").expect("Failed to load grid");

    println!("grid width : {}, grid height {}", grid.width, grid.height);

    let mut next_agent_id = agents.len();

    let _ = load_population_and_create_agents(
        // "./data_pacitan/agent2mboundariesjalan_reprojected.asc",
        "./Jembrana New/agen/agenjembrana.asc",
        grid.width,
        grid.height,
        &mut grid,
        &mut agents,
        &mut next_agent_id,
    )
    .expect("Failed to populate grid");

    export_agent_statistics(&agents).expect("Failed to export agent statistics");
    
    let tsunami_data = read_tsunami_data(
        // "./data_pacitan/git /tsunami_pacitan_all",
        "./Jembrana New/tsunami_ascii",
        grid.width,
        grid.height,
    )
    .expect("Failed to read tsunami data");

    println!("Loaded {} tsunami data files", tsunami_data.len());
    grid.tsunami_data = tsunami_data;
    let tsunami_len = grid.tsunami_data.len();
    println!("Number of tsunami data {}", tsunami_len);

    fs::create_dir_all("output").expect("Gagal membuat folder output");

    let mut model = Model {
        grid,
        agents,
        dead_agents: 0,
        dead_agent_types: Vec::new(),
    };

    let mut death_json_counter: Vec<serde_json::Value> = Vec::new();
    let mut shelter_json_counter: Vec<serde_json::Value> = Vec::new();
    let mut current_step = 0;

    let num_steps = 100;

    while current_step < num_steps {
        let mut is_tsunami = false;
        let mut index = -1 as isize;

        if current_step % 4 == 0 && current_step != 0 {
            is_tsunami = true;

            if index > tsunami_len as isize {
                continue;
            } else {
                index += 1;
            }
        }

        model.step(current_step, is_tsunami, index as usize);
        println!("Step {}", current_step);

        if current_step % 10 == 0 {
            let mut dead_agent_counts = DeadAgentTypeData::default();
            
            // Count dead agents by type
            for agent in &model.dead_agent_types {
                match agent {
                    AgentType::Child => dead_agent_counts.child += 1,
                    AgentType::Teen => dead_agent_counts.teen += 1,
                    AgentType::Adult => dead_agent_counts.adult += 1,
                    AgentType::Elder => dead_agent_counts.elder += 1,
                    AgentType::Car => dead_agent_counts.car += 1,
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
                    "car": dead_agent_counts.car,
                    "total": dead_agent_counts.total
                }
            }));

            // Add step information to shelter data
            let mut shelter_info: HashMap<String, ShelterAgentTypeData> = model
                .grid
                .shelters
                .iter()
                .map(|&(_, _, id)| {
                    let key = format!("shelter_{}", id);
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

            shelter_info.insert("step".to_string(), ShelterAgentTypeData {
                child: current_step as u32,
                teen: 0,
                adult: 0,
                elder: 0,
                car: 0,
            });

            shelter_json_counter.push(json!(shelter_info));

            let filename = format!("output/step_{}.asc", current_step);
            if let Err(e) = write_grid_to_ascii(&filename, &model) {
                eprintln!("Error writing {}: {}", filename, e);
            } else {
                println!("Saved output to {}", filename);
            }
        }

        current_step += 1;
    }

    // Save shelter data with current dead agents count
    if let Err(e) = model.save_shelter_data(&death_json_counter, &shelter_json_counter) {
        eprintln!("Error saving shelter data: {}", e);
    }

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
            for _ in 0..pop {
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
            .filter_map(|token| token.parse::<f64>().ok().map(|val| val as u32 + 1000))
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

    // Sort files by their numeric index
    tsunami_files.sort_by_key(|path| {
        path.file_name()
            .and_then(|name| name.to_str())
            .and_then(|name|
                // 
                name    
                          .trim_end_matches(".asc")
                          .parse::<u32>()
                          .ok())
            .unwrap_or(0)
    });

    // Read all tsunami data files
    let mut all_tsunami_data = Vec::new();
    for file_path in tsunami_files {
        match read_tsunami_data_file(&file_path, ncols, nrows) {
            Ok(data) => all_tsunami_data.push(data),
            Err(e) => eprintln!("Error reading tsunami file {:?}: {}", file_path, e),
        }
    }

    if all_tsunami_data.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "No valid tsunami data files found",
        ));
    }

    Ok(all_tsunami_data)
}