use crate::game::agent::{Agent, AgentType};
use crate::game::State;
use std::collections::BinaryHeap;
use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::io::{self, BufRead, BufReader};

#[derive(Clone, Debug, PartialEq)]
pub enum Terrain {
    Blocked,
    Road,
    Shelter(u32),  // Now includes shelter ID
}

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub width: u32,
    pub height: u32,
    pub xllcorner: f64,
    pub yllcorner: f64,
    pub cellsize: f64,
    pub terrain: Vec<Vec<Terrain>>,
    pub shelters: Vec<(u32, u32, u32)>,  // (x, y, shelter_id)

    pub agents_in_cell: Vec<Vec<Vec<usize>>>,
    // Jarak ke jalan/shelter sudah dihitung sebelumnya
    pub distance_to_road: Vec<Vec<Option<u32>>>,
    pub distance_to_shelter: Vec<Vec<Option<u32>>>,
    pub shelter_agents: HashMap<u32, Vec<(usize, AgentType)>>,
    pub population: Vec<Vec<u32>>,
    pub tsunami_events: Vec<(u32, u32, u32)>,  // (time_step, x, y)
    pub tsunami_data: Vec<Vec<Vec<u32>>>,  // [timestep][y][x] -> height
    pub nrow: u32,
    pub ncol: u32
}

impl Grid {
    pub fn remove_agent(&mut self, x: u32, y: u32, agent_id: usize) {
        // let cell = &mut self.agents_in_cell[y as usize][x as usize];
        // if let Some(pos) = cell.iter().position(|&id| id == agent_id) {
        //     cell.remove(pos);
        // }
        let y_usize = y as usize;
        let x_usize = x as usize;

        if y_usize < self.agents_in_cell.len() && x_usize < self.agents_in_cell[y_usize].len() {
            // Cari posisi ID agen di sel dan hapus jika ditemukan
            if let Some(index) = self.agents_in_cell[y_usize][x_usize]
                .iter()
                .position(|&id| id == agent_id)
            {
                self.agents_in_cell[y_usize][x_usize].remove(index);
            }
        }
    }

    pub fn add_to_shelter(&mut self, shelter_id: u32, agent_id: usize, agent_type: AgentType) {
        self.shelter_agents
            .entry(shelter_id)
            .or_insert(Vec::new())
            .push((agent_id, agent_type));
    }

    pub fn add_agent(&mut self, x: u32, y: u32, agent_id: usize) {
        self.agents_in_cell[y as usize][x as usize].push(agent_id);
    }

    pub fn compute_distance_to_shelters(&mut self) {
        let mut queue = VecDeque::new();
        let mut visited = vec![vec![false; self.width as usize]; self.height as usize];

        // Inisialisasi semua shelter dengan jarak 0
        for &(x, y, _) in &self.shelters {
            let x = x as usize;
            let y = y as usize;
            self.distance_to_shelter[y][x] = Some(0);
            queue.push_back((x, y));
            visited[y][x] = true;
        }

        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];

        while let Some((x, y)) = queue.pop_front() {
            let current_dist = self.distance_to_shelter[y][x].unwrap();

            for &(dx, dy) in &dirs {
                let nx = (x as i32) + dx;
                let ny = (y as i32) + dy;

                if nx >= 0 && ny >= 0 && nx < self.width as i32 && ny < self.height as i32 {
                    let nx = nx as usize;
                    let ny = ny as usize;

                    if !visited[ny][nx] && self.terrain[ny][nx] != Terrain::Blocked {
                        visited[ny][nx] = true;
                        self.distance_to_shelter[ny][nx] = Some(current_dist + 1);
                        queue.push_back((nx, ny));
                    }
                }
            }
        }
    }

    pub fn compute_road_distances_from_agents(&mut self) {
        // Buat matriks jarak, inisialisasi dengan None
        let mut dist = vec![vec![None; self.width as usize]; self.height as usize];
        let mut heap = BinaryHeap::new();

        // Inisialisasi: semua sel Road diberi jarak 0 dan dimasukkan ke dalam heap.
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                if self.terrain[y][x] == Terrain::Road {
                    dist[y][x] = Some(0);
                    heap.push(State {
                        cost: 0,
                        x: x as u32,
                        y: y as u32,
                    });
                }
            }
        }

        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];

        // Lakukan Dijkstra untuk menyebarkan jarak ke seluruh sel.
        while let Some(State { cost, x, y }) = heap.pop() {
            // Jika nilai cost di state sekarang sudah lebih besar daripada yang tersimpan, lewati.
            if let Some(current) = dist[y as usize][x as usize] {
                if cost > current {
                    continue;
                }
            }
            for &(dx, dy) in &dirs {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < self.width as i32 && ny < self.height as i32 {
                    let nx = nx as u32;
                    let ny = ny as u32;
                    // Tentukan biaya tambahan: jika sel tetangga Blocked, beri penalty (misalnya 2),
                    // jika tidak, biaya 1.
                    let extra_cost = if self.terrain[ny as usize][nx as usize] == Terrain::Blocked {
                        2 // Penalty untuk sel Blocked
                    } else {
                        1
                    };
                    let next_cost = cost + extra_cost;
                    // Jika sel tetangga belum dikunjungi atau kita menemukan jarak yang lebih pendek,
                    // update jarak tersebut dan masukkan ke heap.
                    if dist[ny as usize][nx as usize].is_none()
                        || next_cost < dist[ny as usize][nx as usize].unwrap()
                    {
                        dist[ny as usize][nx as usize] = Some(next_cost);
                        heap.push(State {
                            cost: next_cost,
                            x: nx,
                            y: ny,
                        });
                    }
                }
            }
        }
        // Simpan hasil perhitungan ke dalam field distance_to_road
        self.distance_to_road = dist;
    }

    pub fn is_valid_coordinate(&self, x: u32, y: u32) -> bool {
        x < self.width && y < self.height
    }

    pub fn get_terrain(&self, x: u32, y: u32) -> Option<&Terrain> {
        if self.is_valid_coordinate(x, y) {
            Some(&self.terrain[y as usize][x as usize])
        } else {
            None
        }
    }

    pub fn is_affected_by_tsunami(&self, time_step: u32, x: u32, y: u32) -> bool {
        self.tsunami_events.iter().any(|(t, ex, ey)| {
            *t == time_step && *ex == x && *ey == y
        })
    }

    pub fn get_tsunami_height(&self, tsunami_index: usize, x: u32, y: u32) -> u32 {
        if tsunami_index < self.tsunami_data.len() &&
           y < self.tsunami_data[tsunami_index].len() as u32 &&
           x < self.tsunami_data[tsunami_index][y as usize].len() as u32 {
            self.tsunami_data[tsunami_index][y as usize][x as usize]
        } else {
            0 // No tsunami at this position or invalid coordinates
        }
    }
}

pub fn load_grid_from_ascii(
    path: &str,
) -> Result<(Grid, Vec<crate::game::agent::Agent>), std::io::Error> {
    println!("Opening file {}", path);
    let content = std::fs::read_to_string(path)?;
    let mut lines = content.lines();

    // Baca dan parse header (6 baris)
    let ncols_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Missing ncols line"))?;
    let nrows_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Missing nrows line"))?;
    let xll_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Missing xllcorner line"))?;
    let yll_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Missing yllcorner line"))?;
    let cellsize_line = lines
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Missing cellsize line"))?;
    let _nodata_line = lines.next().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::Other, "Missing NODATA_value line")
    })?;

    // Parse nilai header
    let ncols: u32 = ncols_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Cannot parse ncols"))?
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid ncols value"))?;
    let nrows: u32 = nrows_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Cannot parse nrows"))?
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid nrows value"))?;
    let xllcorner: f64 = xll_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Cannot parse xllcorner"))?
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid xllcorner value"))?;
    let yllcorner: f64 = yll_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Cannot parse yllcorner"))?
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid yllcorner value"))?;
    let cellsize: f64 = cellsize_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::Other, "Cannot parse cellsize"))?
        .parse()
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "Invalid cellsize value"))?;

    println!("Cellsize: {}", cellsize);
    println!("Nrows: {}", nrows);
    println!("Ncols: {}", ncols);
    println!("xll_line: {}", xll_line);
    println!("yll_line: {}", yll_line);
    println!("Xllcorner: {:.10}", xllcorner);
    println!("Yllcorner: {:.10}", yllcorner);

    // Inisialisasi struktur grid sesuai dimensi yang didapatkan dari header
    let mut terrain = vec![vec![Terrain::Blocked; ncols as usize]; nrows as usize];
    let mut shelters = Vec::new();
    let mut agent_positions = Vec::new();
    let mut road_cells = Vec::new();

    // Iterasi tiap baris data (setelah header)
    for (y, line) in lines.enumerate() {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < ncols as usize {
            continue; // Atau kembalikan error jika data tidak lengkap
        }
        for (x, token) in tokens.iter().enumerate().take(ncols as usize) {
            terrain[y][x] = match *token {
                "0" | "0.0" => Terrain::Blocked,
                "1" => {
                    road_cells.push((x as u32, y as u32));
                    Terrain::Road
                }
                token if token.starts_with("20") => {
                    if let Ok(shelter_id) = token[2..].parse::<u32>() {
                        shelters.push((x as u32, y as u32, shelter_id));
                        Terrain::Shelter(shelter_id)
                    } else {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("Invalid shelter ID format: {}", token),
                        ));
                    }
                }
                // "2" => {
                //     shelters.push((x as u32, y as u32, 0));  // Default shelter ID
                //     Terrain::Shelter(0)
                // }
                "3" => {
                    agent_positions.push((x as u32, y as u32, AgentType::Adult));
                    Terrain::Road
                }
                "4" => {
                    // Child agent
                    agent_positions.push((x as u32, y as u32, AgentType::Child));
                    Terrain::Road
                }
                "5" => {
                    // Teen agent
                    agent_positions.push((x as u32, y as u32, AgentType::Teen));
                    Terrain::Road
                }
                "6" => {
                    // Elder agent
                    agent_positions.push((x as u32, y as u32, AgentType::Elder));
                    Terrain::Road
                }
                // "7" => {
                //     // Car agent
                //     agent_positions.push((x as u32, y as u32, AgentType::Car));
                //     Terrain::Road
                // }
                _ => Terrain::Blocked,
            };
        }
    }

    let agents: Vec<crate::game::agent::Agent> = agent_positions
        .into_iter()
        .enumerate()
        .map(|(index, (x, y, agent_type))| {
            let is_on_road = terrain[y as usize][x as usize] == Terrain::Road;
            crate::game::agent::Agent::new(index, x, y, agent_type, is_on_road) 
            // {
            //     id: index,
            //     x,
            //     y,
            //     speed: crate::game::agent::Agent::new(0, 0, agent_type).speed,
            //     remaining_steps: crate::game::agent::Agent::new(0, 0, agent_type).speed,
            //     is_on_road,
            //     agent_type,
            //     shelter_priority: crate::game::agent::Agent::new(0, 0, agent_type).shelter_priority,
            // }
        })
        .collect();

    let mut grid = Grid {
        width: ncols,
        height: nrows,
        xllcorner,
        yllcorner,
        cellsize,
        terrain,
        shelters,
        agents_in_cell: vec![vec![Vec::new(); ncols as usize]; nrows as usize],
        shelter_agents: HashMap::new(),
        distance_to_shelter: vec![vec![None; ncols as usize]; nrows as usize],
        distance_to_road: vec![vec![None; ncols as usize]; nrows as usize],
        population: vec![vec![0; ncols as usize]; nrows as usize],
        tsunami_events: Vec::new(),
        tsunami_data: Vec::new(),
        nrow: nrows,
        ncol: ncols
    };

    grid.compute_distance_to_shelters();
    grid.compute_road_distances_from_agents();

    grid.tsunami_data = Vec::new();

    Ok((grid, agents))
}
