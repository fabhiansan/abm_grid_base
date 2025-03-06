use super::agent::{Agent, AgentType, DeadAgentsData};
use super::grid::{Grid, Terrain};
use crate::ShelterData;
use crate::SimulationData;
use rand::prelude::IndexedRandom;
use rand::seq::SliceRandom;
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;

pub struct Model {
    pub grid: Grid,
    pub agents: Vec<Agent>,
    pub dead_agents: usize,
    pub dead_agent_types: Vec<AgentType>,
}

impl Model {
    pub fn step(&mut self, step: u32, is_tsunami: bool, tsunami_number: usize) {
        let mut dead_agents_this_step = 0;

        if is_tsunami {
            println!("TSUNAMI IS COMMING ----- {}", tsunami_number);
            let tsunami_data = self.grid.tsunami_data[tsunami_number].clone();

            for i in (0..self.agents.len()).rev() {
                let agent = &self.agents[i];
                if (agent.y as usize) < tsunami_data.len()
                    && (agent.x as usize) < tsunami_data[0].len()
                {
                    let tsunami_height = tsunami_data[agent.y as usize][agent.x as usize];
                    if tsunami_height > 0 {
                        dead_agents_this_step += 1;
                        self.grid.remove_agent(agent.x, agent.y, i);
                        println!(
                            "Agent {} mati akibat tsunami pada koordinat ({}, {})",
                            i, agent.x, agent.y
                        );

                        self.dead_agent_types.push(agent.agent_type);
                        self.agents.remove(i);
                    }
                }
            }
            println!("Jumlah agen mati pada step ini: {}", dead_agents_this_step);

            for row in self.grid.tsunami_data[tsunami_number].iter_mut() {
                for cell in row.iter_mut() {
                    if *cell != 0 {
                        *cell = 10;
                    }
                }
            }
        }

        self.dead_agents += dead_agents_this_step;

        let mut rng = rand::thread_rng();
        let mut agent_order: Vec<usize> = (0..self.agents.len()).collect();

        for agent in &mut self.agents {
            agent.remaining_steps = agent.speed;
        }

        for _ in 0..self.agents.iter().map(|a| a.speed).max().unwrap_or(1) {
            agent_order.shuffle(&mut rng);
            let mut reserved_cells = HashSet::new();
            let mut moves = Vec::new();

            for &id in &agent_order {
                let agent = &self.agents[id];
                if agent.remaining_steps == 0 || self.is_in_shelter(agent.x, agent.y) {
                    continue;
                }
                if let Some((nx, ny, fallback)) = self.find_best_move(agent, &reserved_cells) {
                    reserved_cells.insert((nx, ny));
                    moves.push((id, nx, ny, fallback));
                }
            }

            for &(id, new_x, new_y, fallback) in &moves {
                let (old_x, old_y) = {
                    let agent = &self.agents[id];
                    (agent.x, agent.y)
                };

                self.grid.remove_agent(old_x, old_y, id);

                let agent = &mut self.agents[id];
                let was_on_road = agent.is_on_road;
                agent.is_on_road =
                    self.grid.terrain[new_y as usize][new_x as usize] == Terrain::Road;

                if !was_on_road && agent.is_on_road {
                    // println!("Agent {} reached road at ({}, {})", id, new_x, new_y);
                }

                agent.x = new_x;
                agent.y = new_y;

                if fallback {
                    if agent.remaining_steps >= 2 {
                        agent.remaining_steps -= 2;
                    } else {
                        agent.remaining_steps = 0;
                    }
                } else {
                    agent.remaining_steps -= 1;
                }

                let in_shelter = self.is_in_shelter(new_x, new_y);
                if in_shelter {
                    self.enter_shelter(id, new_x, new_y);
                    // self.agents.remove(id);
                    self.grid.remove_agent(new_x, new_y, id);
                }

                self.grid.add_agent(new_x, new_y, id);
            }
        }
    }

    pub fn is_in_shelter(&self, x: u32, y: u32) -> bool {
        matches!(
            self.grid.terrain[y as usize][x as usize],
            Terrain::Shelter(_)
        )
    }

    pub fn enter_shelter(&mut self, agent_id: usize, x: u32, y: u32) {
        if let Terrain::Shelter(shelter_id) = self.grid.terrain[y as usize][x as usize] {
            let agent = &self.agents[agent_id];
            self.grid
                .add_to_shelter(shelter_id, agent_id, agent.agent_type);
        }
    }

    fn find_best_move(
        &self,
        agent: &Agent,
        reserved: &HashSet<(u32, u32)>,
    ) -> Option<(u32, u32, bool)> {
        let dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)];
        let mut candidates = Vec::new();

        if self.grid.terrain[agent.y as usize][agent.x as usize] != Terrain::Road {
            if let Some(current_dist) =
                self.grid.distance_to_road[agent.y as usize][agent.x as usize]
            {
                for &(dx, dy) in &dirs {
                    let nx = agent.x as i32 + dx;
                    let ny = agent.y as i32 + dy;

                    if nx >= 0
                        && ny >= 0
                        && nx < self.grid.width as i32
                        && ny < self.grid.height as i32
                    {
                        let nx = nx as u32;
                        let ny = ny as u32;

                        if !reserved.contains(&(nx, ny))
                            && self.grid.agents_in_cell[ny as usize][nx as usize].is_empty()
                        {
                            if let Some(new_dist) =
                                self.grid.distance_to_road[ny as usize][nx as usize]
                            {
                                if new_dist <= current_dist {
                                    candidates.push((new_dist, nx, ny));
                                }
                            }
                        }
                    }
                }
            }

            if !candidates.is_empty() {
                candidates.sort_by_key(|&(d, _, _)| d);
                let (_, nx, ny) = candidates[0];
                return Some((nx, ny, false));
            }
        } else if self.grid.terrain[agent.y as usize][agent.x as usize] == Terrain::Road {
            for &(dx, dy) in &dirs {
                let nx = agent.x as i32 + dx;
                let ny = agent.y as i32 + dy;

                if nx >= 0 && ny >= 0 && nx < self.grid.width as i32 && ny < self.grid.height as i32
                {
                    let nx = nx as u32;
                    let ny = ny as u32;

                    if matches!(
                        self.grid.terrain[ny as usize][nx as usize],
                        Terrain::Shelter(_)
                    ) && !reserved.contains(&(nx, ny))
                    {
                        return Some((nx, ny, false));
                    }

                    if (self.grid.terrain[ny as usize][nx as usize] == Terrain::Road
                        || matches!(
                            self.grid.terrain[ny as usize][nx as usize],
                            Terrain::Shelter(_)
                        ))
                        && !reserved.contains(&(nx, ny))
                        && self.grid.agents_in_cell[ny as usize][nx as usize].is_empty()
                    {
                        if let Some(dist) = self.grid.distance_to_shelter[ny as usize][nx as usize]
                        {
                            candidates.push((dist, nx, ny));
                        }
                    }
                }
            }

            if !candidates.is_empty() {
                candidates.sort_by_key(|&(d, _, _)| d);
                let (_, nx, ny) = candidates[0];
                return Some((nx, ny, false));
            }
        }

        let fallback_moves: Vec<(u32, u32)> = dirs
            .iter()
            .filter_map(|&(dx, dy)| {
                let nx = agent.x as i32 + dx;
                let ny = agent.y as i32 + dy;
                if nx >= 0 && ny >= 0 && nx < self.grid.width as i32 && ny < self.grid.height as i32
                {
                    let nx = nx as u32;
                    let ny = ny as u32;
                    if !reserved.contains(&(nx, ny))
                        && self.grid.agents_in_cell[ny as usize][nx as usize].is_empty()
                    {
                        Some((nx, ny))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();

        if !fallback_moves.is_empty() {
            let mut rng = rand::thread_rng();
            let chosen = fallback_moves.choose(&mut rng).unwrap();
            Some((chosen.0, chosen.1, true))
        } else {
            None
        }
    }

    pub fn save_shelter_data(
        &self,
        death_json_counter: &Vec<serde_json::Value>,
        shelter_json_counter: &Vec<serde_json::Value>,
    ) -> std::io::Result<()> {
        let filename = "output/shelter_data.json";

        let mut shelter_counts: HashMap<String, ShelterAgentCounts> = HashMap::new();

        for (&shelter_id, agents) in &self.grid.shelter_agents {
            let shelter_key = format!("shelter_{}", shelter_id);
            let counts = shelter_counts
                .entry(shelter_key)
                .or_insert(ShelterAgentCounts::new());

            for &(_, agent_type) in agents {
                match agent_type {
                    AgentType::Child => counts.child += 1,
                    AgentType::Teen => counts.teen += 1,
                    AgentType::Adult => counts.adult += 1,
                    AgentType::Elder => counts.elder += 1,
                }
            }
        }

        let current_shelter_data: HashMap<String, serde_json::Value> = shelter_counts
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    json!({
                        "child": v.child,
                        "teen": v.teen,
                        "adult": v.adult,
                        "elder": v.elder,
                    }),
                )
            })
            .collect();

        let data = json!({
            "death_json_counter": death_json_counter,
            "shelter_json_counter": shelter_json_counter,
            "shelter_agent_types": current_shelter_data,
        });

        let file = File::create(filename)?;
        serde_json::to_writer_pretty(file, &data)?;
        println!("Updated simulation data in {}", filename);

        Ok(())
    }
}

pub struct ShelterAgentCounts {
    pub child: u32,
    pub teen: u32,
    pub adult: u32,
    pub elder: u32,
}

impl ShelterAgentCounts {
    pub fn new() -> Self {
        ShelterAgentCounts {
            child: 0,
            teen: 0,
            adult: 0,
            elder: 0,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        json!({
            "child": self.child,
            "teen": self.teen,
            "adult": self.adult,
            "elder": self.elder,
        })
    }
}
