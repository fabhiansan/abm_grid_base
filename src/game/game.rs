use super::agent::{Agent, AgentType, DeadAgentsData};
use super::grid::{Grid, Terrain};
use crate::ShelterData;
use crate::SimulationData;
use rand::prelude::*; // Import Rng for random number generation
use rand::seq::SliceRandom;
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;

#[derive(Clone)]
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
            println!("TSUNAMI IS COMING ----- {}", tsunami_number);

            // Check if tsunami_number is within bounds of tsunami_data
            if tsunami_number < self.grid.tsunami_data.len() {
                // Add debug info
                let total_positive_heights = self.grid.tsunami_data[tsunami_number]
                    .iter()
                    .map(|row| row.iter().filter(|&&height| height > 0).count())
                    .sum::<usize>();

                println!(
                    "Tsunami timestep {} has {} cells with positive height",
                    tsunami_number, total_positive_heights
                );

                // Print some tsunami height data points
                if total_positive_heights > 0 {
                    // Find and print first 5 positions with positive tsunami heights
                    let mut found_positions = 0;
                    for (y, row) in self.grid.tsunami_data[tsunami_number].iter().enumerate() {
                        for (x, &height) in row.iter().enumerate() {
                            if height > 0 {
                                println!("Tsunami at position ({}, {}): height {}", x, y, height);
                                found_positions += 1;
                                if found_positions >= 5 {
                                    break;
                                }
                            }
                        }
                        if found_positions >= 5 {
                            break;
                        }
                    }
                } else {
                    println!("WARNING: No positive tsunami heights in this timestep!");
                }

                // Print agent position stats
                let mut min_x = u32::MAX;
                let mut max_x = 0;
                let mut min_y = u32::MAX;
                let mut max_y = 0;

                for agent in &self.agents {
                    min_x = min_x.min(agent.x);
                    max_x = max_x.max(agent.x);
                    min_y = min_y.min(agent.y);
                    max_y = max_y.max(agent.y);
                }

                println!(
                    "Agent position range: x: ({} to {}), y: ({} to {})",
                    min_x, max_x, min_y, max_y
                );

                // Process agents in reverse order to safely remove them
                for i in (0..self.agents.len()).rev() {
                    let agent = &self.agents[i];

                    // Get tsunami height at agent position
                    let tsunami_height =
                        self.grid
                            .get_tsunami_height(tsunami_number, agent.x, agent.y);

                    // Debug print for first 5 agents
                    if i < 5 || tsunami_height > 0 {
                        println!(
                            "Agent {} at ({}, {}) - tsunami height: {}",
                            i, agent.x, agent.y, tsunami_height
                        );
                    }

                    // If tsunami height > 0, agent dies
                    if tsunami_height > 0 {
                        dead_agents_this_step += 1;
                        self.grid.remove_agent(agent.x, agent.y, i);
                        println!(
                            "Agent {} mati akibat tsunami pada koordinat ({}, {}) dengan ketinggian air {}",
                            i, agent.x, agent.y, tsunami_height
                        );

                        self.dead_agent_types.push(agent.agent_type);
                        self.agents.remove(i);
                    }
                }

                println!("Jumlah agen mati pada step ini: {}", dead_agents_this_step);
            } else {
                println!(
                    "Warning: No tsunami data found for tsunami number: {}",
                    tsunami_number
                );
            }
        }

        self.dead_agents += dead_agents_this_step;

        let mut rng = rand::thread_rng();

        // --- Trigger and Decision Logic (Paper 2 & 3) ---
        if is_tsunami {
            for agent in &mut self.agents {
                // Trigger: Mark when agent becomes aware (if not already)
                if agent.evacuation_trigger_time.is_none() {
                    agent.evacuation_trigger_time = Some(step);
                    // println!("Agent {} triggered at step {}", agent.id, step); // Optional debug
                }

                // Decision: Only decide once after being triggered
                if agent.evacuation_trigger_time.is_some() && !agent.has_decided_to_evacuate {
                    // Simple probability based on knowledge and household size
                    // Higher knowledge increases chance, larger household slightly decreases (placeholder logic)
                    let knowledge_factor = agent.knowledge_level as f32 / 100.0;
                    let household_factor = 1.0 - ((agent.household_size.saturating_sub(1)) as f32 * 0.05); // Small penalty per extra member
                    let evacuation_probability = (knowledge_factor * household_factor).clamp(0.0, 1.0);

                    if rng.gen::<f32>() < evacuation_probability {
                        agent.has_decided_to_evacuate = true;
                        // println!("Agent {} decided to evacuate (Prob: {:.2})", agent.id, evacuation_probability); // Optional debug
                    } else {
                         // println!("Agent {} decided NOT to evacuate (Prob: {:.2})", agent.id, evacuation_probability); // Optional debug
                    }
                }
            }
        }
        // --- End Trigger and Decision Logic ---


        let mut agent_order: Vec<usize> = (0..self.agents.len()).collect();

        // Reset remaining steps for all agents at the start of the main step
        for agent in &mut self.agents {
            agent.remaining_steps = agent.speed;
        }

        // --- Movement Loop ---
        // Determine max steps needed in this cycle based on fastest agent who decided to move
        let max_steps_needed = self.agents.iter()
                                .filter(|a| a.has_decided_to_evacuate) // Only consider agents who decided
                                .map(|a| a.speed)
                                .max()
                                .unwrap_or(0); // If no one decided, 0 steps

        for _ in 0..max_steps_needed {
            agent_order.shuffle(&mut rng);
            let mut reserved_cells = HashSet::new();
            let mut moves = Vec::new();

            for &id in &agent_order {
                let agent = &self.agents[id];

                // --- Movement Modification ---
                // Only move if agent has decided to evacuate, has steps left, and is not already in shelter
                if agent.has_decided_to_evacuate && agent.remaining_steps > 0 && !self.is_in_shelter(agent.x, agent.y) {
                    if let Some((nx, ny, fallback)) = self.find_best_move(agent, &reserved_cells) {
                        reserved_cells.insert((nx, ny));
                        moves.push((id, nx, ny, fallback));
                    }
                }
                // --- End Movement Modification ---
            }


            // Apply moves
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

                // Apply step cost - Now always 1, regardless of fallback status
                if agent.remaining_steps > 0 {
                    agent.remaining_steps -= 1;
                }
                // Optional: Log if a fallback move was taken
                // if fallback {
                //     println!("Agent {} took a fallback move to ({}, {})", id, new_x, new_y);
                // }

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
                // Sort candidates by distance (f64) using total_cmp
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
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
                        // Allow moving towards shelter even if occupied, as long as not reserved in this sub-step
                        && !reserved.contains(&(nx, ny))
                        // REMOVED: && self.grid.agents_in_cell[ny as usize][nx as usize].is_empty()
                    {
                        if let Some(dist) = self.grid.distance_to_shelter[ny as usize][nx as usize]
                        {
                            candidates.push((dist, nx, ny));
                        }
                    }
                }
            }

            if !candidates.is_empty() {
                // Sort candidates by distance (f64) using total_cmp
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
                let (_, nx, ny) = candidates[0];
                return Some((nx, ny, false));
            }
        }

        // --- Revised Fallback Logic ---
        let mut rng = rand::thread_rng();
        let mut valid_neighbors = Vec::new();

        // 1. Find all valid neighbors (within bounds, not blocked terrain)
        for &(dx, dy) in &dirs {
            let nx = agent.x as i32 + dx;
            let ny = agent.y as i32 + dy;
            if nx >= 0 && ny >= 0 && nx < self.grid.width as i32 && ny < self.grid.height as i32 {
                let nx = nx as u32;
                let ny = ny as u32;
                if self.grid.terrain[ny as usize][nx as usize] != Terrain::Blocked {
                    valid_neighbors.push((nx, ny));
                }
            }
        }

        if valid_neighbors.is_empty() {
            // println!("Agent {} has no valid neighbors", agent.id); // Debug
            return None; // No valid neighbors at all
        }

        // 2. Prioritize empty, non-reserved valid neighbors
        let empty_non_reserved: Vec<&(u32, u32)> = valid_neighbors
            .iter()
            .filter(|&&(nx, ny)| {
                !reserved.contains(&(nx, ny))
                    && self.grid.agents_in_cell[ny as usize][nx as usize].is_empty()
            })
            .collect();

        if !empty_non_reserved.is_empty() {
            let chosen = empty_non_reserved.choose(&mut rng).unwrap();
            // println!("Agent {} fallback to empty neighbor ({}, {})", agent.id, chosen.0, chosen.1); // Debug
            return Some((chosen.0, chosen.1, true)); // Mark as fallback (costly)
        }

        // 3. If no empty non-reserved, consider ANY non-reserved valid neighbor (might be occupied)
        let any_non_reserved: Vec<&(u32, u32)> = valid_neighbors
            .iter()
            .filter(|&&(nx, ny)| !reserved.contains(&(nx, ny)))
            .collect();

        if !any_non_reserved.is_empty() {
            let chosen = any_non_reserved.choose(&mut rng).unwrap();
            // println!("Agent {} fallback to potentially occupied neighbor ({}, {})", agent.id, chosen.0, chosen.1); // Debug
            // Keep cost same as empty fallback for now. Could increase cost later if needed.
            return Some((chosen.0, chosen.1, true));
        }

        // 4. If all valid neighbors are reserved in this sub-step, wait.
        // println!("Agent {} waiting (all valid neighbors reserved)", agent.id); // Debug
        None // No move possible in this sub-step
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
