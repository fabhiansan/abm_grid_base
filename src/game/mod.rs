use rand::seq::SliceRandom; // Impor trait SliceRandom

use std::cmp::Ordering;
use std::collections::BinaryHeap;

// Remove all derives, implement manually for heap
pub struct State {
    cost: f64, // Changed cost to f64
    x: u32,
    y: u32,
}

// Custom Ord implementation for min-heap behavior using f64 cost
impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse the comparison of costs for min-heap
        other.cost.total_cmp(&self.cost)
            // If costs are equal, break ties using position (optional but good practice)
            .then_with(|| self.y.cmp(&other.y))
            .then_with(|| self.x.cmp(&other.x))
    }
}

// Custom PartialOrd implementation
impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Custom PartialEq implementation (needed for Eq)
impl PartialEq for State {
     fn eq(&self, other: &Self) -> bool {
         // Two states are equal if their cost (using total_cmp for NaN/Inf safety)
         // AND their positions are the same.
         self.cost.total_cmp(&other.cost) == Ordering::Equal &&
         self.x == other.x &&
         self.y == other.y
     }
}

// Eq is required by Ord
impl Eq for State {}


pub mod grid;
pub mod agent;
pub mod game;
