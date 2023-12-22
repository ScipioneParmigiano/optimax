///! Module containing the implementation of the stochastic annealing method
use rand::Rng;

#[derive(Debug, Clone)]
pub struct Solution {
    pub coords: Vec<f64>,
    pub value: f64,
}

impl Solution {
    pub fn new(coords: Vec<f64>, value: f64) -> Solution {
        Solution { coords, value }
    }
}

fn acceptance_probability(old_energy: f64, new_energy: f64, temperature: f64) -> f64 {
    if new_energy < old_energy {
        1.0
    } else {
        f64::exp((old_energy - new_energy) / temperature)
    }
}
/// # Example
/// objective function: f(x,y) = x^2 + y^2
/// ```
/// use optimax::{stochastic_annealing, Solution};
///
/// let objective_function = |x: &[f64]| x[0].powi(2) + x[1].powi(2);
/// let initial_solution = Solution::new(vec![1.0, 1.0], objective_function(&[1.0, 1.0]));
/// let final_solution = stochastic_annealing(&objective_function, initial_solution, 100.0, 0.75);
/// println!("Optimal Point: {:?}", final_solution.coords);
/// println!("Optimal Value: {}", final_solution.value);
/// ```
pub fn stochastic_annealing(
    objective_function: &dyn Fn(&[f64]) -> f64,
    mut current_solution: Solution,
    mut temperature: f64,
    cooling_rate: f64,
) -> Solution {
    let mut rng = rand::thread_rng();

    while temperature > 0.1 {
        let new_solution = Solution::new(
            current_solution
                .coords
                .iter()
                .map(|&x| x + temperature.powf(-1.0 / 10.0) * rng.gen_range(-1.0..=1.0))
                .collect(),
            objective_function(&current_solution.coords),
        );

        let acceptance_prob =
            acceptance_probability(current_solution.value, new_solution.value, temperature);

        if acceptance_prob > rng.gen::<f64>() {
            current_solution = new_solution;
        }

        temperature *= 1.0 - cooling_rate;
    }

    current_solution
}
