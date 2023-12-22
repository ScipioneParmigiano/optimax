//! ```rustonomicon-optima``` is a collection of numerical methods to find minima/maxima of a function.

pub mod gradient_descent;
pub mod stochastic_annealing;
pub mod bfgs; 

use bfgs::*;

fn main() {
    // Example usage:
    let initial_values = ndarray::Array1::from_vec(vec![1.0, 2.0]);
    let result = bfgs(
        &|x: &ndarray::Array1<f64>| x[0].powi(2) + x[1].powi(2), // Objective function f(x)
        initial_values,
        1e-6, // Tolerance
        100,  // Maximum iterations
    );
    println!("Optimal parameters: {:?}", result);
}
