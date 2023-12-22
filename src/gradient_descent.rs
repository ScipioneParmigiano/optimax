///! Module containing the implementation of the gradient descent method
use finitediff::FiniteDiff;

/// # Example
/// objective function: f(x,y) = x^2 + y^2 +3x^2 y^3
/// ```
/// use optimax::{gradient_descent};
///
/// let objective_fn = |params: &Vec<f64>| {
/// let x = params[0];
/// let y = params[1];
/// x.powi(2) + y.powi(2) + 3.0 * x.powi(2) * y.powi(3)};
///
/// let initial_values = vec![1.0, 1.0];
/// let learning_rate = 0.1;
/// let tolerance = 1e-6;
///
/// let optimized_parameters = gradient_descent(
/// &objective_fn,
/// initial_values,
/// learning_rate,
/// tolerance,
/// 10000,
/// );
///
/// let x_optimal = optimized_parameters[0];
/// let y_optimal = optimized_parameters[1];
/// let z_optimal = objective_fn(&vec![x_optimal, y_optimal]);
/// println!("Optimal x: {}", x_optimal);
/// println!("Optimal y: {}", y_optimal);
/// println!("Optimal z: {}", z_optimal);
/// ```
pub fn gradient_descent(
    f: &dyn Fn(&Vec<f64>) -> f64,
    initial_values: Vec<f64>,
    learning_rate: f64,
    tolerance: f64,
    max_iterations: i64,
) -> Vec<f64> {
    let mut parameters = initial_values.clone();
    let mut iter = 0;

    while iter < max_iterations {
        iter += 1;

        // let value = f(&parameters);
        let gradient = parameters.forward_diff(&f); //gradient computation

        let delta_magnitude: f64 = gradient.iter().map(|&grad| grad * grad).sum::<f64>().sqrt();

        // Check for convergence based on the tolerance value
        if delta_magnitude < tolerance {
            break;
        }

        // Update parameters using gradients and learning rate
        for i in 0..parameters.len() {
            parameters[i] -= learning_rate * gradient[i];
        }
    }

    return parameters;
}
