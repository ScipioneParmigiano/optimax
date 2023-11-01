//! ```rustonomicon-optima``` is a collection of numerical methods to find minima/maxima of a function.

pub mod gradient_descent;
pub mod stochastic_annealing;

pub use gradient_descent::gradient_descent;
pub use stochastic_annealing::{stochastic_annealing, Solution};
