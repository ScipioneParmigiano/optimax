use ndarray::{Array1, Array2};
use ndarray_linalg::{Solve, Norm};

fn compute_gradient<F>(
    f: F,
    x: &Array1<f64>,
    epsilon: f64,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let mut grad = Array1::zeros(x.len());

    for i in 0..x.len() {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();

        x_plus[i] += epsilon;
        x_minus[i] -= epsilon;

        let f_plus = f(&x_plus);
        let f_minus = f(&x_minus);

        grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
    }

    grad
}

pub fn bfgs<F>(
    f: F,
    initial_values: Array1<f64>,
    tolerance: f64,
    max_iterations: usize,
) -> Array1<f64>
where
    F: Fn(&Array1<f64>) -> f64,
{
    let mut parameters = initial_values.clone();
    let mut hessian_approximation = Array2::eye(parameters.len());
    let mut gradient = compute_gradient(&f, &parameters, 1e-5);
    let mut iter = 0;

    while iter < max_iterations {
        let direction = solve_bfgs(&hessian_approximation, &gradient);
        let alpha = line_search(&f, &parameters, &direction);
        let prev_parameters = parameters.clone();
        parameters += &(alpha * &direction);

        let prev_gradient = gradient.clone();
        gradient = compute_gradient(&f, &parameters, 1e-5);

        let y = &gradient - &prev_gradient;
        let s = &parameters - &prev_parameters;

        let rho = 1.0 / y.dot(&s);
        let a = Array2::eye(parameters.len()) - rho * &s.dot(&y.t());
        let b = Array2::eye(parameters.len()) - rho * &y.dot(&s.t());

        hessian_approximation = &a.dot(&hessian_approximation.dot(&b)) + rho * &s.dot(&s.t());

        if norm(&gradient) < tolerance {
            break;
        }

        iter += 1;
    }

    parameters
}

fn solve_bfgs(
    hessian: &Array2<f64>,
    gradient: &Array1<f64>,
) -> Array1<f64> {
    hessian.solve(gradient).expect("Failed to solve linear system")
}
fn line_search(
    f: &dyn Fn(&Array1<f64>) -> f64,
    x_k: &Array1<f64>,
    direction: &Array1<f64>,
) -> f64 {

    let alpha = 0.01; 
    let new_x = x_k + &(alpha * direction); 

    let f_new = f(&new_x);

    println!("New Position: {:?}", new_x);
    println!("Objective Function Value at New Position: {}", f_new);

    alpha 
}

fn norm(array: &Array1<f64>) -> f64 {
    array.norm_l2()
}