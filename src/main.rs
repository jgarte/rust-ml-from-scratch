#![allow(dead_code)]

use rand::{thread_rng, Rng};

mod linear;
mod perceptron;
mod pipeline;
mod processing;
mod sgd;

pub use crate::processing::{io, metrics, utilities};

fn main() {
    let x = vec![1, 2, 3];
    let y = vec![1, 3, 5];
    let error = metrics::rmse(&x, &y);
    println!("{}", error);
    _test_perceptron()
}

fn test_sgd() {
    let mut rng = thread_rng();
    let mut xx: Vec<f64> = Vec::new();
    let mut yy: Vec<f64> = Vec::new();

    for idx in 0..100 {
        xx.push(idx as f64 + rng.gen_range(0.0..1.0));
        yy.push(idx as f64 * 3.0 + 2.0 + rng.gen_range(0.0..1.0));
    }

    let mut model = sgd::SGDRegressor::sgd_regressor(0.01, 80.0);
    model.fit(&xx, &yy, 50);
    let predictions = model.predict(&xx);
    println!("MSE {}", metrics::rmse(&xx, &predictions));
}

fn test_linear_regressor() {
    let mut rng = thread_rng();
    let mut xx: Vec<f64> = Vec::new();
    let mut yy: Vec<f64> = Vec::new();

    for idx in 0..100 {
        xx.push(idx as f64 + rng.gen_range(0.0..1.0));
        yy.push(idx as f64 * 3.0 + 2.0 + rng.gen_range(0.0..1.0));
    }

    let mut lin_reg = linear::LinearRegressor::linear_regressor();
    lin_reg.fit(&xx, &yy);
    let predictions2 = lin_reg.predict(&xx);
    println!("MSE {}", metrics::rmse(&xx, &predictions2));
}

fn _test_perceptron() {
    // load data
    let file_path = "/home/jordan/code/rust-ml/src/data_banknote_authentication.csv".to_string();
    let (x, y) = io::read_csv(file_path);
    let (x_train, x_test, y_train, y_test) = utilities::train_test_split(&x, &y, 0.25);

    // scale
    let x_train_sc = utilities::scaler(&x_train);
    let x_test_sc = utilities::scaler(&x_test);

    // fit
    let mut model = perceptron::Perceptron::perceptron(1.0, 1.0, 100);
    model.fit(&x_train_sc, &y_train);

    // score
    let y_predictions = model.predict(&x_test_sc);
    let score = metrics::accuracy_score(&y_test, &y_predictions);
    println!("Accuracy Score: {}", score);
}
