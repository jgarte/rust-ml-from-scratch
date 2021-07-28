use rand::{ Rng };

pub struct Perceptron {
    weights: Vec<f64>,
    lr: f64,
    bias: f64,
    epochs: i64,
}

impl Perceptron {
    pub fn perceptron(learning_rate: f64,
                      bias_value: f64,
                      num_epochs: i64) -> Self {
        Perceptron {
            weights: vec![],
            lr: learning_rate,
            bias: bias_value,
            epochs: num_epochs,
        }
    }

    pub fn fit(&mut self,
           x: &Vec<Vec<i64>>,
           y: &Vec<i64>) {
        let len_x_values = x.len();
        assert!((len_x_values == y.len()),
                "Length of X and y must match.");

        // initialize weights
        if self.weights.len() == 0 {
            let mut rng = rand::thread_rng();
            for _ in 0..len_x_values {
                self.weights.push(rng.gen_range(0.0..1.0))
            }
        }

        // fit
        let mut epoch_number: i64 = 1;
        let mut num_errors: i64 = 0;
        let mut total_predictions: i64 = 0;
        for _ in 0..self.epochs {
            for idx in 0..len_x_values {
                let x = &x[idx];
                let y = &y[idx];
                let x_len = x.len();
                let mut value: f64 = 0.0;
                for idx in 0..x_len {
                    value += x[idx] as f64 * self.weights[idx]
                }
                value += self.bias * self.weights[x_len];
                let prediction: i64;
                if value > 0.0 {
                    prediction = 1
                } else {
                    prediction = 0
                }
                let error: f64 = *y as f64 - prediction as f64;
                for idx in 0..x_len {
                    self.weights[idx] += error * x[idx] as f64 * self.lr
                }
                self.weights[x_len] += error * self.bias * self.lr;
                if error != 0.0 {
                    num_errors += 1
                }
                total_predictions += 1;
            }
            let error_rate = (total_predictions as f64 - num_errors as f64)
                / total_predictions as f64;
            println!("Epoch: {0: <8} | Accuracy: {1: <8}",
                     epoch_number,
                     error_rate);
            epoch_number += 1;
        }
    }

    pub fn predict(self, x: &Vec<i64>) -> i64 {
        let x_len = x.len();
        let mut value: f64 = 0.0;
        for n in 0..x_len {
            value += x[n] as f64 * self.weights[n]
        }
        value += self.bias * self.weights[x_len];
        let prediction: i64;
        if value > 0.0 {
            prediction = 1
        } else {
            prediction = 0
        }
        prediction
    }
}
