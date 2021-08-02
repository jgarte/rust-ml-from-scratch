use rand::Rng;

pub struct Perceptron {
    weights: Vec<f64>,
    lr: f64,
    bias: f64,
    epochs: i32,
}

impl Perceptron {
    pub fn perceptron<T: Into<f64> + Copy>(
        learning_rate: T,
        bias_value: T,
        num_epochs: i32,
    ) -> Self {
        Perceptron {
            weights: vec![],
            lr: learning_rate.into(),
            bias: bias_value.into(),
            epochs: num_epochs,
        }
    }

    pub fn fit<T: Into<f64> + Copy>(&mut self, x: &Vec<Vec<T>>, y: &Vec<T>) {
        let len_x_values = x.len();
        assert!(
            (len_x_values == y.len()),
            "Lengths of X and y do not match."
        );

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
                let y: &f64 = &y[idx].into();
                let x_len = x.len();
                let mut value: f64 = 0.0;
                for idx in 0..x_len {
                    value += x[idx].into() as f64 * self.weights[idx]
                }
                value += self.bias * self.weights[x_len];
                let prediction: i64;
                if value > 0.0 {
                    prediction = 1
                } else {
                    prediction = 0
                }
                let error: f64 = y - prediction as f64;
                for idx in 0..x_len {
                    self.weights[idx] += error * x[idx].into() * self.lr
                }
                self.weights[x_len] += error * self.bias * self.lr;
                if error != 0.0 {
                    num_errors += 1
                }
                total_predictions += 1;
            }
            let error_rate =
                (total_predictions as f64 - num_errors as f64) / total_predictions as f64;
            println!(
                "Epoch: {0: <8} | Accuracy: {1: <8}",
                epoch_number, error_rate
            );
            epoch_number += 1;
        }
    }

    pub fn predict<T: Into<f64> + Copy>(self, x: &[Vec<T>]) -> Vec<f64> {
        // predict values of x; returns vector of predictions

        let mut predictions: Vec<f64> = Vec::new();
        for v in x {
            let x_len = v.len();
            let mut value: f64 = 0.0;
            for (n, &input) in v.iter().enumerate().take(x_len) {
                value += input.into() * self.weights[n]
            }
            value += self.bias * self.weights[x_len];
            let prediction: f64;
            if value > 0.0 {
                prediction = 1.0
            } else {
                prediction = 0.0
            }
            predictions.push(prediction)
        }
        predictions
    }
}
