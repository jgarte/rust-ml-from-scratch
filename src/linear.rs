pub use crate::processing::utilities;

pub struct LinearRegressor {
    pub slope: f64,
    pub intercept: f64,
}

impl LinearRegressor {
    pub fn linear_regressor() -> Self {
        LinearRegressor {
            slope: 0.0,
            intercept: 0.0,
        }
    }

    pub fn fit<T: Into<f64> + Copy>(&mut self, x: &Vec<T>, y: &Vec<T>) {
        // uses the least squares method to find the slope and intercept
        assert!(x.len() == y.len(), "Lenghts of X and y do not match");
        let len_data = x.len();
        let x_mean = utilities::mean(x);
        let y_mean = utilities::mean(y);
        let mut correlation: f64 = 0.0;
        let mut standard_dev: f64 = 0.0;
        for idx in 0..len_data {
            correlation += (x[idx].into() - x_mean) * (y[idx].into() - y_mean);
            standard_dev += (x[idx].into() - x_mean).powf(2.0);
        }
        self.slope = &correlation / &standard_dev;
        self.intercept = y_mean - (self.slope * x_mean);
    }

    pub fn predict<T: Into<f64> + Copy>(self, x: &Vec<T>) -> Vec<f64> {
        let mut predictions: Vec<f64> = Vec::new();
        for &value in x {
            predictions.push(self.slope * value.into() + self.intercept)
        }
        predictions
    }
}
