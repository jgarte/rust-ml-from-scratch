use ndarray::{arr1,arr2,Array2};
use rand::{thread_rng, Rng};
use rand_distr::{StandardNormal};

pub struct SGDRegressor {
    pub theta: Array2<f64>,
    t0: f64,
    t1: f64,
}

impl SGDRegressor {
    pub fn sgd_regressor<T: Into<f64> + Copy>(t0: T, t1: T) -> Self {
        // initialize params
        let mut rng = thread_rng();
        let sn = StandardNormal;
        let theta = arr2(&[[rng.sample(sn)],
                           [rng.sample(sn)]]);
        SGDRegressor {
            theta: theta,
            t0: t0.into(),
            t1: t1.into(),
        }
    }

    pub fn fit<T: Into<f64> + Copy>(&mut self, x: &Vec<T>, y: &Vec<T>, n_epochs: i32) {
        let m = x.len();
        assert!((m == y.len()), "X and y must have the same length.");
        let mut rng = thread_rng();
        for epoch in 0..n_epochs {
            for i in 0..m {
                let random_index = rng.gen_range(0..m);
                let xi = arr2(&[[1.0, x[random_index].into()]]); // sample x
                let yi = arr1(&[y[random_index].into()]); // sample y
                let inner_dot = xi.dot(&self.theta) - yi;
                let gradients = 2.0 * xi.t().dot(&inner_dot);
                let t: f64 = epoch as f64 * m as f64 + i as f64;
                // let eta = Self::learning_schedule(self, t);
                let eta: f64 = self.t0 / (t + self.t1);
                self.theta = &self.theta - eta * gradients;
            }
        }
    }

    pub fn predict<T: Into<f64> + Copy>(self, x: &Vec<T>) -> Vec<f64> {
        let mut predictions: Vec<f64> = Vec::new();
        let intercept = self.theta[[0,0]];
        let slope = self.theta[[1,0]];
        for &value in x {
            predictions.push(slope * value.into() + intercept)
        }
        predictions
    }
}
