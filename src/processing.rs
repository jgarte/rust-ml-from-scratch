pub mod io {
    use std::fs::File;

    pub fn read_csv(file_path: String) -> (Vec<Vec<f64>>, Vec<f64>) {
        // Reads a csv and converts to X and y vectors. The X vector
        // contains vectors of each row of the csv up to the last
        // column. The last column is used to construct the y vector.

        let file = File::open(file_path).expect("File not found!");
        let mut reader = csv::Reader::from_reader(file);
        let mut data: Vec<Vec<f64>> = Vec::new();
        let mut target: Vec<f64> = Vec::new();

        for record in reader.records().into_iter() {
            let mut vector: Vec<f64> = Vec::new();
            let current_record = &record.unwrap();
            let len_record = current_record.len();
            for idx in 0..len_record {
                let s = &current_record[idx].parse().unwrap();
                if idx < (len_record - 1) {
                    vector.push(*s)
                } else {
                    target.push(*s)
                }
            }
            data.push(vector)
        }
        (data, target)
    }
}

pub mod utilities {
    use rand::seq::SliceRandom;

    pub fn scaler(vector: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Scale input values between 0 and 1

        let mut final_result: Vec<Vec<f64>> = Vec::new();

        // find the smallest and largest value
        let mut all_mins: Vec<f64> = Vec::new();
        let mut all_maxes: Vec<f64> = Vec::new();

        // start with max value of each inner vector
        for v in vector {
            let max_int = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let min_int = v.iter().copied().fold(f64::INFINITY, f64::min);
            all_mins.push(min_int);
            all_maxes.push(max_int);
        }

        // find max of the maxes of each inner vector
        let max_int = all_maxes.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_int = all_mins.iter().copied().fold(f64::INFINITY, f64::min);

        // scale the data around the min and max as 0, 1
        for v in vector {
            let mut result: Vec<f64> = Vec::new();
            let offset = 0.0 - min_int;
            let max_offset = max_int + offset;
            for i in v {
                let new_value: f64 = (i + offset) / max_offset;
                result.push(new_value)
            }
            final_result.push(result);
        }
        final_result
    }

    pub fn std<T: Into<f64> + Copy>(x: &Vec<T>) -> f64 {
        // standard deviation
        let mean_x = mean(x);
        let len_x = x.len();
        let mut numerator: f64 = 0.0;
        for &value in x {
            numerator += (value.into() - mean_x).powf(2.0);
        }
        (numerator / len_x as f64).sqrt()
    }

    pub fn mean<T: Into<f64> + Copy>(vector: &Vec<T>) -> f64 {
        // mean
        let mut sum: f64 = 0.0;
        for &n in vector {
            sum += n.into()
        }
        sum / vector.len() as f64
    }

    pub fn train_test_split(
        x: &Vec<Vec<f64>>,
        y: &Vec<f64>,
        ratio: f64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        // Shuffle X and y and split in to train/test sets.

        // create shuffled indices
        let mut rng = rand::thread_rng();
        let len_x = x.len();
        let mut indices: Vec<usize> = Vec::new();
        for idx in 0..len_x {
            indices.push(idx)
        }
        indices.shuffle(&mut rng);
        assert!(x.len() == y.len());

        // shuffle x and y
        let mut x_shuffled: Vec<Vec<f64>> = Vec::new();
        let mut y_shuffled: Vec<f64> = Vec::new();
        for i in indices {
            x_shuffled.push(x[i].clone());
            y_shuffled.push(y[i]);
        }

        // split into train and test sets
        let num_test = len_x as f64 * ratio;
        let num_test = num_test as usize;
        let x_train = &x_shuffled[num_test..];
        let x_test = &x_shuffled[..num_test];
        let y_train = &y_shuffled[num_test..];
        let y_test = &y_shuffled[..num_test];
        (
            x_train.to_vec(),
            x_test.to_vec(),
            y_train.to_vec(),
            y_test.to_vec(),
        )
    }

    pub fn norm<T: Into<f64> + Copy>(vector: &Vec<T>) -> f64 {
        // norm of a vector
        let mut sum: f64 = 0.0;
        for &value in vector {
            sum += value.into() * value.into()
        }
        sum.sqrt()
    }
}

pub mod metrics {
    // pub fn r2_score() -> f64 {

    pub fn accuracy_score<T: Into<f64> + Copy>(y: &Vec<T>, y_pred: &Vec<T>) -> f64 {
        // compute the ratio of correct predictions
        let len_values = y.len();
        let mut num_correct: f64 = 1.0;
        for i in 0..len_values {
            if (y[i].into() - y_pred[i].into()).abs() == 0.0 {
                num_correct += 1.0
            }
        }
        let score: f64 = num_correct / len_values as f64;
        score
    }

    pub fn mse<T: Into<f64> + Copy>(y: &Vec<T>, y_pred: &Vec<T>) -> f64 {
        // mean squared error
        assert!(
            (y.len() == y_pred.len()),
            "Lengths of y and y_pred do not match"
        );
        let len_data = y.len();
        let mut numerator: f64 = 0.0;
        for idx in 0..len_data {
            numerator += (y[idx].into() - y_pred[idx].into()).powf(2.0)
        }
        numerator / len_data as f64
    }

    pub fn rmse<T: Into<f64> + Copy>(y: &Vec<T>, y_pred: &Vec<T>) -> f64 {
        // root mean squared error
        let mse: f64 = mse(y, y_pred);
        mse.sqrt()
    }
}
