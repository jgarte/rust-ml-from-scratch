use rand::{ seq::SliceRandom };

pub fn _scaler(vector: &Vec<f64>) -> Vec<f64> {
    // Scale input values between 0 and 1

    let mut result: Vec<f64> = Vec::new();
    let max_int = vector.iter().copied().fold(f64::NEG_INFINITY,
                                              f64::max);
    let min_int = vector.iter().copied().fold(f64::INFINITY,
                                              f64::min);
    let offset = 0.0 - min_int;
    let max_offset = max_int + offset;
    for i in vector {
        let new_value: f64 = (i + offset) / max_offset;
        result.push(new_value)
    }
    result
}

pub fn _train_test_shuffle_split(ratio: f64) -> (Vec<i32>,
                                                 Vec<i32>,
                                                 Vec<i32>,
                                                 Vec<i32>) {
    // Shuffle X and y and split in to train and test sets
    let x = vec![1i32, 2, 3, 4, 5];
    let y = vec![1i32, 2, 3, 4, 5];


    // create shuffled indices
    let mut rng = rand::thread_rng();
    let len_x = x.len();
    let mut indices: Vec<usize> = Vec::new();
    for idx in 0..len_x {
        indices.push(idx)
    }
    indices.shuffle(&mut rng);

    // shuffle x and y
    let mut x_shuffled: Vec<i32> = Vec::new();
    let mut y_shuffled: Vec<i32> = Vec::new();
    for i in indices {
        x_shuffled.push(x[i]);
        y_shuffled.push(y[i]);
    }

    // split into train and test sets
    let num_test = len_x as f64 * ratio;
    let num_test = num_test as usize;
    let x_train = &x_shuffled[num_test..];
    let x_test = &x_shuffled[..num_test];
    let y_train = &y_shuffled[num_test..];
    let y_test = &y_shuffled[..num_test];
    (x_train.to_vec(), x_test.to_vec(), y_train.to_vec(), y_test.to_vec())
}

pub fn make_data() -> (Vec<Vec<i64>>, Vec<i64>) {
    // Create sample data

    // Initialize Data
    let x_1 = vec![1i64,1];
    let x_2 = vec![1i64,0];
    let x_3 = vec![0i64,1];
    let x_4 = vec![0i64,0];

    let mut x_values: Vec<Vec<i64>> = Vec::new();
    x_values.push(x_1);
    x_values.push(x_2);
    x_values.push(x_3);
    x_values.push(x_4);

    // let mut y_values: Vec<i64 > = Vec::new();
    let y_values = vec![1i64, 1, 0, 0];
    (x_values, y_values)
}

pub fn accuracy_score(y_test: &Vec<i64>, y_predictions: &Vec<i64>) -> f64 {
    // Compute the ratio of correct predictions
    let len_values = y_test.len();
    let mut num_correct: i64 = 0;
    for i in 0..len_values {
        if y_test[i] == y_predictions[i] {
            num_correct += 1
        }
    }
    let score: f64 = num_correct as f64 / len_values as f64;
    score
}
