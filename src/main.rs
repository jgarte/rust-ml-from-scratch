mod processing;
mod perceptron;

// A simple Perceptron in Rust

fn main() {
    let x_new = vec![1i64, 0];
    let (x_train, y_train) = processing::make_data();

    // train and predict a model
    let mut model = perceptron::Perceptron::perceptron(1.0, 1.0, 50);
    model.fit(&x_train, &y_train);
    let y_prediction = model.predict(&x_new);
    println!("Prediction for {:?}: {}", x_new, y_prediction);

    // check accuracy score
    let y_pred: Vec<i64> = vec![1i64, 2, 3, 4, 5, 8];
    let y_test: Vec<i64> = vec![1i64, 2, 3, 4, 6, 7];
    let score = processing::accuracy_score(&y_test, &y_pred);
    println!("Accuracy Score: {}", score)
}
