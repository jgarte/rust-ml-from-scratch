# Maching Learning from Scratch in Rust 

This was my first foray in to the Rust language. The project consists of several machine learning algorithms and utilities written from scratch. The API is modeled after the popular Scikit-Learn library, with the fit() and predict() methods. I no longer work on this project but it was a fun and valuable learning experience.

## Modules

**Learning algorithms**
* Perceptron (`perceptron.rs`)
* Linear Regression (`linear.rs`)
* Linear Regression with stochastic gradient descent (`sgd.rs`)

**Utilities (`processing.rs`)**
* Accuracy Score 
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* Rudimentary CSV reader
* Scaler - scale input values between 0 and 1
* Norm
* Standard Deviation
* Mean
* Train test split  - shuffle data and split into train/test 
