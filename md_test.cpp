#include <iostream>
#include "tensor/xtensor/xarray.hpp"
#include "tensor/xtensor/xtensor.hpp"
#include "tensor/xtensor/xmath.hpp"
#include "tensor/xtensor/xview.hpp"
#include "tensor/xtensor-blas/xlinalg.hpp"
#include "tensor/xtensor/xio.hpp"

// using namespace std;

// int main() {
//     cout << 1;
//     return 0;
// }

// A simple linear regression: y = X*w + b
// We'll optimize for weights w and bias b

// Mean Squared Error Loss
double mse_loss(const xt::xarray<double>& y_pred, const xt::xarray<double>& y_true) {
    auto diff = y_pred - y_true;
    return xt::mean(xt::square(diff))();
}

int main() {
    // Example training data: 4 samples, 2 features
    xt::xarray<double> X = {{1.0, 2.0},
                              {2.0, 1.0},
                              {3.0, 4.0},
                              {4.0, 3.0}};
    
    // True outputs
    xt::xarray<double> y = {5.0, 4.0, 10.0, 9.0};

    // Initialize parameters (weights and bias)
    xt::xarray<double> w = {0.5, 0.5};  // 2 features
    double b = 0.0;
    
    double learning_rate = 0.01;
    int epochs = 1000;

    // Training loop using gradient descent
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Compute predictions: y_pred = X * w + b
        xt::xarray<double> y_pred = xt::linalg::dot(X, w) + b;

        // Compute error (loss)
        double loss = mse_loss(y_pred, y);

        // Compute gradients (very simplified; note: real implementations need careful derivation)
        // For weights: gradient = (2/N) * X^T * (y_pred - y)
        auto error = y_pred - y;
        xt::xarray<double> grad_w = (2.0 / X.shape()[0]) * xt::linalg::dot(xt::transpose(X), error);
        
        // For bias: gradient = (2/N) * sum(y_pred - y)
        double grad_b = (2.0 / X.shape()[0]) * xt::sum(error)();

        // Update parameters
        w -= learning_rate * grad_w;
        b -= learning_rate * grad_b;

        if (epoch % 100 == 0)
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
    }

    std::cout << "Trained weights: " << w << std::endl;
    std::cout << "Trained bias: " << b << std::endl;

    return 0;
}