// Import Eigen library for linear algebra operations
#include <Eigen/Dense>

// Import MPC header file
#include "mpc.h"

// Define MPC constructor
MPC::MPC(Eigen::VectorXd (*model)(const Eigen::VectorXd&, const Eigen::VectorXd&), double (*cost_func)(const Eigen::VectorXd&, const Eigen::VectorXd&), int horizon, double dt) {
    // Assign model, cost_func, horizon, and dt to attributes
    this->model = model;
    this->cost_func = cost_func;
    this->horizon = horizon;
    this->dt = dt;
}

// Define MPC solve method
Eigen::VectorXd MPC::solve(const Eigen::VectorXd& state) {
    // Initialize some variables
    int n = state.size(); // The dimension of the state vector
    int m = 4; // The dimension of the action vector (assuming 4 control commands)
    Eigen::VectorXd action(m); // The optimal action vector to return
    double min_cost = std::numeric_limits<double>::max(); // The minimum cost to minimize

    // Define some constants and parameters for the optimization technique
    const int MAX_ITER = 100; // The maximum number of iterations
    const double ALPHA = 0.01; // The learning rate
    const double EPSILON = 0.001; // The convergence criterion

    // Initialize a random action vector as the initial guess
    Eigen::VectorXd x(m);
    x.setRandom();

    // Loop until convergence or maximum iterations
    for (int i = 0; i < MAX_ITER; i++) {
        // Calculate the cost and gradient of the current action vector
        double cost = 0;
        Eigen::VectorXd grad(m);
        grad.setZero();

        // Loop over the horizon
        Eigen::VectorXd s = state; // The current state
        for (int j = 0; j < horizon; j++) {
            // Predict the next state using the model
            Eigen::VectorXd s_next = model(s, x);

            // Update the cost using the cost function
            cost += cost_func(s, x);

            // Update the gradient using finite differences
            for (int k = 0; k < m; k++) {
                Eigen::VectorXd x_plus = x;
                Eigen::VectorXd x_minus = x;
                x_plus(k) += EPSILON;
                x_minus(k) -= EPSILON;
                grad(k) += (cost_func(s, x_plus) - cost_func(s, x_minus)) / (2 * EPSILON);
            }

            // Update the state for the next iteration
            s = s_next;
        }

        // Update the action vector using gradient descent
        x -= ALPHA * grad;

        // Check if the cost is lower than the minimum cost
        if (cost < min_cost) {
            // Update the optimal action and minimum cost
            action = x;
            min_cost = cost;
        }

        // Check if the gradient norm is smaller than the convergence criterion
        if (grad.norm() < EPSILON) {
            // Break the loop
            break;
        }
    }

    return action;
}

// Define MPC save method
void MPC::save(const std::string& file_name) {
    // Open a file stream for writing
    std::ofstream file(file_name);

    // Check if the file is opened successfully
    if (file.is_open()) {
        // Write the model, cost_func, horizon, and dt to the file
        file << model << std::endl;
        file << cost_func << std::endl;
        file << horizon << std::endl;
        file << dt << std::endl;

        // Close the file stream
        file.close();
    } else {
        // Print an error message
        std::cerr << "Error: Unable to open file " << file_name << std::endl;
    }
}

// Define MPC load method
void MPC::load(const std::string& file_name) {
    // Open a file stream for reading
    std::ifstream file(file_name);

    // Check if the file is opened successfully
    if (file.is_open()) {
        // Read the model, cost_func, horizon, and dt from the file
        file >> model;
        file >> cost_func;
        file >> horizon;
        file >> dt;

        // Close the file stream
        file.close();
    } else {
        // Print an error message
        std::cerr << "Error: Unable to open file " << file_name << std::endl;
    }
}

