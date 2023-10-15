// Import the necessary libraries and headers
#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "mpc.h"
#include "madras.h"

// Initialize the MPC agent and the MADRaS environment
MPC mpc;
MADRaS env;

// Define the observation space, action space, reward function, and termination condition
const int OBS_DIM = 10; // The dimension of the observation vector
const int ACT_DIM = 4; // The dimension of the action vector
const int HORIZON = 20; // The planning horizon for MPC
const double DT = 0.1; // The time step for MPC

double reward_func(const Eigen::VectorXd& obs); // The reward function that takes an observation vector and returns a scalar reward
bool termination_func(const Eigen::VectorXd& obs); // The termination function that takes an observation vector and returns a boolean flag

// Loop over the episodes and steps
int main() {
    int num_episodes = 100; // The number of episodes to run
    int num_steps = 1000; // The maximum number of steps per episode

    for (int i = 0; i < num_episodes; i++) {
        // Reset the environment and get the initial observation
        Eigen::VectorXd obs = env.reset();

        // Initialize the agent's state and cumulative reward
        Eigen::VectorXd state = obs;
        double total_reward = 0;

        for (int j = 0; j < num_steps; j++) {
            // Loop over the horizon
            Eigen::VectorXd action;
            double min_cost = std::numeric_limits<double>::max();

            for (int k = 0; k < HORIZON; k++) {
                // Solve the optimization problem using MPC and get the optimal action
                Eigen::VectorXd act = mpc.solve(state);

                // Apply the action to the environment model and get the next state and cost
                Eigen::VectorXd next_state = env.model(state, act);
                double cost = mpc.cost_func(state, act);

                // Update the optimal action and minimum cost if necessary
                if (cost < min_cost) {
                    action = act;
                    min_cost = cost;
                }

                // Update the state for the next iteration
                state = next_state;
            }

            // Apply the optimal action to the environment and get the next observation, reward, and done flag
            Eigen::VectorXd next_obs;
            double reward;
            bool done;
            std::tie(next_obs, reward, done) = env.step(action);

            // Update the agent's state and cumulative reward
            state = next_obs;
            total_reward += reward;

            // Break the loop if done flag is True
            if (done) {
                break;
            }
        }

        // Print the episode number and total reward
        std::cout << "Episode " << i << ": Total reward = " << total_reward << std::endl;
    }

    // Save the agent's model and parameters
    mpc.save("mpc_model.h5");
}
double reward_func(const Eigen::VectorXd& obs) {
    // Extract the observation features
    double x = obs(0); // The x-coordinate of the drone
    double y = obs(1); // The y-coordinate of the drone
    double z = obs(2); // The z-coordinate of the drone
    double vx = obs(3); // The x-velocity of the drone
    double vy = obs(4); // The y-velocity of the drone
    double vz = obs(5); // The z-velocity of the drone
    double roll = obs(6); // The roll angle of the drone
    double pitch = obs(7); // The pitch angle of the drone
    double yaw = obs(8); // The yaw angle of the drone
    double dist = obs(9); // The distance to the nearest obstacle

    // Define some constants and parameters
    const double MAX_SPEED = 10; // The maximum speed of the drone
    const double MIN_ALT = 5; // The minimum altitude of the drone
    const double MAX_ALT = 20; // The maximum altitude of the drone
    const double SAFE_DIST = 10; // The safe distance to avoid obstacles
    const double GOAL_X = 100; // The x-coordinate of the goal
    const double GOAL_Y = 100; // The y-coordinate of the goal

    const double SPEED_WEIGHT = 1; // The weight for the speed term
    const double ALT_WEIGHT = 1; // The weight for the altitude term
    const double ANGLE_WEIGHT = 1; // The weight for the angle term
    const double DIST_WEIGHT = 1; // The weight for the distance term
    const double GOAL_WEIGHT = 1; // The weight for the goal term

    // Calculate the speed term
    double speed = std::sqrt(vx * vx + vy * vy + vz * vz); // The speed of the drone
    double speed_term = SPEED_WEIGHT * (speed / MAX_SPEED); // The normalized speed term

    // Calculate the altitude term
    double alt_term;
    if (z < MIN_ALT) {
        alt_term = -ALT_WEIGHT * (MIN_ALT - z); // A negative penalty for being too low
    } else if (z > MAX_ALT) {
        alt_term = -ALT_WEIGHT * (z - MAX_ALT); // A negative penalty for being too high
    } else {
        alt_term = ALT_WEIGHT * ((z - MIN_ALT) / (MAX_ALT - MIN_ALT)); // A positive reward for being in the optimal range
    }

    // Calculate the angle term
    double angle_term = -ANGLE_WEIGHT * (std::abs(roll) + std::abs(pitch) + std::abs(yaw)); // A negative penalty for having large angles

    // Calculate the distance term
    double dist_term;
    if (dist < SAFE_DIST) {
        dist_term = -DIST_WEIGHT * (SAFE_DIST - dist); // A negative penalty for being too close to an obstacle
    } else {
        dist_term = DIST_WEIGHT * (dist / SAFE_DIST); // A positive reward for being far from an obstacle
    }

    // Calculate the goal term
    double dx = GOAL_X - x; // The x-distance to the goal
    double dy = GOAL_Y - y; // The y-distance to the goal
    double dxy = std::sqrt(dx * dx + dy * dy); // The euclidean distance to the goal
    double goal_term = -GOAL_WEIGHT * dxy; // A negative penalty for being far from the goal

    // Calculate the total reward as a weighted sum of the terms
    double reward = speed_term + alt_term + angle_term + dist_term + goal_term;

    return reward;
}
bool termination_func(const Eigen::VectorXd& obs) {
    // Extract the observation features
    double x = obs(0); // The x-coordinate of the drone
    double y = obs(1); // The y-coordinate of the drone
    double z = obs(2); // The z-coordinate of the drone
    double vx = obs(3); // The x-velocity of the drone
    double vy = obs(4); // The y-velocity of the drone
    double vz = obs(5); // The z-velocity of the drone
    double roll = obs(6); // The roll angle of the drone
    double pitch = obs(7); // The pitch angle of the drone
    double yaw = obs(8); // The yaw angle of the drone
    double dist = obs(9); // The distance to the nearest obstacle

    // Define some constants and parameters
    const double MAX_SPEED = 10; // The maximum speed of the drone
    const double MIN_ALT = 5; // The minimum altitude of the drone
    const double MAX_ALT = 20; // The maximum altitude of the drone
    const double SAFE_DIST = 10; // The safe distance to avoid obstacles
    const double GOAL_X = 100; // The x-coordinate of the goal
    const double GOAL_Y = 100; // The y-coordinate of the goal
    const double GOAL_TOL = 5; // The tolerance for reaching the goal

    // Calculate the done flag as a logical expression of the features
    bool done;

    if (z <= 0) {
        done = true; // Terminate if the drone hits the ground
    } else if (dist <= 0) {
        done = true; // Terminate if the drone hits an obstacle
    } else if (std::abs(x - GOAL_X) < GOAL_TOL && std::abs(y - GOAL_Y) < GOAL_TOL) {
        done = true; // Terminate if the drone reaches the goal
    } else {
        done = false; // Continue otherwise
    }

    return done;
}
