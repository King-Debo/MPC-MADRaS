// Import Eigen library for linear algebra operations
#include <Eigen/Dense>

// Import MADRaS library for multi-agent simulation
#include "madras.h"

// Define Agent constructor
Agent::Agent(Eigen::VectorXd (*reward_func)(const Eigen::VectorXd&), bool (*termination_func)(const Eigen::VectorXd&), Eigen::VectorXd (*model)(const Eigen::VectorXd&, const Eigen::VectorXd&), const std::string& name) {
    // Assign reward_func, termination_func, model, and name to attributes
    this->reward_func = reward_func;
    this->termination_func = termination_func;
    this->model = model;
    this->name = name;

    // Initialize observation space and action space using [MADRaS API]
    this->observation_space = MadrasEnv::get_observation_space(name);
    this->action_space = MadrasEnv::get_action_space(name);
}

// Define MADRaS constructor
MADRaS::MADRaS(int num_agents, const std::vector<Agent>& agents) {
    // Assign num_agents and agents to attributes
    this->num_agents = num_agents;
    this->agents = agents;

    // Create and initialize env object using [MADRaS API]
    this->env = new MadrasEnv(num_agents);
    this->env->init();
}

// Define MADRaS reset method
std::vector<Eigen::VectorXd> MADRaS::reset() {
    // Initialize a vector of observation vectors to return
    std::vector<Eigen::VectorXd> obs_vec(num_agents);

    // Reset the env object and get the initial observations using [MADRaS API]
    std::vector<std::vector<double>> obs_raw = env->reset();

    // Loop over the agents
    for (int i = 0; i < num_agents; i++) {
        // Convert the raw observation to an Eigen vector
        Eigen::Map<Eigen::VectorXd> obs(obs_raw[i].data(), obs_raw[i].size());

        // Reset the agent's state using its own observation space
        agents[i].state = agents[i].observation_space.project(obs);

        // Assign the agent's state to the observation vector
        obs_vec[i] = agents[i].state;
    }

    return obs_vec;
}

// Define MADRaS step method
std::tuple<std::vector<Eigen::VectorXd>, std::vector<double>, std::vector<bool>> MADRaS::step(const std::vector<Eigen::VectorXd>& actions) {
    // Initialize a vector of next observation vectors, a vector of rewards, and a vector of done flags to return
    std::vector<Eigen::VectorXd> next_obs_vec(num_agents);
    std::vector<double> reward_vec(num_agents);
    std::vector<bool> done_vec(num_agents);

    // Convert the action vectors to raw actions using [MADRaS API]
    std::vector<std::vector<double>> actions_raw(num_agents);
    for (int i = 0; i < num_agents; i++) {
        actions_raw[i] = env->action_space.unproject(actions[i]);
    }

    // Apply the raw actions to the env object and get the next observations, rewards, and done flags using [MADRaS API]
    std::tuple<std::vector<std::vector<double>>, std::vector<double>, std::vector<bool>> result = env->step(actions_raw);
    std::vector<std::vector<double>> next_obs_raw = std::get<0>(result);
    std::vector<double> reward_raw = std::get<1>(result);
    std::vector<bool> done_raw = std::get<2>(result);

    // Loop over the agents
    for (int i = 0; i < num_agents; i++) {
        // Convert the raw next observation to an Eigen vector
        Eigen::Map<Eigen::VectorXd> next_obs(next_obs_raw[i].data(), next_obs_raw[i].size());

        // Update the agent's state using its own model
        agents[i].state = agents[i].model(agents[i].state, actions[i]);

        // Assign the agent's state to the next observation vector
        next_obs_vec[i] = agents[i].state;

        // Calculate the reward using the agent's own reward function
        reward_vec[i] = agents[i].reward_func(agents[i].state);

        // Calculate the done flag using the agent's own termination function
        done_vec[i] = agents[i].termination_func(agents[i].state) || done_raw[i];
    }

    return std::make_tuple(next_obs_vec, reward_vec, done_vec);
}

// Define MADRaS render method
void MADRaS::render() {
    // Render the env object using [PyGame] or [OpenCV]
    env->render();

    // Display some information about each agent's state, such as position, velocity, orientation, etc.
    // Loop over the agents
    for (int i = 0; i < num_agents; i++) {
        // Extract the agent's state features
        double x = agents[i].state(0); // The x-coordinate of the agent
        double y = agents[i].state(1); // The y-coordinate of the agent
        double z = agents[i].state(2); // The z-coordinate of the agent
        double vx = agents[i].state(3); // The x-velocity of the agent
        double vy = agents[i].state(4); // The y-velocity of the agent
        double vz = agents[i].state(5); // The z-velocity of the agent
        double roll = agents[i].state(6); // The roll angle of the agent
        double pitch = agents[i].state(7); // The pitch angle of the agent
        double yaw = agents[i].state(8); // The yaw angle of the agent

        // Create a string to display the information
        std::string info = "Agent " + agents[i].name + ": ";
        info += "Position: (" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ") ";
        info += "Velocity: (" + std::to_string(vx) + ", " + std::to_string(vy) + ", " + std::to_string(vz) + ") ";
        info += "Orientation: (" + std::to_string(roll) + ", " + std::to_string(pitch) + ", " + std::to_string(yaw) + ")";

        // Display the information using [PyGame] or [OpenCV]
        env->display(info);
    }
}

// Define MADRaS close method
void MADRaS::close() {
    // Close and clean up the env object
    env->close();

    // Delete the env object pointer
    delete env;
}
