// Import Eigen library for linear algebra operations
#include <Eigen/Dense>

// Import MADRaS library for multi-agent simulation
#include "madras.h"

// Define Agent class
class Agent {
public:
    // Constructor
    Agent(Eigen::VectorXd (*reward_func)(const Eigen::VectorXd&), bool (*termination_func)(const Eigen::VectorXd&), Eigen::VectorXd (*model)(const Eigen::VectorXd&, const Eigen::VectorXd&), const std::string& name);

private:
    // Observation space attribute
    gympp::Space observation_space;

    // Action space attribute
    gympp::Space action_space;

    // Reward function attribute
    Eigen::VectorXd (*reward_func)(const Eigen::VectorXd&);

    // Termination function attribute
    bool (*termination_func)(const Eigen::VectorXd&);

    // Model attribute
    Eigen::VectorXd (*model)(const Eigen::VectorXd&, const Eigen::VectorXd&);

    // State attribute
    Eigen::VectorXd state;

    // Name attribute
    std::string name;
};

// Define MADRaS class
class MADRaS {
public:
    // Constructor
    MADRaS(int num_agents, const std::vector<Agent>& agents);

    // Reset method
    std::vector<Eigen::VectorXd> reset();

    // Step method
    std::tuple<std::vector<Eigen::VectorXd>, std::vector<double>, std::vector<bool>> step(const std::vector<Eigen::VectorXd>& actions);

    // Render method
    void render();

    // Close method
    void close();

private:
    // Number of agents attribute
    int num_agents;

    // Agents attribute
    std::vector<Agent> agents;

    // Environment attribute
    MadrasEnv* env;
};
