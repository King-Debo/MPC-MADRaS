// Import Eigen library for linear algebra operations
#include <Eigen/Dense>

// Define MPC class
class MPC {
public:
    // Constructor
    MPC(Eigen::VectorXd (*model)(const Eigen::VectorXd&, const Eigen::VectorXd&), double (*cost_func)(const Eigen::VectorXd&, const Eigen::VectorXd&), int horizon, double dt);

    // Solve method
    Eigen::VectorXd solve(const Eigen::VectorXd& state);

    // Save method
    void save(const std::string& file_name);

    // Load method
    void load(const std::string& file_name);

private:
    // Model attribute
    Eigen::VectorXd (*model)(const Eigen::VectorXd&, const Eigen::VectorXd&);

    // Cost function attribute
    double (*cost_func)(const Eigen::VectorXd&, const Eigen::VectorXd&);

    // Horizon attribute
    int horizon;

    // Time step attribute
    double dt;
};
