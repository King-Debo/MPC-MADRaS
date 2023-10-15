# MPC-MADRaS: A Multi-Agent Reinforcement Learning Program Using Model Predictive Control and MADRaS Simulator

MPC-MADRaS is a program that implements a multi-agent reinforcement learning (MARL) algorithm using model predictive control (MPC) and MADRaS simulator. The program aims to create and train multiple agents that can control different types of vehicles, such as cars, trucks, buses, and bikes, in a realistic and flexible simulation environment. The program uses MPC as a model-based method that can plan ahead and optimize the actions using a learned or given model of the environment. The program also uses MADRaS as a high-fidelity multi-agent simulator that can model the dynamics and interactions of the agents and the environment.

## Table of Contents

- Installation
- Usage
- Contributing
- License

## Installation

To install and run the program, you need to have the following prerequisites:

- C++ compiler (such as [GCC] or [Clang])
- Eigen library (for linear algebra operations)
- MADRaS library (for multi-agent simulation)
- PyGame or OpenCV library (for rendering the environment)

To install the program, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the root directory of the repository.
3. Run `make` command to compile the source files.
4. Run `./mpc_madras` executable to launch the program.

## Usage

To use the program, you need to configure some parameters and settings for the MPC agent and the MADRaS environment. These parameters and settings are located in the `config.h` file in the `include` directory. You can modify them according to your needs and preferences.

The parameters and settings include:

- The choice of programming language (C++ or Rust)
- The choice of RL model (MPC or MCTS)
- The choice of simulation environment (MADRaS or AirSim)
- The number of agents and their types
- The observation space, action space, reward function, and termination function for each agent
- The model, cost function, horizon, and time step for MPC
- The simulation scenarios, traffic conditions, and terrains for MADRaS

After configuring the parameters and settings, you can run the program again to see the results. The program will create and train multiple agents in the simulation environment using MPC as the RL model. You can also render the environment and display some information about each agent's state using PyGame or OpenCV.

## Contributing

If you want to contribute to this project, you are welcome to do so. Please follow these guidelines:

- Fork this repository and create a new branch for your feature or bug fix.
- Write clear and concise code with comments and documentation.
- Follow the coding style and conventions of this project.
- Test your code thoroughly and ensure it is free of errors and bugs.
- Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License. See the [LICENSE] file for more details.