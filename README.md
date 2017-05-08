# Driving Simulator

We are designing and implementing a clean, elegant driving simulator for deep learning experiments. The simulator provides a diagrammatic top-down view of one or more cars on a track, taking as input car controls and parameters and outputting the next position and velocity for each car so it can be linked with a policy. We will learn a policy using deep learning from demonstrations, evaluate the policy under learning variations, and finally evaluate policies with the real cars.


To use this driving simulator, you will need to install PyGame and OpenAI Gym.

The simulator must be installed before running:

> pip install -e .

To run the sim, execute:

> python run_simulator.py


# Documentation Summary

Full documentation coming soon.

## Environments
driving_env.py: Gym interface for environment, interactions with outside code should happen here. environment.py does most of the heavy lifting; most function calls here are simply passed onto envrionment.py

environment.py: Main environment. Does the heavy lifting for updates.

## Assets

### Cars

All cars have same interface, just different dynamics.

car.py: Point model. 

kinematic_car.py: Kinematic bicycle model.

dynamic_car.py: Dynamic bicycle model with slip and friction.

### Tracks

rectangle.py: Abstract class for rectangles.

terrain.py: Class for terrain.

## Manual Input

controller.py: Handles manual inputs from different controllers (keyboard, Xbox controller) and passes it onto the environment

xboxController.py: Feeds inputs from the Xbox controller into controller.py

## Configuration

generate_config.py: Generates configuration file that the environment reads from. Allows configuration of most features in the simulator, including state space, action space, number of and positions of CPU cars, track placement, etc. 

configs/: Folder for storing configuration files.

The default demo script (run_simulator.py) reads from the configuration script in configs/config.json. To change the config it runs from, use the command 

> python run_simulator.py --config new_config.json



