#  Personal work
Q-Learning, DQN, and REINFORCE RL algorithms are implemented to solve this problem.

# wind_turbine_env
Working environment to simulate the power output of a wind turbine given a simulation of the wind

## Packages requirements
numpy, scipy and matlplotlib

## Environment

### Wind
The wind has a short term variation from the Ornstein-Uhlenbeck and a long term variation with a period of 24h. The short term variation
are here to simulate events like wind gusts whereas the long term variation corresponds to diurnal cycles.

### Wind turbine
The wind turbine corresponds to a Vestas V80 machine. It outputs power for a given wind and wind angle. Its sensor reading is not perfect, it has a little
constant bias.

The turbine heading can be moved clockwise, trigo or remain the same. Each action that modifies the angle costs some power that penalizes the output.

### Basic agent
A very simple agent that will tell the wind turbine to follow the wind when their relative angle becomes too big

### Simu
It allows to glue together the different structures and to run a simulation for a given duration. An exemple of how this can work together
is given in the demo.py file
