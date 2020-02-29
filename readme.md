Depreciated version of nailbot that trains in a gym environment.

End goal of this project is to get two robotic arms to use a nail gun to nail together two planks and place them in a specific position that they can only achieve if they are nailed together

Currently, there is just one robotic arm trying to pick up and move a cube 90 degrees to the right.

Run nailbot\_task.py to train the model Learning\_ddpg.png shows the graph of timesteps vs. reward function ddpg\_nailbot.zip is where the model is saved view\_model.py will show the current iteration of the model

nail\_bot/envs/nailbot\_env.py is the gym environment that the simulation is using. The observation space is the position and orientation of the cube, and each of the segments of the kuka arm/gripper. The action space is the position of each of the joints of the kuka arm/gripper (rotation or translation depending on the joint) The reward function is 1/distance from the position [0,0.5], the cube starts at [0.5,0]

Note for later-Reward function should take into account: How close the planks are to the specified position Are they nailed together Are the arms touching them (negative points if they make contact or are close to it) Can it support itself Punished for pulling the nailgun more than is necessary

