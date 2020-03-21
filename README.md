# Laelaps2_DDPG
Laelaps II training for slope handling using Gazebo, ROS and Gym. Work by A. S. Mastrogeorgiou and Yehia S. Elbahrawy.

Laelaps II is a quadruped robot built by the Legged Team at the Control Systems Lab of NTUA
(http://nereus.mech.ntua.gr/legged/)

This work focuses on how from trajectory planning on toe level and open loop stability as a starting point, it is possible to
develop a controller using deep RL that enables Laelaps II quadruped to handle positive and negative slopes.

### Steps to train Laelaps II using DDPG

I) Spawn Laelaps II in Gazebo and choose one of the following training tasks.

1) Inclination +10°.

        roslaunch laelaps_gazebo laelaps_world_upRamp.launch

2) Inclination -10°.

        roslaunch laelaps_gazebo laelaps_world_downRamp.launch

3) Level terrain:

        roslaunch laelaps_gazebo laelaps_world_flat.launch 

II) Launch the training environment and algorithm:

    roslaunch training_system training_ddpg_RL_Ellipse_gaits.launch

### Steps to test the generalization of a pre-trained saved model

I) Spawn Laelaps II in Gazebo and choose one of the following training tasks depending on the trained model.

1) Inclination +15°.

        roslaunch laelaps_gazebo laelaps_world_upRamp_15degrees.launch

2) Inclination -15°.

        roslaunch laelaps_gazebo laelaps_world_downRamp_15degrees.launch

3) Inclination +10° with longer ramp.

    	roslaunch laelaps_gazebo laelaps_world_upRamp_4ramps.launch

4) Inclination -10° with longer ramp.

    	roslaunch laelaps_gazebo laelaps_world_downRamp_4ramps.launch 

II) Launch pre-trained saved model.

1) Choose a saved model from the runs folder in the training_system package, where you can find your training results. Inside the training_results folder, the saved models are in the Model_best_test_rewards folder.

2) Run the test node.

		roslaunch training_system test_pretrained_model.launch

#### Note
In training and testing change the ERP value to 0.5 in Gazebo at the beginning after launching step II. The ERP value can be found in the left coloumn in World settings then Physics then constraints then ERP.

### References

[1] Maxim Lapan., Deep Reinforcement Learning Hands-On: Apply modern RL methods, with deep Q-networks, value iteration, policy gradients, TRPO, AlphaGo Zero and more. Packt Publishing, 2018.
