#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
Deep Determinitic Policy Gradient (DDPG) Network & Agent based on:
1) Max Lapan's implementation in the book Deep Reinforcement Learning Hands-On: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
2) Pytorch 
3) PTAN (PyTorch AgentNet) package for RL https://github.com/Shmuma/ptan
"""

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import rospy
from gazebo_msgs.msg import ModelStates
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
from rosgraph_msgs.msg import Clock
import time
import numpy as np
from numpy import interp
import math
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from laelaps_control.msg import Toe
import logging
import threading
from controllers_connection import ControllersConnection
from gazebo_connection import GazeboConnection
import os
import sys
from tensorboardX import SummaryWriter

print("environment python version", sys.version)

# Global variables:
# epsilon for avoiding whether a number is close to zero
n = math.pi/2
eps = np.finfo(float).eps * 4.0

motor_joints = ["RF_knee",
                "RF_hip",
                "RH_knee",
                "RH_hip",
                "LF_knee",
                "LF_hip",
                "LH_knee",
                "LH_hip"]

laelaps_feet = ["RF_foot",
                "RH_foot",
                "LF_foot",
                "LH_foot"]

laelaps_toe = ["RF_toe",
               "RH_toe",
               "LF_toe",
               "LH_toe"]

controllers_list = ["joint_state_controller", "RF_knee", "RF_hip",
                    "RH_knee", "RH_hip", "LF_knee", "LF_hip", "LH_knee", "LH_hip"]

joint_state_topic = "/laelaps_robot/joint_states"
gazebo_model_states_topic = "/gazebo/model_states"

rf_toe_nan_topic = "/laelaps_robot/RF_toe/nan"
rh_toe_nan_topic = "/laelaps_robot/RH_toe/nan"
lf_toe_nan_topic = "/laelaps_robot/LF_toe/nan"
lh_toe_nan_topic = "/laelaps_robot/LH_toe/nan"

# ________________Defining the initial motor angles and toe_position___________________________________________________________
step_time_out = 1  # 1 second: The network chooses a new action every one second

# init angles are not 0 to avoid legs exploding in simulation and have more stable initial position
init_angles = [0.21644, -0.30604, 0.21644, -
               0.30604, 0.21644, -0.30604, 0.21644, -0.30604]

x_init = 0.0
y_init = -0.58

x_high = 0.1
x_low = -0.1
y_high = -0.50
y_low = -0.55

init_toe_commands = [x_init]*4
init_toe_commands.extend([y_init]*4)

init_phase_shift = [0.0]*4
step_phase_shift = [0.0, math.pi, 0.0, math.pi]
# ________________Global variables___________________________________________________________
action_eps = 0.01
observation_eps = 0.01

joints_states = JointState()  # initializing the JointState msg
toe_class_msg = Toe()  # initializing the Toe msg
sim_time = 0

saving_path = ""
episode_savingpath = ""
episode_start = False
# ________________Register the Laelaps custom environment in Gym___________________________________________________________

register(
    id='LaelapsEnvEllipse-v0',
    entry_point='laelaps_env_ellipse:LaelapsEnvEllipse'
)


class LaelapsEnvEllipse(gym.Env):

    def __init__(self):

        # fixed frame is nearly on the origin of the ground 0,0,0.6 at laelaps body center of mass
        self.last_base_position = [0, 0, 0]
        self.distance_weight = 1
        self.drift_weight = 2
        self.time_step = 0.001  # default Gazebo simulation time step

        self.episode_number = 0

        self.frames = 0

        self.torques_step = []
        self.euler_angles = []
        self.euler_rates = []
        self.base_zaxis = []
        self.base_x_y = []
        self.sim_t = []
        self.saving_option = False

        # Rospy get parameters from config file:
        self.laelaps_model_number = rospy.get_param("/laelaps_model_number")
        self.ramp_model_number = rospy.get_param("/ramp_model_number")
        self.ramp_available = rospy.get_param("/ramp_available")
        self.env_goal = rospy.get_param("/env_goal")

        self.gazebo = GazeboConnection()
        self.controllers_object = ControllersConnection(
            namespace="laelaps_robot", controllers_list=controllers_list)

        # _______________SUBSCRIBERS__________________________________________________

        # give base position and quaternions
        rospy.Subscriber(gazebo_model_states_topic,
                         ModelStates, self.models_callback)
        # give motor angle, velocity, torque
        rospy.Subscriber(joint_state_topic, JointState,
                         self.joints_state_callback)
        rospy.Subscriber("/clock", Clock, self.clock_callback)

        rospy.Subscriber(rf_toe_nan_topic, Bool, self.rf_toe_nan_callback)
        rospy.Subscriber(rh_toe_nan_topic, Bool, self.rh_toe_nan_callback)
        rospy.Subscriber(lf_toe_nan_topic, Bool, self.lf_toe_nan_callback)
        rospy.Subscriber(lh_toe_nan_topic, Bool, self.lh_toe_nan_callback)

        # _______________MOTOR PUBLISHERS__________________________________________________

        self.motor_pub = list()

        for joint in motor_joints:
            joint_controller = "/laelaps_robot/"+str(joint)+"/command"
            x = rospy.Publisher(joint_controller, Float64, queue_size=1)
            self.motor_pub.append(x)
        #

        # _______________Toe PUBLISHERS__________________________________________________

        #  toe4_pos_publisher node : RH_foot: toe1 , RF_foot: toe2, LF_foot: toe3, LH_foot: toe4
        self.toe_pub = list()

        for idx in range(len(laelaps_feet)):
            toe_commands = "/laelaps_robot/toe"+str(idx+1)+"/command"
            x = rospy.Publisher(toe_commands, Toe, queue_size=1)
            self.toe_pub.append(x)

        # ______________Defining observation and action space____________________________

        observation_high = (self.GetObservationUpperBound() + observation_eps)
        observation_low = (self.GetObservationLowerBound() - observation_eps)

        # Four legs toe x,y are estimated by RL
        low = [x_low]*4
        low.extend([y_low]*4)
        high = [x_high]*4
        high.extend([y_high]*4)
        self.action_space = spaces.Box(low=np.array(
            low), high=np.array(high), dtype=np.float32)

        # the epsilon to avoid 0 values entering the neural network in the algorithm
        observation_high = (self.GetObservationUpperBound() + observation_eps)
        observation_low = (self.GetObservationLowerBound() - observation_eps)

        self.observation_space = spaces.Box(
            observation_low, observation_high, dtype=np.float32)

        # ______________Reset and seed the environment____________________________
        self.seed()  # seed the environment in the initial function
        self.init_reset()  # reset the environment in the initial function

    def GetObservationUpperBound(self):
        upper_bound = np.zeros(6)  # 6 observation space dimension

        upper_bound[0:3] = [2 * math.pi] * 3  # roll,pitch yaw
        upper_bound[3:6] = [2 * math.pi / self.time_step] * \
            3  # Roll, pitch, yaw rate

        return upper_bound

    def GetObservationLowerBound(self):
        return -1*self.GetObservationUpperBound()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# _______________call back functions____________________
    def clock_callback(self, msg):
        global sim_time
        sim_time_s = msg.clock.secs
        sim_time_ns = msg.clock.nsecs*1e-9
        sim_time = sim_time_s+sim_time_ns
        if episode_start:
            if self.saving_option:
                self.sim_t.append(sim_time)

    def models_callback(self, msg):
        self.base_pos = msg.pose[self.laelaps_model_number].position
        self.base_orientation = msg.pose[self.laelaps_model_number].orientation
        self.base_velocity = msg.twist[self.laelaps_model_number].linear
        self.base_angular_velocity = msg.twist[self.laelaps_model_number].angular
        if self.ramp_available:
            # both ramps have the same inclination
            self.ramp_inclination = msg.pose[self.ramp_model_number].orientation

        if episode_start:
            if self.saving_option:
                self.base_zaxis.append([self.base_pos.z, self.base_velocity.z])
                self.base_x_y.append([self.base_pos.x, self.base_pos.y])
                euler = self.transfrom_euler_from_quaternion(
                    [self.base_orientation.x, self.base_orientation.y, self.base_orientation.z, self.base_orientation.w])
                self.euler_angles.append(euler)
                self.euler_rates.append(
                    [self.base_angular_velocity.x, self.base_angular_velocity.y, self.base_angular_velocity.z])

    def GetLaelapsBaseInfo(self):
        p = self.base_pos
        q = self.base_orientation
        v = self.base_velocity
        w = self.base_angular_velocity
        base_rot_x = q.x
        base_rot_y = q.y
        base_rot_z = q.z
        base_rot_w = q.w
        base_orientation = [base_rot_x, base_rot_y, base_rot_z, base_rot_w]
        base_p_x = p.x
        base_p_y = p.y
        base_p_z = p.z
        base_pos = [base_p_x, base_p_y, base_p_z]
        base_v_x = v.x
        base_v_y = v.y
        base_v_z = v.z
        base_vel = [base_v_x, base_v_y, base_v_z]
        base_w_x = w.x
        base_w_y = w.y
        base_w_z = w.z
        base_angular_vel = [base_w_x, base_w_y, base_w_z]
        return base_pos, base_orientation, base_vel, base_angular_vel

    def GetRampInclination(self):
        ramp_q = self.ramp_inclination
        ramp_rot_x = ramp_q.x
        ramp_rot_y = ramp_q.y
        ramp_rot_z = ramp_q.z
        ramp_rot_w = ramp_q.w
        ramp_orientation = [ramp_rot_x, ramp_rot_y, ramp_rot_z, ramp_rot_w]
        euler = self.transfrom_euler_from_quaternion(ramp_orientation)
        ramp_inclination = euler[1]  # ramp inclination pitch (around y-axis)
        return ramp_inclination

    def joints_state_callback(self, msg):
        global joints_states
        joints_states = msg

    def GetMotorAngles(self):
        global joints_states
        motor_angles = joints_states.position
        return motor_angles

    def GetMotorVelocities(self):
        global joints_states
        motor_velocities = joints_states.velocity
        return motor_velocities

    def GetMotorTorques(self):
        global joints_states
        motor_torques = joints_states.effort
        return motor_torques
        if episode_start:
            if self.saving_option:
                self.torques_step.append(joints_states.effort)

    def rf_toe_nan_callback(self, msg):
        self.RF_toe_isnan = msg.data

    def rh_toe_nan_callback(self, msg):
        self.RH_toe_isnan = msg.data

    def lf_toe_nan_callback(self, msg):
        self.LF_toe_isnan = msg.data

    def lh_toe_nan_callback(self, msg):
        self.LH_toe_isnan = msg.data

    def GetNanToeCheck(self):
        return [self.RF_toe_isnan, self.RH_toe_isnan, self.LF_toe_isnan, self.LH_toe_isnan]

    # _______________Quaternion to Euler conversion_____________________________________

    def transfrom_euler_from_quaternion(self, quaternion):
        # input quaternion should be list example: [0.06146124, 0, 0, 0.99810947]
        q = np.array(quaternion[:4], dtype=np.float64, copy=True)
        nq = np.dot(q, q)  # gives scalar
        if nq < eps:
            H = np.identity(4)  # identity matrix of 4x4
        q *= math.sqrt(2.0 / nq)
        q = np.outer(q, q)
        H = np.array((
            (1.0-q[1, 1]-q[2, 2],     q[0, 1] -
             q[2, 3],     q[0, 2]+q[1, 3], 0.0),
            (q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
            (q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
            (0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)  # Homogenous transformation matrix

        # Obtain euler angles from Homogenous transformation matrix:
        # Note that many Euler angle triplets can describe one matrix.
        # take only first three rows and first three coloumns = rotation matrix
        M = np.array(H, dtype=np.float64, copy=False)[:3, :3]
        i = 0  # x-axes (first coloumn)
        j = 1  # y-axes
        k = 2  # z-axis
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > eps:
            ax = math.atan2(M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2(M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0
        euler = [ax, ay, az]
        return euler  # roll, pitch ,yaw

    # __________________________________Get Observations____________________________________________________________
    def GetObservation(self):
        observation = []
        # returns base: position, orientation(in quaternion), linear velocity, angular velocity
        _, base_orientation_quaternion, _, euler_rate = self.GetLaelapsBaseInfo()
        euler = self.transfrom_euler_from_quaternion(
            base_orientation_quaternion)
        observation.extend(euler)
        observation.extend(euler_rate)
        return observation

    # ____________________________________ROS wait_________________________________________________________________________________

    def ros_wait(self, t):
        # It makes the execution of the next command in the python-ROS node wait for t secs, equivalent to rospy.sleep(time)
        start = rospy.get_rostime()
        ros_time_start = start.secs+start.nsecs*1e-9
        timeout = ros_time_start+t  # example 0.6 sec
        wait = True
        while wait:
            now = rospy.get_rostime()
            ros_time_now = now.secs+now.nsecs*1e-9
            if ros_time_now >= timeout:
                wait = False
    # _______________Publish commands from ROS to Gazebo__________________________________________________________________________________

    def publish_threads_motor_angles(self, i, motor_angle):
        self.motor_pub[i].publish(motor_angle[i])

    def publish_motor_angles(self, motor_angle):

        threads = list()

        for index in range(8):
            # logging.info("Main    : create and start thread %d.", index)
            x = threading.Thread(
                target=self.publish_threads_motor_angles, args=(index, motor_angle))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            # logging.info("Main    : before joining thread %d.", index)
            thread.join()
            # logging.info("Main    : thread %d done", index)

    def publish_threads_toe(self, i, toe_commands, phase):
        toe_class_msg.toex = toe_commands[i]  # toe_x
        toe_class_msg.toey = toe_commands[i+4]  # toe_y
        toe_class_msg.phase = phase[i]
        self.toe_pub[i].publish(toe_class_msg)

    def pubilsh_toe_commands(self, toe_x_y, phase_shift):
        threads = list()

        for index in range(len(laelaps_feet)):
            # toe_class_msg=Toe()
            x = threading.Thread(target=self.publish_threads_toe, args=(
                index, toe_x_y, phase_shift))
            threads.append(x)
            x.start()

        for index, thread in enumerate(threads):
            thread.join()

    # _______________Reset function__________________________________________________________________________________
    def init_reset(self):
        global init_angles, init_toe_commands, init_phase_shift
        self.gazebo.unpauseSim()
        self.gazebo.resetSim()
        self.gazebo.pauseSim()
        self.gazebo.resetJoints(init_angles)
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self.pubilsh_toe_commands(init_toe_commands, init_phase_shift)
        # wait some time until robot stabilizes itself in the environment
        self.ros_wait(0.01)
        self.gazebo.pauseSim()
        self.episode_reward = 0
        return self.GetObservation()

    def reset(self):

        global init_angles, init_toe_commands, init_phase_shift, saving_path, episode_savingpath
        self.gazebo.unpauseSim()
        self.gazebo.resetSim()
        self.gazebo.pauseSim()
        # rospy.loginfo("-----RESET-----")
        self.gazebo.resetJoints(init_angles)  # reset joint angles
        self.gazebo.unpauseSim()

        # Reset JoinStateControlers because resetSim doesnt reset TFs, generating issues with simulation time
        self.controllers_object.reset_controllers()

        self.pubilsh_toe_commands(init_toe_commands, init_phase_shift)
        self.ros_wait(0.01)

        self.gazebo.pauseSim()
        # self.plot_tensorboard("Episode Total Rewards", self.episode_reward,self.frames)
        # Initialize Episode Values
        # reset the variable base position to 0
        self.last_base_position = [0.0, 0.0, 0.0]
        self.episode_step = 0
        self.episode_reward = 0

        self.episode_number += 1
        episode_savingpath = str(saving_path) + \
            "/Episode_" + str(self.episode_number)
        if self.saving_option:
            os.makedirs(episode_savingpath)

        self.save_time("Start")

        return self.GetObservation()  # initial state S_0

    # _______________Step function__________________________________________________________________________________
    def step(self, action):

        global x_high, x_low, y_high, y_low, step_phase_shift
        # Action values from NN from [-1,1] are mapped to the action space limits
        action_x = interp(action[0:4], [-1, 1], [x_low, x_high])
        action_y = interp(action[4:8], [-1, 1], [y_low, y_high])
        action = np.append([action_x], [action_y])

        # rospy.loginfo("------Step Action: %s",action)
        time_out = rospy.get_time()+step_time_out

        while True:

            self.gazebo.unpauseSim()
            self.pubilsh_toe_commands(action, step_phase_shift)
            self.gazebo.pauseSim()
            done = self.termination()
            if rospy.get_time() > time_out or done == True:
                break
            else:
                continue

        self.gazebo.pauseSim()

        reward = self.reward()

        self.episode_reward += reward
        # rospy.loginfo("-------------EnvFram idx %s : Reward in Env per Step %s-------------", self.frames ,reward)
        # Tensorboard
        # self.plot_tensorboard("env step r", reward,self.frames)

        self.frames += 1
        self.episode_step += 1

        return self.GetObservation(), reward, done, {}

    # ______________Episode Termination________________________________________________________________________________
    def termination(self):
        global episode_start
        pos, quaternion, _, _ = self.GetLaelapsBaseInfo()

        roll, pitch, yaw = self.transfrom_euler_from_quaternion(quaternion)

        if self.ramp_available:
            # pitch angle of the ramp, inclined slopes goal 3.2 m
            ramp_pitch = self.GetRampInclination()
            is_fallen = math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.3+math.fabs(ramp_pitch) or math.fabs(
                yaw) > 0.2 or math.fabs(pos[0]) >= self.env_goal or math.fabs(pos[1]) > 0.2  # 3.2 m is nearly the end of the ramp
        else:
            ramp_pitch = 0  # flat terrain goal 6 m
            is_fallen = math.fabs(roll) > 0.3 or math.fabs(pitch) > 0.3+math.fabs(ramp_pitch) or math.fabs(
                yaw) > 0.2 or math.fabs(pos[0]) >= self.env_goal or math.fabs(pos[1]) > 0.2 or math.fabs(pos[2]) > 0.17

        if is_fallen:
            self.save_time("Finish")  # episode finish
            self.save_termination([math.fabs(roll), math.fabs(
                pitch), math.fabs(yaw), math.fabs(pos[0]), math.fabs(pos[1])])
            rospy.loginfo("ROll: %s : %s , Pitch: %s : %s , Yaw: %s : %s , X: %s : %s , Y: %s : %s , Z: %s : %s", math.fabs(roll), math.fabs(roll) > 0.3, math.fabs(pitch), math.fabs(pitch) > 0.3+math.fabs(ramp_pitch),
                          math.fabs(yaw), math.fabs(yaw) > 0.2, math.fabs(pos[0]), math.fabs(pos[0]) >= 3.2, math.fabs(pos[1]), math.fabs(pos[1]) >= 0.2, math.fabs(pos[2]), math.fabs(pos[2]) > 0.17)
            # SAVING
            self.save_torques(self.torques_step, self.frames)
            self.save_baze_zaxis(self.base_zaxis, self.frames)
            self.save_baze_x_y(self.base_x_y, self.frames)
            self.save_euler_angles(self.euler_angles, self.frames)
            self.save_angular_velocities(self.euler_rates, self.frames)
            self.save_sim_time(self.sim_t, self.frames)
            self.torques_step = []
            self.euler_angles = []
            self.euler_rates = []
            self.base_zaxis = []
            self.base_x_y = []
            self.sim_t = []
            episode_start = False
        return is_fallen

    # _______________Reward function__________________________________________________________________________________
    def reward(self):

        nan_check = self.GetNanToeCheck()
        if any(nan_check):
            rospy.loginfo(
                "---Angles Inverse Kinematics: NaN %s, try again---r=-1", nan_check)
            reward = -1
        else:
            if self.ramp_available:
                ramp_pitch = self.GetRampInclination()  # pitch angle of the ramp
            else:
                ramp_pitch = 0
            current_base_position, _, _, _ = self.GetLaelapsBaseInfo()

            # Reward forward motion
            forward_reward = math.fabs(current_base_position[0]/math.cos(
                ramp_pitch)) - math.fabs(self.last_base_position[0]/math.cos(ramp_pitch))
            # rospy.loginfo("current_base_position[0] %s self.last_base_position[0] %s forward_reward %s",math.fabs(current_base_position[0]),math.fabs(self.last_base_position[0]),forward_reward )

            # Penalize drifting
            drift_reward = math.fabs(
                current_base_position[1]) - math.fabs(self.last_base_position[1])
            # rospy.loginfo("current_base_position[1] %s self.last_base_position[1] %s drift_reward %s",math.fabs(current_base_position[1]),math.fabs(self.last_base_position[1]),drift_reward )

            self.last_base_position = current_base_position

            reward = (self.distance_weight * forward_reward -
                      self.drift_weight * drift_reward)
        return reward
    # ______________Saving values and ploting tensorboard_____________________________________________-

    def tensorboardwriter(self, folder, use_tensorboardEnV, use_tensorboardAlg):
        self.enable_tensorboard = use_tensorboardEnV
        if use_tensorboardEnV or use_tensorboardAlg:
            self.writer = SummaryWriter(folder)
        else:
            self.writer = []
        return self.writer

    def savingpath(self, folder, saving_option):
        global saving_path
        self.saving_option = saving_option
        saving_path = folder

    def plot_tensorboard(self, title, var1, var2):
        if self.enable_tensorboard:
            self.writer.add_scalar(title, var1, var2)

    def save_actions(self, action, step_number):
        global episode_savingpath
        if self.saving_option:
            action = np.array(action, dtype=np.float64)
            np.save(str(episode_savingpath)+"/"+str(step_number), action)

    def save_torques(self, torques, step_number):
        global episode_savingpath
        if self.saving_option:
            torques = np.array(torques, dtype=np.float64)
            np.save(str(episode_savingpath) +
                    "/Torques_"+str(step_number), torques)

    def save_euler_angles(self, euler_angles, step_number):
        global episode_savingpath
        if self.saving_option:
            euler_angles = np.array(euler_angles, dtype=np.float64)
            np.save(str(episode_savingpath)+"/EulerAngles_" +
                    str(step_number), euler_angles)

    def save_angular_velocities(self, angular_velocities, step_number):
        global episode_savingpath
        if self.saving_option:
            angular_velocities = np.array(angular_velocities, dtype=np.float64)
            np.save(str(episode_savingpath)+"/AngularVelocities_" +
                    str(step_number), angular_velocities)

    def save_baze_zaxis(self, base_zaxis, step_number):
        global episode_savingpath
        if self.saving_option:
            base_zaxis = np.array(base_zaxis, dtype=np.float64)
            np.save(str(episode_savingpath)+"/BaseZaxis_" +
                    str(step_number), base_zaxis)

    def save_baze_x_y(self, base_x_y, step_number):
        global episode_savingpath
        if self.saving_option:
            base_x_y = np.array(base_x_y, dtype=np.float64)
            np.save(str(episode_savingpath)+"/BaseX_Y" +
                    str(step_number), base_x_y)

    def save_sim_time(self, sim_t, step_number):
        global episode_savingpath
        if self.saving_option:
            sim_t = np.array(sim_t, dtype=np.float64)
            np.save(str(episode_savingpath)+"/Sim_t"+str(step_number), sim_t)

    def save_time(self, start_or_finish):
        global episode_savingpath
        if self.saving_option:
            episode_time = str(time.ctime(time.time()))
            np.save(str(episode_savingpath)+"/Episode_" +
                    str(start_or_finish), episode_time)

    def save_termination(self, termination_critiria):
        global episode_savingpath
        if self.saving_option:
            np.save(str(episode_savingpath) +
                    "/termination", termination_critiria)

    def close(self):
        rospy.logdebug("Closing RobotGazeboEnvironment")
        rospy.signal_shutdown("Closing RobotGazeboEnvironment")