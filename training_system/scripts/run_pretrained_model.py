#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Run saved validation models  

Test generalization of the saved model: 

i) Test generalization of the model in inclinations up to 15 degrees
ii) Test generalization in longer ramps 

"""
import gym
import os, os.path
import numpy as np 
import ddpg_model
import torch 
from tensorboardX import SummaryWriter
import laelaps_env_ellipse
import rospy

ENV_ID="LaelapsEnvEllipse-v0"
print("Environment ID",ENV_ID)


saving=False

if __name__ == "__main__":


    model_path=input("Specifiy the pretrained model path from the runs folder: ")
    

    rospy.init_node('test_model',disable_signals=True, log_level=rospy.INFO) 

    env=gym.make(ENV_ID)
    
    if saving:   
        test_condition=input("Specifiy the saving folder name: ")
        save_folder=os.path.dirname(os.path.realpath(__file__))+"/runs/"+"PretrainedModel_ddpg"+test_condition
        os.makedirs(save_folder) # Main folder
        env.savingpath(save_folder,saving_option=saving)
    else:
        save_folder=os.path.dirname(os.path.realpath(__file__))+"/runs"

    writer=env.tensorboardwriter(save_folder,use_tensorboardEnV=False,use_tensorboardAlg=False)
    net = ddpg_model.DDPGActor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(model_path))
    test_rewards=0.0
    test_number=20
    for i in range(test_number):
        obs = env.reset()
        total_reward = 0.0
        total_steps = 0
        while True:
            obs_v = torch.FloatTensor([obs])
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.numpy()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1
            if done:
                test_rewards+=total_reward
                print("----------Terminated----------TEST: ", i)
                print("In %d steps we got %.3f reward" % (total_steps, total_reward))
                break
    print("Mean total reward in all tests: ",test_rewards/test_number)
    

