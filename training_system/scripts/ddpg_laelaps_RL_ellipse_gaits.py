#!/usr/bin/env python3

"""
Deep Determinitic Policy Gradient (DDPG) Network & Agent based on:
1) Max Lapan's implementation in the book Deep Reinforcement Learning Hands-On: https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On
2) Pytorch 
3) PTAN (PyTorch AgentNet) package for RL https://github.com/Shmuma/ptan
"""

import rospy
import laelaps_env_ellipse
import os
import ptan
import time
from datetime import datetime
import gym
from tensorboardX import SummaryWriter
import numpy as np
import ddpg_model
import torch
import torch.optim as optim
import torch.nn.functional as F
import sys

print("training python version", sys.version)


ENV_ID = "LaelapsEnvEllipse-v0"
print("Environment ID", ENV_ID)

rospy.init_node('ddpg_laelaps_RL_ellipse',
                anonymous=True, log_level=rospy.INFO)

# Get Parameters from Yaml file ddpg_ellipse.yaml
gamma = rospy.get_param("/gamma")
batch_size = rospy.get_param("/batch_size")
learning_rate = rospy.get_param("/learning_rate")
replay_size = rospy.get_param("/replay_size")
replay_initial = rospy.get_param("/replay_initial")

validation_steps = rospy.get_param("/validation_steps")
rewards_steps_count = rospy.get_param("/rewards_steps_count")
time_iter_print = 1000
validation_tests = rospy.get_param("/validation_tests")


def validation_net(actor_net, critic_net, frames, env, count=validation_tests, device="cpu"):
    rewards = 0.0
    steps = 0
    env.savingpath(test_episode_actions, saving_option=False)
    for i in range(count):
        print("Local time test beginning: ", time.ctime(time.time()))
        obs = env.reset()
        while True:
            obs = ptan.agent.float32_preprocessor([obs]).to(device)
            mu = actor_net(obs)
            action = mu.squeeze(dim=0).data.cpu().numpy()

            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                print("Finished validation count", i)
                print("Local time after test: ", time.ctime(time.time()))
                break
    return rewards / count, steps / count


def unpack_batch(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            # bec u dont want None values in the target network, it is a trick to avoid none due to termination of the episode
            last_states.append(exp.state)
            # but then using the dones bool value Q value will be setted to zero to avoid nan in the network as Q value for None (terminated state) is predetermined to have value 0
        else:
            last_states.append(exp.last_state)
    states = ptan.agent.float32_preprocessor(states).to(device)
    actions = ptan.agent.float32_preprocessor(actions).to(device)
    rewards = ptan.agent.float32_preprocessor(rewards).to(device)
    last_states = ptan.agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states, actions, rewards, dones_t, last_states


if __name__ == "__main__":
    test_condition = input(
        "Specifiy the test condition for running the algorithm: ")
    start_time = time.ctime(time.time())
    print("START TIME: ", start_time)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    if torch.cuda.is_available():
        print("I am using Device: GPU")
    else:
        print("I am using Device: CPU")

    env = gym.make(ENV_ID)
    validation_env = gym.make(ENV_ID)

    act_net = ddpg_model.DDPGActor(
        env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    crt_net = ddpg_model.DDPGCritic(
        env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(act_net)
    print(crt_net)
    tgt_act_net = ptan.agent.TargetNet(act_net)
    tgt_crt_net = ptan.agent.TargetNet(crt_net)

    # _________________________LOGGING AND TENSORBOARD______________________________________________________
    # Creat folder to save tensorboard files depending on test conditions
    #
    date_now = datetime.now()
    newDirName = date_now.strftime("%B_%d_%y_%H-%M")
    save_folder = os.path.dirname(os.path.realpath(
        __file__))+"/runs/"+str(newDirName)+"_ddpg"+test_condition

    train_folder = save_folder+"/Train"
    test_folder = save_folder+"/Test"
    model_save = save_folder+"/Model_best_test_rewards"
    train_episode_actions = train_folder+"/Episode_actions"
    test_episode_actions = test_folder+"/Episode_actions"

    os.makedirs(save_folder)  # Main folder
    os.makedirs(train_folder)  # Train folder
    os.makedirs(test_folder)  # Test folder
    os.makedirs(model_save)  # Model per test save
    # Save env values (i.e. actions, torques, etc.) used in each episode
    os.makedirs(train_episode_actions)
    # Save env values (i.e. actions, torques, etc.) used in each episode
    os.makedirs(test_episode_actions)

    writer = env.tensorboardwriter(
        train_folder, use_tensorboardEnV=False, use_tensorboardAlg=True)
    writer_val = validation_env.tensorboardwriter(
        test_folder, use_tensorboardEnV=False, use_tensorboardAlg=True)

    # ____________________________________________
    agent = ddpg_model.AgentDDPG(act_net, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(
        env, agent, gamma=gamma, steps_count=rewards_steps_count)
    """
    ExperienceSourceFirstLast: store tajectories with the first and last states only instead of full trajectory. 
    For every trajectory it outputs exp_source: 
    first states of the episode, and last states of the episode, total discounted rewards, and actions taken in first states 
    """
    buffer = ptan.experience.ExperienceReplayBuffer(
        exp_source, buffer_size=replay_size)  # Circular buffer (FIFO)
    # use different optimizers not to complicate how we deal with gradients
    act_opt = optim.Adam(act_net.parameters(), lr=learning_rate)
    crt_opt = optim.Adam(crt_net.parameters(), lr=learning_rate)

    frames = 0

    ''' PTAN Tensorboard logging: RewardTracker: tracks the rewards per episodes and their steps
    min_ts_diff: ts:training speed difference between time sampling and 
    processing of steps between episodes, 0 show every episode in the
    print of the console, default 1 sec, 
    class RewardTracker, inputs reward, frame (here the steps it 
    tracks the rewards per step), epsilon (for epsilon greedy) value 
    default is None
    track the key quantities with TensorBoard to be able to 
    monitor them during the training.

    TBMeanTracker allows to batch 
    fixed amount of historical values and write their mean into TB
    batch_size: integer size of batch to track
    '''

    # min_ts_diff print mean reward every 120 seconds
    with ptan.common.utils.RewardTracker(writer, min_ts_diff=120) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            while True:
                env.savingpath(train_episode_actions, saving_option=False)
                frames += 1
                # fills samples into buffer, here one sample
                buffer.populate(1)

                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    # total steps reward per episode
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track("episode_steps", steps[0], frames)
                    tracker.reward(rewards[0], frames)

                if len(buffer) < replay_initial:  # replay initial to start training
                    continue

                # In every iteration store the experience into the replay buffer and sample randomly the training batch
                batch = buffer.sample(batch_size)

                # first states of the episode, and last states of the episode, total immidiate rewards, and actions taken in first states
                states, actions, rewards, dones, last_states = unpack_batch(
                    batch, device)

                # train critic
                crt_opt.zero_grad()
                # batch actions comes from agent inside the experience by running actor network
                q = crt_net(states, actions)
                last_act = tgt_act_net.target_model(last_states)
                q_last = tgt_crt_net.target_model(last_states, last_act)
                # in the dones vector the index of the element with True changes the q_last to 0.0
                q_last[dones] = 0.0
                # Bellman equation using target network Q* output
                q_ref = rewards.unsqueeze(dim=-1) + q_last * gamma
                critic_loss = F.mse_loss(q, q_ref.detach())  # Bellman Error
                critic_loss.backward()
                crt_opt.step()
                tb_tracker.track("loss_critic", critic_loss, frames)
                tb_tracker.track("Y_target mean", q_ref.mean(), frames)

                # train actor
                act_opt.zero_grad()
                cur_actions = act_net(states)
                actor_loss = -crt_net(states, cur_actions)
                actor_loss = actor_loss.mean()
                actor_loss.backward()
                act_opt.step()
                tb_tracker.track("loss_actor", actor_loss, frames)

                # Target networks update
                # soft sync (better for continuous action space) the weights of the optimized network to the target network on every step,
                # but only small ratio of the optimized network weights are transfered to the target network
                tgt_act_net.alpha_sync(alpha=1 - 1e-3)
                tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

                # Validation learning
                if frames % validation_steps == 0:
                    print("START TIME: ", start_time)
                    if frames % time_iter_print == 0:
                        local_time_now = time.ctime(time.time())
                        print("Local time now: ", local_time_now)
                    ts = time.time()
                    rewards, steps = validation_net(
                        act_net, crt_net, frames, validation_env, device=device)
                    print("Validation done in %.2f sec, reward %.3f, steps %d" % (
                        time.time() - ts, rewards, steps))
                    # the sum of the reward happened until termination (one episode) in every step divided by how many the model is validated
                    writer_val.add_scalar("test_reward", rewards, frames)
                    # the sum of test episodes steps divided by how many the model is validated
                    writer_val.add_scalar("test_steps", steps, frames)
                    name = "Test_Model_%+.3f_%d.dat" % (rewards, frames)
                    fname = os.path.join(model_save, name)
                    # save validation network parameters
                    torch.save(act_net.state_dict(), fname)
    pass
