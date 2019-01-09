import numpy as np
import torch as th
import gym
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
from models.mlp_comm import CommNet
from models.mlp_critic import ValueMulti
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

'''
First test is that the single-agent bipedal walker can learn through
the multi-agent system.
'''
dtype = th.float64
th.set_default_dtype(dtype)
device = th.device('cpu')
#TODO: cuda

# environment things
env = gym.make("BipedalWalker-v2")
state_dim = np.shape(env.observation_space)[0]*2
action_dim = np.shape(env.action_space)[0]

# seed
seed_id = 12
np.random.seed(seed_id)
th.manual_seed(seed_id)
env.seed(seed_id)

# multi-agent params
communication_dim = 64
communication_steps = 2
num_agents = 2

# create networks
actor_net = CommNet(
            state_dim=state_dim,
            action_dim=action_dim,
            # for the single-agent case this is
            #  equivalent to a hidden layer size.
            communication_dim=communication_dim,
            num_agents=num_agents,
            # for the single-agent case this is
            #  equivalent to the number of hidden layers.
            communication_steps=communication_steps,
            activation_name='tanh',
            log_std=0.0
)
critic_net = ValueMulti(state_dim // num_agents)

print(actor_net)
print(critic_net)

actor_net.to(device)
critic_net.to(device)
learning_rate_actor = 3e-4
learning_rate_critic = 3e-4
l2_reg = 1e-3
clip_epsilon = 0.2
save_model_interval = 0
log_interval = 1
min_batch_size = 2048 # size of a rollout batch
max_iter_num = 500
gamma = 0.99
tau = 0.95

optimiser_actor = th.optim.Adam(
            actor_net.parameters(),
            learning_rate_actor,
)
optimiser_critic = th.optim.Adam(
            critic_net.parameters(),
            learning_rate_critic
)

# PPO params
optim_epochs = 10 # number of times to repeat the collected samples
optim_batch_size = 64 # size of a training batch

# agent
agent = Agent(env, actor_net, device)

def update_params(batch, i_iter):
    # because each item in the batch is a list we must convert it
    #  accordingly.
    # print(batch.reward)
    # print(batch.state)
    states = [torch.from_numpy(a).to(dtype).to(device) for a in np.stack(batch.state, 1)]
    actions = [torch.from_numpy(a).to(dtype).to(device) for a in np.stack(batch.action, 1)]
    rewards = [torch.from_numpy(a).to(dtype).to(device) for a in np.stack(batch.reward, 1)]
    masks = [torch.from_numpy(a).to(dtype).to(device) for a in np.stack(batch.mask, 1)]
    # print(states)
    # print(states[0].size())
    # print(rewards)
    # states = [torch.from_numpy(np.stack(bs)).to(dtype).to(device) for bs in batch.state]
    # actions = [torch.from_numpy(np.stack(bs)).to(dtype).to(device) for bs in batch.action]
    # rewards = [torch.from_numpy(np.stack(bs)).to(dtype).to(device) for bs in batch.reward]
    # masks = [torch.from_numpy(np.stack(bs)).to(dtype).to(device) for bs in batch.mask]
    # print(rewards)

    # states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    # actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    # rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    # masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = critic_net(states)
        # print(states)
        # print(len(states))
        # print(states[0].size())
        # print(actions)
        # print(len(actions))
        # print(actions[0].size())
        fixed_log_probs = actor_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    # advantages, returns = \
    adv_returns_tuple = \
        [estimate_advantages(r, m, v, gamma, tau, device)
        for r,m,v in zip(rewards, masks, values)]
    advantages = []
    returns = []
    for a,r in adv_returns_tuple:
        advantages.append(a)
        returns.append(r)

    # advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    """perform mini-batch PPO update"""
    optim_iter_num = int(math.ceil(states[0].shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states[0].shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states = [st[perm].clone() for st in states]
        actions = [st[perm].clone() for st in actions]
        returns = [st[perm].clone() for st in returns]
        advantages = [st[perm].clone() for st in advantages]
        fixed_log_probs = [st[perm].clone() for st in fixed_log_probs]

        # states, actions, returns, advantages, fixed_log_probs = \
        #     states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states[0].shape[0]))

            states_b = [st[ind] for st in states]    
            actions_b = [st[ind] for st in actions]
            returns_b = [st[ind] for st in returns]
            advantages_b = [st[ind] for st in advantages]
            fixed_log_probs_b = [st[ind] for st in fixed_log_probs]

            # states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
            #     states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(actor_net, critic_net, optimiser_actor, optimiser_critic, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, clip_epsilon, l2_reg, multi_agent=True, num_agents=num_agents)


def main_loop():
    for i_iter in range(max_iter_num):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(min_batch_size)
        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        if i_iter % log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, log['min_reward'], log['max_reward'], log['avg_reward']))

        if save_model_interval > 0 and (i_iter+1) % save_model_interval == 0:
            to_device(torch.device('cpu'), actor_net, critic_net)
            pickle.dump((actor_net, critic_net, running_state),
                        open(os.path.join(assets_dir(), 'learned_models/{}_ppo.p'.format(env_name)), 'wb'))
            to_device(device, actor_net, critic_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()




