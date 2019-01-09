import torch
import gym
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.mlp_comm import CommNet

'''
This will test a CommNet with a state input of 4, a communication dimension of 2,
a single communication step, an action state of 4, and two agents.
'''

num_agents = 2
state_dim_single_agent = 4
state_dim = state_dim_single_agent * num_agents
action_sim_single_agent = 4
action_dim = action_sim_single_agent * num_agents
communication_steps = 1
communication_dim = 2
comm_net = CommNet(
            state_dim=state_dim,
            action_dim=action_dim,
            communication_dim=communication_dim,
            num_agents=num_agents,
            communication_steps=communication_steps,
            activation_name='tanh',
            log_std=0.0
)

comm_net.comm_model[0].model[0].weight.data = torch.ones_like(comm_net.comm_model[0].model[0].weight.data)*0.1
comm_net.comm_model[0].model[0].bias.data = torch.ones_like(comm_net.comm_model[0].model[0].bias.data)*0.0

comm_net.comm_model[2].model[0].hidden_weights.data = torch.ones_like(comm_net.comm_model[2].model[0].hidden_weights.data)*0.2
comm_net.comm_model[2].model[0].communication_weights.data = torch.ones_like(comm_net.comm_model[2].model[0].communication_weights.data)*0.1

comm_net.policy_model.model.action_mean.weight.data = torch.ones_like(comm_net.policy_model.model.action_mean.weight.data)*0.3
comm_net.policy_model.model.action_mean.bias.data = torch.ones_like(comm_net.policy_model.model.action_mean.bias.data)*0.0
comm_net.policy_model.model.action_log_std.data = torch.ones_like(comm_net.policy_model.model.action_log_std)*0.0

print("Parameters list")
for p in comm_net.parameters():
    print(p)

test_input = [torch.ones(1,state_dim_single_agent)*1.0, torch.ones(1,state_dim_single_agent)*2.0]
test_output = [torch.ones(1,state_dim_single_agent)*0.16638, torch.ones(1,state_dim_single_agent)*0.19713]
print("\nRunning test with input: {}".format(test_input))
action_data = comm_net(test_input)
action_mean = [a for a,_,_ in action_data]
agent_1_out = action_mean[0].detach().numpy()
agent_2_out = action_mean[1].detach().numpy()
success_agent1 = np.allclose(test_output[0].numpy(), action_mean[0].detach().numpy(), 1e-2)
success_agent2 = np.allclose(test_output[1].numpy(), action_mean[1].detach().numpy(), 1e-2)
print("\nTest agent1 output: {} \nExpected agent1 output: {}".format(agent_1_out, test_output[0].numpy()))
print("Agent1 success: {}".format(success_agent1))
print("\nTest agent1 output: {} \nExpected agent1 output: {}".format(agent_2_out, test_output[1].numpy()))
print("Agent2 success: {}".format(success_agent2))
print('\nTotal success: {}'.format(success_agent1 and success_agent2))