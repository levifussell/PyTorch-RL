import torch as th
import torch.nn as nn

from models.mlp_policy import Policy

# class MultiInputLinearLayer(nn.Module):
#     '''
#     A linear layer which takes in a vector of concatenated inputs
#     from multiple sources and duplicates
#     '''

class MultiChannelNet(nn.Module):

    def __init__(self, input_dim_single_agent,
                output_dim_single_agent, num_agents):
        '''
        This is an abstract class which will take a list of inputs
        (one input for each agent) and feed them through the network
        independently
        '''
        super(MultiChannelNet, self).__init__()
        self.input_dim_single_agent = input_dim_single_agent
        self.output_dim_single_agent = output_dim_single_agent
        self.num_agents = num_agents

        # self.unsqueeze = UnSqueezeLayer(self.num_agents)

    def channel(self, x):
        '''
        Acts as a channel for a single agent, which each input is
        fed through. This one has to be implemented because it
        corresponds to the 'forward()' methods. More methods
        can be added by using 'split_and_apply()'.
        '''
        raise NotImplementedError

    def split_and_apply(self, *x_input, func_apply):
        '''
        Splits an input and applies a function to each component.
        Similar to the 'map()' method but first divides the input
        into its component vectors.
        '''
        # assert len() == self.input_dim_single_agent * self.num_agents, "Input is not the same for each agent."
        # x_input_unsqueezed = [self.unsqueeze(x) for x in x_input]
        # ys = list(map(func_apply, *x_input_unsqueezed))
        ys = list(map(func_apply, *x_input))
        # ys = [func_apply(x_input[(self.input_dim_single_agent*i):(self.input_dim_single_agent*(i+1))]) 
        #         for i in range(self.num_agents)]
        return ys

    def forward(self, x):
        ys = self.split_and_apply(x, func_apply=self.channel)
        return ys

class MultiChannelFeedForward(MultiChannelNet):

    def __init__(self, input_dim_single_agent,
                output_dim_single_agent, num_agents,
                activation_name='tanh',
                hidden_dims=[16]):
        super(MultiChannelFeedForward, self).__init__(
            input_dim_single_agent, output_dim_single_agent,
            num_agents
        )

        if activation_name == 'tanh':
            self.activation_func = nn.Tanh()
        elif activation_name == 'relu':
            self.activation_func = nn.ReLU()
        else:
            assert False, "NO ACITVATION FUNCTION SPECIFIED."

        layer_sizes = [self.input_dim_single_agent]
        layer_sizes.extend(hidden_dims)
        layer_sizes.append(self.output_dim_single_agent)

        layers = []
        for ins,outs in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(ins, outs))
            layers.append(self.activation_func)
        self.model = nn.Sequential(*layers)

    def channel(self, x):
        y = self.model(x)
        return y
        
class MultiChannelPolicy(MultiChannelNet):

    def __init__(self, input_dim_single_agent,
                action_dim_single_agent, num_agents,
                activation_name='tanh',
                hidden_dims=[16],
                log_std=0):
        super(MultiChannelPolicy, self).__init__(
            input_dim_single_agent, action_dim_single_agent,
            num_agents
        )

        self.model = Policy(
            state_dim=input_dim_single_agent,
            action_dim=action_dim_single_agent,
            hidden_size=hidden_dims,
            activation=activation_name,
            log_std=log_std
        )

    def channel(self, xs):
        y = self.model(xs)
        return y

    def select_action(self, xs):
        ys = self.split_and_apply(xs, func_apply=self.model.select_action)
        return ys

    def get_kl(self, xs):
        ys = self.split_and_apply(xs, func_apply=self.model.get_kl)
        return ys

    def get_log_prob(self, xs, actions):
        ys = self.split_and_apply(xs, actions, func_apply=self.model.get_log_prob)
        return ys

    def get_fim(self, xs):
        ys = self.split_and_apply(xs, func_apply=self.model.get_fim)
        return ys


class SqueezeLayer(nn.Module):

    def __init__(self):
        super(SqueezeLayer, self).__init__()
    
    def forward(self, xs):
        y = th.reshape(th.cat(xs, dim=1), (xs[0].size()[0], -1))
        return y

class UnSqueezeLayer(nn.Module):

    def __init__(self, num_agents):
        super(UnSqueezeLayer, self).__init__()
        self.num_agents = num_agents

    def forward(self, x):
        assert x.size()[1] % self.num_agents == 0, "num_agents does not evenly divide the input for UnSqueeze."
        div_rate = x.size()[1] // self.num_agents
        ys = list(th.split(x, div_rate, dim=1))
        return ys

class CommLayer(nn.Module):
    '''
    Layer which provides communication between agents by taking
    the mean of N-1 of the inputs and adding it to the input of the
    Nth agent. Make sure the agent inputs have been squeeze prior to
    this so they are not in a list.
    '''
    def __init__(self, input_dim, num_agents, activation_name='tanh'):
        super(CommLayer, self).__init__()

        assert input_dim % num_agents == 0, "The number of input dimensions does not evenly divide into the number of agents."

        self.input_dim = input_dim
        self.num_agents = num_agents
        self.input_dim_single_agent = input_dim // num_agents

        if activation_name is 'tanh':
            self.activation_func = nn.Tanh()
        elif activation_name is 'relu':
            self.activation_func = nn.ReLU()
        else:
            assert False, "NO ACTIVATION FUNCTION SPECIFIED."

        # custom weights
        self.hidden_weights = nn.Parameter(th.randn(self.input_dim_single_agent, 
                                            self.input_dim_single_agent))
        self.hidden_weights.data.mul_(0.1)

        self.communication_weights = nn.Parameter(th.randn(self.input_dim_single_agent, 
                                            self.input_dim_single_agent))
        self.hidden_weights.data.mul_(0.1)

        # this matrix is used to compute the mean of the inputs
        self.norm_term = self.num_agents - 1 if self.num_agents > 1 else 1
        # self.mean_mat = (th.eye(self.num_agents)
        #                 ).repeat(1,self.input_dim_single_agent).reshape(
        #                     self.input_dim, -1).transpose(0, 1)
        self.mean_mat = (th.eye(self.num_agents).repeat(
                        self.input_dim_single_agent, 1).transpose(
                        0, 1).reshape(self.input_dim, -1)).repeat(
                        self.input_dim_single_agent, 1).transpose(
                        1, 0).reshape(self.input_dim, -1)

        # just a quick test to check one of the blocks is correct
        assert (self.mean_mat[0:self.input_dim_single_agent,
                0:self.input_dim_single_agent]).sum().data.numpy() == \
                self.input_dim_single_agent**2, \
                "Matrix for communication layer is incorrect: {}".format(
                self.mean_mat)

        # INCORRECT VERSION BELOW
        # create the matrix which computes the mean across N-1 of the agents
        # if self.num_agents != 1:
        #     self.mean_mat = (
        #         # communication portion takes an average of all other inputs
        #         (1.0 - th.eye(self.num_agents)) / (self.num_agents - 1) + \
        #         # identity portion preserves the hidden state of the agent
        #         #  recieving the communication
        #         th.eye(self.num_agents)
        #     # duplicates the elements along dim=1 so that the matrix can take the input
        #     #  relative to the size of the input dimension
        #     ).repeat(1,self.input_dim_single_agent).reshape(
        #             self.input_dim, -1).transpose(0, 1)
        # else:

        #     self.mean_mat = th.eye(self.num_agents).repeat(1,self.input_dim_single_agent).reshape(
        #             self.input_dim, -1).transpose(0, 1)
        #      # i.e. this is just a 1xD matrix with the value 1.

        # # just a quick test to check one of the sections is correct
        # assert (self.mean_mat[0, 0:self.input_dim_single_agent]
        #         ).sum().data.numpy() == self.input_dim_single_agent, \
        #         "Matrix for communication layer is incorrect: {}".format(
        #         self.mean_mat)

        # # i.e. the matrix should look like this for 3 agents of dim 2:
        # # [[1.0, 1.0, 1/(N-1), 1/(N-1), 1/(N-1), 1/(N-1)],
        # #  [1/(N-1), 1/(N-1), 1.0, 1.0, 1/(N-1), 1/(N-1)],
        # #  [1/(N-1), 1/(N-1), 1/(N-1), 1/(N-1), 1.0, 1.0]]

    def forward(self, x):
        assert x.size()[1] == self.input_dim, "Input dimensionality incorrect: {} vs. {}".format(x.size()[1], self.input_dim)

        hid_mat = self.hidden_weights.repeat(self.num_agents, self.num_agents)
        com_mat = self.communication_weights.repeat(self.num_agents, self.num_agents)

        # print(hid_mat)
        # print(hid_mat.size())
        # print(com_mat)
        # print(com_mat.size())
        # print(self.mean_mat.size())

        final_mat = hid_mat * self.mean_mat + com_mat * (1.0 - self.mean_mat) / self.norm_term
        y = self.activation_func(nn.functional.linear(x, final_mat, None))  

        # print(y)

        # print(x.size())
        # print(self.mean_mat)
        # print(self.mean_mat.size())
        # y = th.matmul(self.mean_mat, x)
        return y
    
class CommBlock(nn.Module):
    '''
    CommBlock takes as input a concatenation of agent states, puts them through
    a CommLayer and then UnSqueezes them (separates them into a list of inputs)
    and passes each agent input through an MLP.
    '''
    def __init__(self, input_dim, output_dim, num_agents):
        super(CommBlock, self).__init__()
        assert input_dim % num_agents == 0, "The number of input dimensions does not evenly divide into the nubmer of agents."

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_agents = num_agents
        self.input_dim_single_agent = input_dim // num_agents

        self.model = nn.Sequential(
            CommLayer(self.input_dim, self.num_agents),
            UnSqueezeLayer(self.num_agents),
            # MultiChannelFeedForward(
            #     input_dim_single_agent=self.input_dim_single_agent,
            #     output_dim_single_agent=self.input_dim_single_agent,
            #     num_agents=self.num_agents,
            #     activation_name='tanh',
            #     hidden_dims=[] # no hidden layers, just a single-layer MLP.
            # ),
            # the output of this model will be an UnSqueezed list,
            #  which will need to be squeezed if a CommLayer follows,
            #  but not squeezed if the final output decoding layer 
            #  follows.
        )

    def forward(self, x):
        ys = self.model(x)
        return ys

class CommNet(nn.Module):

    def __init__(self, state_dim, action_dim, 
                communication_dim,
                num_agents, communication_steps, 
                activation_name='tanh', log_std=0):
        super(CommNet, self).__init__()

        assert action_dim % num_agents == 0, "The number of input dimensions does not evenly divide into the nubmer of agents."

        self.is_disc_action = False
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_dim_single_agent = action_dim // num_agents
        self.communication_dim = communication_dim
        self.state_dim_single_agent = state_dim // num_agents
        self.num_agents = num_agents
        self.communication_steps = communication_steps
        self.log_std = log_std

        if activation_name is 'tanh':
            self.activation_func = nn.Tanh()
        elif activation_name is 'relu':
            self.activation_func = nn.ReLU()
        else:
            assert False, "NO ACTIVATION FUNCTION SPECIFIED."

        comm_layers = [
                # first encode the state input into the communication layer(s).
                MultiChannelFeedForward(
                    input_dim_single_agent=self.state_dim_single_agent,
                    output_dim_single_agent=self.communication_dim,
                    num_agents=self.num_agents,
                    activation_name=activation_name,
                    hidden_dims=[] # none.
                    )
        ]

        # now add the communication layers
        for _ in range(self.communication_steps):
            comm_layers.extend([
                SqueezeLayer(),
                CommBlock(
                    input_dim=self.communication_dim*self.num_agents,
                    output_dim=self.communication_dim*self.num_agents,
                    num_agents=self.num_agents
                    )
            ])

        self.comm_model = nn.Sequential(*comm_layers)

        # finally decode layer(s) and output a policy (actions).
        self.policy_model = MultiChannelPolicy(
                input_dim_single_agent=self.communication_dim,
                action_dim_single_agent=self.action_dim_single_agent,
                num_agents=self.num_agents,
                activation_name=activation_name,
                hidden_dims=[], # none.
                log_std=self.log_std
            )

    def __comm_forward__(self, xs):
        assert type(xs) == type([]), "The input must be a list of agent states."
        ys = self.comm_model(xs)
        return ys

    def forward(self, xs):
        ys = self.__comm_forward__(xs)
        # action_mean, action_log_std, action_std = self.policy_model(ys)
        action_data = self.policy_model(ys)
        return action_data

    # relevant methods wrapped to call for the policy
    def select_action(self, xs):
        ys = self.__comm_forward__(xs)
        action = self.policy_model.select_action(ys)
        return action

    def get_kl(self, xs):
        ys = self.__comm_forward__(xs)
        kl = self.policy_model.get_kl(ys)
        return kl

    def get_log_prob(self, xs, actions):
        ys = self.__comm_forward__(xs)
        log_prob = self.policy_model.get_log_prob(ys, actions)
        return log_prob

    def get_fim(self, xs):
        ys = self.__comm_forward__(xs)
        # cov_inv, mean, std_data = self.policy_model.forward(ys)
        fim_data = self.policy_model.forward(ys)
        return fim_data