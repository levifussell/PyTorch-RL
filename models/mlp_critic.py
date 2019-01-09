import torch.nn as nn
import torch


class Value(nn.Module):
    def __init__(self, state_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        last_dim = state_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.value_head = nn.Linear(last_dim, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))

        value = self.value_head(x)
        return value

class ValueMulti(Value):
    '''
    Equivalent to Value network but can take a list of states
    (one element for each agent) and concatenate them before
    feeding them through the Value network. On output it
    converts the concatenated input back into a list. Note
    that the concatentation/de-concatentation is not the same
    as Squeeze/Unsqueeze because each input can be fed through
    independently.
    '''
    def __init__(self, state_dim, hidden_size=[128, 128], activation='tanh'):
        super(ValueMulti, self).__init__(
                state_dim=state_dim,
                hidden_size=hidden_size,
                activation=activation
        )

    def forward(self, x):
        is_list = type(x) == type([])
        # if the input is a list, then we have to concatenate it to a batch
        xs = torch.cat(x, dim=0) if is_list else x
        value = super(ValueMulti, self).forward(xs)
        # if it was a list we want to convert it back aswell
        if is_list:
            div_rate = xs.size()[0] // len(x) # len(x) := number of agents
            ys = list(torch.split(value, div_rate, dim=0))
        else:
            ys = value
        return ys
