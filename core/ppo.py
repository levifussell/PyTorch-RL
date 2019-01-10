import torch


def ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, optim_value_iternum, states, actions,
             returns, advantages, fixed_log_probs, clip_epsilon, l2_reg,
             multi_agent=False, num_agents=1):

    """update critic"""
    for _ in range(optim_value_iternum):
        values_pred = value_net(states)
        # print(values_pred)
        # print(returns)
        if multi_agent:
            # convert the values to a single batch
            values_pred = torch.cat(values_pred, dim=0)
            # print(returns)
            returns = torch.cat(returns, dim=0)#.repeat(num_agents, 1)
        # print(returns)
        # print(returns.size())
        # print(values_pred)
        # print(values_pred.size())
        value_loss = (values_pred - returns).pow(2).mean()
        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        optimizer_value.zero_grad()
        value_loss.backward()
        optimizer_value.step()

    """update policy"""
    log_probs = policy_net.get_log_prob(states, actions)

    if multi_agent:
        # convert the log probs to a single batch
        log_probs = torch.cat(log_probs, dim=0)
        fixed_log_probs = torch.cat(fixed_log_probs, dim=0)
        advantages = torch.cat(advantages, dim=0)#.repeat(num_agents, 1)

    ratio = torch.exp(log_probs - fixed_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    policy_surr = -torch.min(surr1, surr2).mean()
    optimizer_policy.zero_grad()
    policy_surr.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 40)
    optimizer_policy.step()
