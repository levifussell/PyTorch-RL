import multiprocessing
from utils.replay_memory import Memory
from utils.torch import *
import math
import time


def collect_samples(pid, queue, env, policy, custom_reward,
                    mean_action, render, running_state, min_batch_size,
                    num_agents=1):
    torch.randn(pid)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = [0]*num_agents #0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0
    # render=False

    while num_steps < min_batch_size:
        state = env.reset()
        # print(state)
        if running_state is not None:
            state = running_state(state)
        reward_episode = [0]*num_agents

        # env.render()
        for t in range(150):
            # time.sleep(1)
            #TODO: temporarily for a single agent we make the state
            #  artificially into a list.
            # state_var = [tensor(state).unsqueeze(0), tensor(state).unsqueeze(0)]
            state_var = [tensor(s).unsqueeze(0) for s in state]
            # print('STATE', len(state_var))
            # state_var = state_var[:-1]
            # print(len(state_var))
            # state_var = [tensor(state).unsqueeze(0)]
            # print(state_var)
            # state_var = tensor(state).unsqueeze(0)
            with torch.no_grad():
                if mean_action:
                    # print('mean', policy(state_var))
                    action = policy(state_var)[0][0].numpy()
                else:
                    # print('else', policy.select_action(state_var))
                    # print(state_var)
                    action = policy.select_action(state_var)
                    # print(action[0][0].numpy())
                    # print(action)
                    # action_var = torch.stack([a[0] for a inaction, dim=1)

                    # action_var = torch.cat(action, dim=1)[0].numpy()

                    # action = [a[0] for a in action]
                    action = [a[0].numpy().tolist() for a in action]
                    # print(action_var)
                    # print(action)
                    # action = policy.select_action(state_var)[0].numpy()
            # TODO: this is added so that the prey is automatically controlled
            #  by arbitrary input.
            # action.append(np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
            # print('ACT', action)
            # print(action)
            action_var = action
            # print(action_var)
            # print(action)
            # action = int(action) if policy.is_disc_action else action.astype(np.float64)
            # print(action)
            # action_var = action
            # action_var = [int(a) for a in action_var] if policy.is_disc_action else [a.astype(np.float64) for a in action_var]
            # print(action_var)
            # print('aa', action)
            # print('av', action_var)
            # next_state, reward, done, _ = env.step(action)
            # TODO: while we use an environment that doesn't accept multi-agent
            #  action lists
            next_state, reward, done, _ = env.step(action_var)
            # print(reward)
            # reward = reward[:-1]
            # done = done[:-1]
            # reward_all = np.sum(reward)
            # for r in range(len(reward)):
            #     reward[r] = reward_all
            # print(reward)
            # reward = [reward, reward]
            # reward_episode += reward
            for r in range(len(reward_episode)):
                reward_episode[r] += reward[r]
            # print(reward_episode)
            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            # mask = 0 if done else 1
            mask = [float(d) for d in done]
            # print(mask)

            #TODO while we use an environment that doesn't accept multi-agent
            #  action lists
            # print(state)
            # state = [s.tolist() for s in state]
            # print(state)
            memory.push(state, action, mask, next_state, reward)
            # memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
                time.sleep(0.1)
            if np.all(done):
                break

            state = next_state

        # log stats
        num_steps += (t + 1)
        num_episodes += 1
        for r in range(len(reward_episode)):
            total_reward[r] += reward_episode[r]
        # total_reward += reward_episode
        min_reward = min(min_reward, np.min(reward_episode))
        max_reward = max(max_reward, np.max(reward_episode))
        # min_reward = 0.0
        # max_reward = 0.0

    log['num_steps'] = num_steps
    log['avg_steps'] = num_steps / num_episodes
    log['num_episodes'] = num_episodes
    log['total_reward'] = total_reward
    log['avg_reward'] = [t / num_episodes for t in total_reward] #total_reward / num_episodes
    log['max_reward'] = max_reward
    log['min_reward'] = min_reward
    if custom_reward is not None:
        log['total_c_reward'] = total_c_reward
        log['avg_c_reward'] = total_c_reward / num_steps
        log['max_c_reward'] = max_c_reward
        log['min_c_reward'] = min_c_reward

    if queue is not None:
        queue.put([pid, memory, log])
    else:
        return memory, log


def merge_log(log_list):
    log = dict()
    log['total_reward'] = sum([x['total_reward'] for x in log_list])
    log['num_episodes'] = sum([x['num_episodes'] for x in log_list])
    log['num_steps'] = sum([x['num_steps'] for x in log_list])
    log['avg_reward'] = log['total_reward'] / log['num_episodes']
    log['max_reward'] = max([x['max_reward'] for x in log_list])
    log['min_reward'] = min([x['min_reward'] for x in log_list])
    if 'total_c_reward' in log_list[0]:
        log['total_c_reward'] = sum([x['total_c_reward'] for x in log_list])
        log['avg_c_reward'] = log['total_c_reward'] / log['num_steps']
        log['max_c_reward'] = max([x['max_c_reward'] for x in log_list])
        log['min_c_reward'] = min([x['min_c_reward'] for x in log_list])

    return log


class Agent:

    def __init__(self, env, policy, device, custom_reward=None,
                 mean_action=False, render=False, running_state=None, num_threads=1):
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, num_agents=1, render=False):
        self.render = render
        t_start = time.time()
        to_device(torch.device('cpu'), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

        for i in range(self.num_threads-1):
            worker_args = (i+1, queue, self.env, self.policy, self.custom_reward, self.mean_action,
                           False, self.running_state, thread_batch_size)
            workers.append(multiprocessing.Process(target=collect_samples, args=worker_args))
        for worker in workers:
            worker.start()

        memory, log = collect_samples(0, None, self.env, self.policy, self.custom_reward, self.mean_action,
                                      self.render, self.running_state, thread_batch_size, num_agents=num_agents)

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log['sample_time'] = t_end - t_start
        log['action_mean'] = np.mean(np.vstack(batch.action), axis=0)
        log['action_min'] = np.min(np.vstack(batch.action), axis=0)
        log['action_max'] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
