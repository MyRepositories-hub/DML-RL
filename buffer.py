import torch


class Buffer:
    def __init__(self, args, envs, device):
        self.single_observation_space = envs.single_observation_space.shape
        self.states = torch.zeros((args.num_steps, args.num_envs) + self.single_observation_space).to(device)
        self.actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        self.dones = torch.zeros((args.num_steps, args.num_envs)).to(device)

        self.step = 0
        self.num_steps = args.num_steps
        self.device = device

    def push(self, states, actions, log_probs, values, rewards, dones):
        self.states[self.step] = states
        self.actions[self.step] = actions
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.step = (self.step + 1) % self.num_steps

    def get(self):
        return (
            self.states.reshape((-1,) + self.single_observation_space),
            self.actions.reshape(-1),
            self.log_probs.reshape(-1),
            self.values.reshape(-1)
        )
