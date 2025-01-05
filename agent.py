import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from buffer import Buffer
from model import build_encoder


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = build_encoder(latent_dim)

    def forward(self, x):
        return self.encoder(x.permute((0, 3, 1, 2)) / 255.0)  # [B, H, W, C] -> [B, C, H, W]


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def gae(agent, next_obs, rewards, device, args, next_done, dones, values):
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(rewards).to(device)
        last_gae_lam = 0
        for t in reversed(range(args.num_steps)):
            next_non_terminal = 1.0 - next_done if t == args.num_steps - 1 else 1.0 - dones[t + 1]
            next_values = next_value if t == args.num_steps - 1 else values[t + 1]
            delta = rewards[t] + args.gamma * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_non_terminal * last_gae_lam
        returns = advantages + values
    return advantages, returns


class Agent(nn.Module):
    def __init__(self, args, envs, writer, device):
        super().__init__()
        self.args = args
        self.envs = envs
        self.writer = writer
        self.device = device
        self.global_step = 0
        self.obs = torch.Tensor(self.envs.reset()).to(self.device)
        self.done = torch.zeros(self.args.num_envs).to(self.device)

        self.buffer = Buffer(args, envs, device)
        self.encoder = Encoder(256)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

        self.return_list = []
        self.update_index = 1

        self.advantages = None
        self.returns = None

    def get_value(self, x):
        return self.critic(self.encoder(x))

    def get_action_and_value(self, x, a=None, show_all=False):
        hidden = self.encoder(x)
        actor_output = self.actor(hidden)
        distribution = Categorical(logits=actor_output)
        if a is None:
            a = distribution.sample()
        if show_all:
            return a, distribution.log_prob(a), distribution.entropy(), self.critic(hidden), distribution.probs
        return a, distribution.log_prob(a), distribution.entropy(), self.critic(hidden)

    def get_probs(self, x):
        actor_output = self.actor(self.encoder(x))
        distribution = Categorical(logits=actor_output)
        return distribution.probs

    def interaction(self, iteration):
        for step in range(self.args.num_steps):
            self.global_step += self.args.num_envs

            with torch.no_grad():
                action, log_prob, _, value = self.get_action_and_value(self.obs)

            # Update the environments
            next_obs, reward, next_done, info = self.envs.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)
            self.buffer.push(
                self.obs,
                action,
                log_prob,
                value.flatten(),
                torch.tensor(reward).to(self.device).view(-1),
                self.done
            )
            self.obs, self.done = next_obs, next_done

            for item in info:
                if 'episode' in item.keys():
                    if iteration == self.update_index:
                        self.return_list.append(item['episode']['r'])
                    else:
                        self.writer.add_scalar(
                            'This is for plotting/return_train', np.mean(self.return_list), self.update_index
                        )
                        self.return_list.clear()
                        self.return_list.append(item['episode']['r'])
                        self.update_index += 1

        # GAE
        with torch.no_grad():
            next_value = self.get_value(self.obs).reshape(1, -1)
            self.advantages = torch.zeros_like(self.buffer.rewards).to(self.device)
            last_gae_lam = 0
            for t in reversed(range(self.args.num_steps)):
                next_non_terminal = 1.0 - self.done if t == self.args.num_steps - 1 else 1.0 - self.buffer.dones[t + 1]
                next_values = next_value if t == self.args.num_steps - 1 else self.buffer.values[t + 1]
                delta = self.buffer.rewards[t] + self.args.gamma * next_values * next_non_terminal - self.buffer.values[t]
                self.advantages[t] = last_gae_lam = delta + self.args.gamma * self.args.gae_lambda * next_non_terminal * last_gae_lam
            self.returns = self.advantages + self.buffer.values
        self.advantages, self.returns = self.advantages.reshape(-1), self.returns.reshape(-1)

    def get_all_data(self):
        b_states, b_actions, b_log_probs, b_values = self.buffer.get()
        return b_states, b_actions, b_log_probs, b_values, self.advantages, self.returns

    def evaluation(self, eval_envs, iteration):
        eval_next_obs = torch.Tensor(eval_envs.reset()).to(self.device)
        eval_episodic_returns = []
        while len(eval_episodic_returns) < 10:
            with torch.no_grad():
                eval_action, _, _, _ = self.get_action_and_value(eval_next_obs)
            eval_next_obs, _, _, eval_info = eval_envs.step(eval_action.cpu().numpy())
            eval_next_obs = torch.Tensor(eval_next_obs).to(self.device)
            for item in eval_info:
                if 'episode' in item.keys():
                    eval_episodic_returns.append(item['episode']['r'])

        self.writer.add_scalar(
            'This is for plotting/return_eval', np.mean(eval_episodic_returns), iteration
        )


if __name__ == '__main__':
    encoder = Encoder(256)
    input_tensor = torch.randn(size=(1, 64, 64, 3))
    output_tensor = encoder(input_tensor)
    print(input_tensor.shape)   # torch.Size([1, 64, 64, 3])
    print(output_tensor.shape)  # torch.Size([1, 256])
