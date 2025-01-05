import argparse
import random

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from procgen import ProcgenEnv
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import build_encoder


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--torch_deterministic', type=bool, default=True)
    parser.add_argument('--total_time_steps', type=int, default=int(5e7))
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_eval_workers', type=int, default=4)
    parser.add_argument('--num_envs', type=int, default=64)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--num_mini_batches', type=int, default=8)
    parser.add_argument('--update_epochs', type=int, default=3)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--norm_adv', type=bool, default=True)
    parser.add_argument('--clip_value_loss', type=bool, default=True)
    parser.add_argument('--c_1', type=float, default=0.5)
    parser.add_argument('--c_2', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--epsilon', type=float, default=0.2)
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_mini_batches)
    args.num_iterations = int(args.total_time_steps // args.batch_size)
    return args


def make_env(envs, gamma):
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs['rgb'])
    envs.is_vector_env = True
    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    envs = gym.wrappers.NormalizeReward(envs, gamma=gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))
    return envs


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.encoder = build_encoder(latent_dim)

    def forward(self, x):
        return self.encoder(x.permute((0, 3, 1, 2)) / 255.0)  # [B, H, W, C] -> [B, C, H, W]

    def save(self, path):
        torch.save(self.encoder.state_dict(), path)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.single_observation_shape = envs.single_observation_space.shape
        self.encoder = Encoder(256)
        self.actor = layer_init(nn.Linear(256, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        return self.critic(self.encoder(x))

    def get_action_and_value(self, x, a=None):
        hidden = self.encoder(x)
        actor_output = self.actor(hidden)
        distribution = Categorical(logits=actor_output)
        if a is None:
            a = distribution.sample()
        return a, distribution.log_prob(a), distribution.entropy(), self.critic(hidden)


def main(env_id, seed):
    args = get_args()
    args.env_id = env_id
    args.seed = seed
    run_name = 'ppo_seed_' + str(args.seed)

    # Save training logs
    path_string = str(args.env_id) + '/' + run_name
    writer = SummaryWriter(path_string)
    writer.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()])),
    )

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Initialize environments
    envs = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        rand_seed=args.seed,
        num_levels=500,
        start_level=0,
        distribution_mode='hard'
    )
    envs = make_env(envs, args.gamma)
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space['rgb']
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only discrete action space is supported'

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Initialize buffer
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    log_probs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Data collection
    global_step = 0
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    return_list = []
    update_index = 1

    for iteration in tqdm(range(1, args.num_iterations + 1)):

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # Compute the logarithm of the action probability output by the old policy network
            with torch.no_grad():
                action, log_prob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            log_probs[step] = log_prob

            # Update the environments
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            for item in info:
                if 'episode' in item.keys():
                    if iteration == update_index:
                        return_list.append(item['episode']['r'])
                    else:
                        writer.add_scalar(
                            'This is for plotting/return_train', np.mean(return_list), update_index
                        )
                        return_list.clear()
                        return_list.append(item['episode']['r'])
                        update_index += 1

        # Use GAE (Generalized Advantage Estimation) technique to estimate the advantage function
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

        # ---------------------- We have collected enough data, now let's start training ---------------------- #
        # Flatten each batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape(-1)
        b_log_probs = log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Update the policy network and value network
        b_index = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_index)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                    b_obs[mb_index], b_actions.long()[mb_index]
                )

                # Ratio
                log_ratio = new_log_prob - b_log_probs[mb_index]
                ratios = log_ratio.exp()

                # Advantage normalization
                mb_advantages = b_advantages[mb_index]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-12)

                # Policy loss
                policy_loss_1 = -mb_advantages * ratios
                policy_loss_2 = -mb_advantages * torch.clamp(ratios, 1 - args.epsilon, 1 + args.epsilon)
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss
                new_value = new_value.view(-1)
                if args.clip_value_loss:
                    value_loss_un_clipped = (new_value - b_returns[mb_index]) ** 2
                    value_clipped = b_values[mb_index] + torch.clamp(
                        new_value - b_values[mb_index],
                        -args.epsilon,
                        args.epsilon
                    )
                    value_loss_clipped = (value_clipped - b_returns[mb_index]) ** 2
                    value_loss_max = torch.max(value_loss_un_clipped, value_loss_clipped)
                    value_loss = 0.5 * value_loss_max.mean()
                else:
                    value_loss = 0.5 * ((new_value - b_returns[mb_index]) ** 2).mean()

                # Policy entropy
                entropy_loss = entropy.mean()

                # Total loss
                loss = policy_loss + value_loss * args.c_1 - entropy_loss * args.c_2

                # Save the data during the training process
                writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
                writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
                writer.add_scalar('losses/entropy', entropy_loss.item(), global_step)

                # Update network parameters
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

        # Evaluation
        eval_envs = ProcgenEnv(
            num_envs=args.num_eval_workers,
            env_name=args.env_id,
            rand_seed=42,  # different to training seed
            num_levels=0,
            start_level=0,
            distribution_mode='hard'
        )
        eval_envs = make_env(eval_envs, args.gamma)
        eval_next_obs = torch.Tensor(eval_envs.reset()).to(device)
        eval_episodic_returns = []

        while len(eval_episodic_returns) < 10:
            with torch.no_grad():
                eval_action, _, _, _ = agent.get_action_and_value(eval_next_obs)
            eval_next_obs, _, _, eval_info = eval_envs.step(eval_action.cpu().numpy())
            eval_next_obs = torch.Tensor(eval_next_obs).to(device)
            for item in eval_info:
                if 'episode' in item.keys():
                    eval_episodic_returns.append(item['episode']['r'])

        writer.add_scalar(
            'This is for plotting/return_eval', np.mean(eval_episodic_returns), iteration
        )
        eval_envs.close()

        torch.save(agent.encoder.encoder.state_dict(), './' + env_id + '/encoder_ppo_seed_' + str(seed) + '.pth')
        torch.save(agent.actor.state_dict(), './' + env_id + '/actor_ppo_seed_' + str(seed) + '.pth')

    envs.close()
    writer.close()


def run():
    for env_id in [
        'bigfish',
        'bossfight',
        'caveflyer',
        'chaser',
        'climber',
        'coinrun',
        'dodgeball',
        'fruitbot',
        'heist',
        'jumper',
        'leaper',
        'maze',
        'miner',
        'ninja',
        'plunder',
        'starpilot'
    ]:
        for seed in [1, 2, 3]:
            print(env_id, 'seed:', seed)
            main(env_id, seed)


if __name__ == '__main__':
    run()
