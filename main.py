import argparse
import random

import gym
import numpy as np
import torch
import torch.optim as optim
from procgen import ProcgenEnv
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from agent import Agent
from trainer import Trainer


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
    parser.add_argument('--alpha', type=float, default=1.0)
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


def main(env_id, seed):
    args = get_args()
    args.env_id = env_id
    args.seed = seed

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # Initialize environments
    envs_1 = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        rand_seed=args.seed,
        num_levels=500,
        start_level=0,
        distribution_mode='hard'
    )
    envs_1 = make_env(envs_1, args.gamma)
    envs_1.single_action_space = envs_1.action_space
    envs_1.single_observation_space = envs_1.observation_space['rgb']
    envs_2 = ProcgenEnv(
        num_envs=args.num_envs,
        env_name=args.env_id,
        rand_seed=args.seed,
        num_levels=500,
        start_level=0,
        distribution_mode='hard'
    )
    envs_2 = make_env(envs_2, args.gamma)
    envs_2.single_action_space = envs_2.action_space
    envs_2.single_observation_space = envs_2.observation_space['rgb']

    # Initialize writers
    writer_1 = SummaryWriter(str(args.env_id) + '/' + 'agent_1_seed_' + str(args.seed))
    writer_1.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()]))
    )
    writer_2 = SummaryWriter(str(args.env_id) + '/' + 'agent_2_seed_' + str(args.seed))
    writer_2.add_text(
        'Hyperparameter',
        '|param|value|\n|-|-|\n%s' % ('\n'.join([f'|{key}|{value}|' for key, value in vars(args).items()]))
    )

    # Agents and optimizer
    agent_1 = Agent(args, envs_1, writer_1, device).to(device)
    agent_2 = Agent(args, envs_2, writer_2, device).to(device)
    optimizer = optim.Adam(list(agent_1.parameters()) + list(agent_2.parameters()), lr=args.learning_rate, eps=1e-5)
    trainer = Trainer(args, optimizer, agent_1, agent_2)

    # Data collection
    for iteration in tqdm(range(1, args.num_iterations + 1)):
        agent_1.interaction(iteration)
        agent_2.interaction(iteration)
        trainer.train()

        eval_envs = ProcgenEnv(
            num_envs=args.num_eval_workers,
            env_name=args.env_id,
            rand_seed=42,  # different to training seed
            num_levels=0,
            start_level=0,
            distribution_mode='hard'
        )
        eval_envs = make_env(eval_envs, args.gamma)
        agent_1.evaluation(eval_envs, iteration)
        eval_envs.close()

        eval_envs = ProcgenEnv(
            num_envs=args.num_eval_workers,
            env_name=args.env_id,
            rand_seed=42,  # different to training seed
            num_levels=0,
            start_level=0,
            distribution_mode='hard'
        )
        eval_envs = make_env(eval_envs, args.gamma)
        agent_2.evaluation(eval_envs, iteration)
        eval_envs.close()

        torch.save(agent_1.encoder.encoder.state_dict(), './' + env_id + '/encoder_agent_1_seed_' + str(seed) + '.pth')
        torch.save(agent_1.actor.state_dict(), './' + env_id + '/actor_agent_1_seed_' + str(seed) + '.pth')

        torch.save(agent_2.encoder.encoder.state_dict(), './' + env_id + '/encoder_agent_2_seed_' + str(seed) + '.pth')
        torch.save(agent_2.actor.state_dict(), './' + env_id + '/actor_agent_2_seed_' + str(seed) + '.pth')

    envs_1.close()
    envs_2.close()
    writer_1.close()
    writer_2.close()


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
