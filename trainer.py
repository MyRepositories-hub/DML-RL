import numpy as np
import torch
import torch.nn as nn


class Trainer:
    def __init__(self, args, optimizer, agent_1, agent_2):
        self.args = args
        self.optimizer = optimizer
        self.agent_1, self.agent_2 = agent_1, agent_2

    def train(self):
        b_states_1, b_actions_1, b_log_probs_1, b_values_1, b_advantages_1, b_returns_1 = self.agent_1.get_all_data()
        b_states_2, b_actions_2, b_log_probs_2, b_values_2, b_advantages_2, b_returns_2 = self.agent_2.get_all_data()
        b_index = np.arange(self.args.batch_size)

        for epoch in range(self.args.update_epochs):
            np.random.shuffle(b_index)

            for start in range(0, self.args.batch_size, self.args.minibatch_size):
                end = start + self.args.minibatch_size
                mb_index = b_index[start:end]

                # The latest outputs of the policy network and value network
                _, new_log_prob_1, new_entropy_1, new_value_1, probs_1_1 = self.agent_1.get_action_and_value(
                    b_states_1[mb_index], b_actions_1[mb_index], show_all=True
                )
                _, new_log_prob_2, new_entropy_2, new_value_2, probs_2_2 = self.agent_2.get_action_and_value(
                    b_states_2[mb_index], b_actions_2[mb_index], show_all=True
                )

                # Get probs
                probs_1_2 = self.agent_2.get_probs(b_states_1[mb_index])
                probs_2_1 = self.agent_1.get_probs(b_states_2[mb_index])

                def compute_kld(p, q):
                    return torch.sum(p * ((p + 1e-8).log() - (q + 1e-8).log()), -1)

                kl_1 = compute_kld(probs_1_2, probs_1_1).mean()
                kl_2 = compute_kld(probs_2_1, probs_2_2).mean()
                self.agent_1.writer.add_scalar('losses/kl_divergence', kl_1.item(), self.agent_1.global_step)
                self.agent_2.writer.add_scalar('losses/kl_divergence', kl_2.item(), self.agent_2.global_step)

                # Probability ratio
                ratios_1 = (new_log_prob_1 - b_log_probs_1[mb_index]).exp()
                ratios_2 = (new_log_prob_2 - b_log_probs_2[mb_index]).exp()

                # Advantage normalization
                mb_advantages_1 = b_advantages_1[mb_index]
                mb_advantages_2 = b_advantages_2[mb_index]
                if self.args.norm_adv:
                    mb_advantages_1 = (mb_advantages_1 - mb_advantages_1.mean()) / (mb_advantages_1.std() + 1e-8)
                    mb_advantages_2 = (mb_advantages_2 - mb_advantages_2.mean()) / (mb_advantages_2.std() + 1e-8)

                # Policy loss
                policy_loss_1 = self.compute_policy_loss(ratios_1, mb_advantages_1)
                policy_loss_2 = self.compute_policy_loss(ratios_2, mb_advantages_2)

                # Value loss
                value_loss_1 = self.compute_value_loss(new_value_1, b_returns_1[mb_index], b_values_1[mb_index])
                value_loss_2 = self.compute_value_loss(new_value_2, b_returns_2[mb_index], b_values_2[mb_index])

                # Policy entropy
                entropy_loss_1 = new_entropy_1.mean()
                entropy_loss_2 = new_entropy_2.mean()

                # Total loss
                loss_1 = policy_loss_1 + value_loss_1 * self.args.c_1 - entropy_loss_1 * self.args.c_2 + kl_1 * self.args.alpha
                loss_2 = policy_loss_2 + value_loss_2 * self.args.c_1 - entropy_loss_2 * self.args.c_2 + kl_2 * self.args.alpha

                loss = 0.5 * (loss_1 + loss_2)

                # Save the data during the training process
                self.agent_1.writer.add_scalar('losses/policy_loss', policy_loss_1.item(), self.agent_1.global_step)
                self.agent_1.writer.add_scalar('losses/value_loss', value_loss_1.item(), self.agent_1.global_step)
                self.agent_1.writer.add_scalar('losses/entropy', entropy_loss_1.item(), self.agent_1.global_step)
                self.agent_2.writer.add_scalar('losses/policy_loss', policy_loss_2.item(), self.agent_2.global_step)
                self.agent_2.writer.add_scalar('losses/value_loss', value_loss_2.item(), self.agent_2.global_step)
                self.agent_2.writer.add_scalar('losses/entropy', entropy_loss_2.item(), self.agent_2.global_step)

                # Update network parameters
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent_1.parameters(), self.args.max_grad_norm)
                nn.utils.clip_grad_norm_(self.agent_2.parameters(), self.args.max_grad_norm)
                self.optimizer.step()

    def compute_value_loss(self, new_value, mb_returns, mb_values):
        new_value = new_value.view(-1)
        if self.args.clip_value_loss:
            value_loss_un_clipped = (new_value - mb_returns) ** 2
            value_clipped = mb_values + torch.clamp(new_value - mb_values, -self.args.epsilon, self.args.epsilon)
            value_loss_clipped = (value_clipped - mb_returns) ** 2
            value_loss_max = torch.max(value_loss_un_clipped, value_loss_clipped)
            value_loss = 0.5 * value_loss_max.mean()
        else:
            value_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()

        return value_loss

    def compute_policy_loss(self, ratios, mb_advantages):
        policy_loss_1 = mb_advantages * ratios
        policy_loss_2 = mb_advantages * torch.clamp(
            ratios, 1 - self.args.epsilon, 1 + self.args.epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        return policy_loss
