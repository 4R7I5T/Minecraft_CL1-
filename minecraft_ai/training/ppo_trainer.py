"""PPO training for CL1 hybrid brain fine-tuning."""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional

from ..config import PPOConfig
from ..networks.distilled_model import DistilledModel
from ..networks.mc_value_net import MCValueNet
from .replay_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Proximal Policy Optimization trainer for distilled models.

    Fine-tunes distilled CL1 models using PPO for improved
    performance beyond what CL1 training alone achieves.
    """

    def __init__(
        self,
        policy: DistilledModel,
        value_net: MCValueNet,
        config: Optional[PPOConfig] = None,
    ):
        self.policy = policy
        self.value_net = value_net
        self.config = config or PPOConfig()

        self.policy_optimizer = optim.Adam(
            policy.parameters(), lr=self.config.learning_rate
        )
        self.value_optimizer = optim.Adam(
            value_net.parameters(), lr=self.config.learning_rate
        )

        self.rollout = RolloutBuffer(
            rollout_length=self.config.rollout_length,
            obs_dim=policy.obs_dim,
        )

        self._update_count = 0
        self._total_loss_history: list = []

    def get_action(self, obs: np.ndarray) -> tuple:
        """Sample action from policy with log probability.

        Returns:
            (action_idx, log_prob, value_estimate, action_probs)
        """
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(obs_tensor)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.value_net(obs_tensor)

        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
            probs.squeeze(0).numpy(),
        )

    def update(self) -> Dict[str, float]:
        """Run PPO update on collected rollout data.

        Returns:
            Dictionary of training metrics
        """
        if not self.rollout.is_full:
            return {}

        # Compute GAE advantages
        last_obs = self.rollout._experiences[-1].next_observation
        with torch.no_grad():
            last_value = self.value_net(
                torch.from_numpy(last_obs).float().unsqueeze(0)
            ).item()

        self.rollout.compute_gae(
            gamma=self.config.gamma,
            lam=self.config.gae_lambda,
            last_value=last_value,
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.config.num_epochs):
            batches = self.rollout.get_batches(self.config.batch_size)

            for batch in batches:
                obs = torch.from_numpy(batch["observations"]).float()
                actions = torch.from_numpy(batch["actions"]).long()
                old_log_probs = torch.from_numpy(batch["old_log_probs"]).float()
                advantages = torch.from_numpy(batch["advantages"]).float()
                returns = torch.from_numpy(batch["returns"]).float()

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Policy loss
                logits = self.policy(obs)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages

                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = self.value_net(obs).squeeze(-1)
                value_loss = nn.functional.mse_loss(values, returns)

                # Combined loss
                loss = (
                    policy_loss
                    + self.config.value_loss_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                # Update policy
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.value_net.parameters(), self.config.max_grad_norm
                )

                self.policy_optimizer.step()
                self.value_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1

        self.rollout.clear()
        self._update_count += 1

        metrics = {
            "policy_loss": total_policy_loss / max(num_updates, 1),
            "value_loss": total_value_loss / max(num_updates, 1),
            "entropy": total_entropy / max(num_updates, 1),
            "update_count": self._update_count,
        }
        self._total_loss_history.append(metrics["policy_loss"])
        return metrics

    def save(self, path: str):
        """Save policy and value network."""
        torch.save({
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "update_count": self._update_count,
        }, path)

    def load(self, path: str):
        """Load saved training state."""
        checkpoint = torch.load(path, weights_only=True)
        self.policy.load_state_dict(checkpoint["policy"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer"])
        self._update_count = checkpoint.get("update_count", 0)
