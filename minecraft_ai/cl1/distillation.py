"""CL1 -> PyTorch model distillation pipeline.

Takes training session data (observations, spike responses) from CL1
hardware and trains a lightweight PyTorch MLP to replicate the
CL1's input-output mapping for runtime use without hardware.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple

from ..config import DistillationConfig
from ..networks.distilled_model import DistilledModel
from .training_session import SessionResult

logger = logging.getLogger(__name__)


class CL1Distiller:
    """Distills CL1 spike patterns into a PyTorch model."""

    def __init__(self, config: DistillationConfig, obs_dim: int = 59, num_actions: int = 9):
        self.config = config
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.student: Optional[DistilledModel] = None
        self._training_history: list = []

    def build_student(self) -> DistilledModel:
        """Create the student model for distillation."""
        self.student = DistilledModel(
            obs_dim=self.obs_dim,
            hidden_dims=self.config.student_hidden_dims,
            num_actions=self.num_actions,
        )
        return self.student

    def prepare_targets(self, session: SessionResult) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert CL1 session data into training targets.

        Maps spike counts across action channel groups into
        soft action probability targets via temperature-scaled softmax.
        """
        obs, spikes, rewards = session.get_arrays()

        # Action channels are indices 1-9 of the 10 channel groups
        # (index 0 is the encoding/input channel)
        action_spikes = spikes[:, 1:self.num_actions + 1]  # (N, 9)

        # Temperature-scaled softmax to get soft targets
        scaled = action_spikes / max(self.config.temperature, 1e-8)
        shifted = scaled - np.max(scaled, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        soft_targets = exp_vals / (np.sum(exp_vals, axis=1, keepdims=True) + 1e-8)

        obs_tensor = torch.from_numpy(obs).float()
        target_tensor = torch.from_numpy(soft_targets).float()

        return obs_tensor, target_tensor

    def distill(self, session: SessionResult) -> DistilledModel:
        """Run the full distillation pipeline.

        Args:
            session: Completed CL1 training session with recorded data

        Returns:
            Trained DistilledModel ready for runtime use
        """
        if session.num_samples < self.config.min_samples:
            raise ValueError(
                f"Not enough samples: {session.num_samples} < {self.config.min_samples}"
            )

        if self.student is None:
            self.build_student()

        obs_tensor, target_tensor = self.prepare_targets(session)

        dataset = TensorDataset(obs_tensor, target_tensor)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = optim.Adam(self.student.parameters(), lr=self.config.learning_rate)
        kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
        ce_loss_fn = nn.CrossEntropyLoss()

        self.student.train()
        self._training_history = []

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for obs_batch, target_batch in loader:
                logits = self.student(obs_batch)

                # KL divergence loss (student log-probs vs teacher soft targets)
                log_probs = torch.log_softmax(logits / self.config.temperature, dim=-1)
                kl_loss = kl_loss_fn(log_probs, target_batch) * (self.config.temperature ** 2)

                # Cross-entropy loss against hard labels
                hard_labels = target_batch.argmax(dim=-1)
                ce_loss = ce_loss_fn(logits, hard_labels)

                loss = self.config.kl_weight * kl_loss + self.config.ce_weight * ce_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / max(num_batches, 1)
            self._training_history.append(avg_loss)

            if (epoch + 1) % 20 == 0:
                logger.info(f"Distillation epoch {epoch + 1}/{self.config.num_epochs}: loss={avg_loss:.4f}")

        self.student.eval()
        logger.info(f"Distillation complete: final_loss={self._training_history[-1]:.4f}")
        return self.student

    def save_student(self, path: str):
        """Save the distilled student model."""
        if self.student is None:
            raise RuntimeError("No student model to save")
        torch.save(self.student.state_dict(), path)
        logger.info(f"Student model saved to {path}")

    def load_student(self, path: str) -> DistilledModel:
        """Load a previously distilled student model."""
        if self.student is None:
            self.build_student()
        self.student.load_state_dict(torch.load(path, weights_only=True))
        self.student.eval()
        return self.student
