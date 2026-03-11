"""Knowledge distillation pipeline: CL1 session -> distilled model -> PPO fine-tune."""

import logging
from typing import Optional

from ..config import MinecraftCL1Config
from ..networks.distilled_model import DistilledModel
from ..networks.mc_value_net import MCValueNet
from ..cl1.distillation import CL1Distiller
from ..cl1.training_session import SessionResult
from .ppo_trainer import PPOTrainer

logger = logging.getLogger(__name__)


class DistillationPipeline:
    """Full pipeline: CL1 training session -> distilled model -> optional PPO.

    Steps:
    1. CL1 hardware training session produces (obs, spike, reward) data
    2. Distill spike patterns into a lightweight PyTorch MLP
    3. Optionally fine-tune the distilled model with PPO
    """

    def __init__(self, config: Optional[MinecraftCL1Config] = None):
        self.config = config or MinecraftCL1Config()
        self.distiller = CL1Distiller(
            config=self.config.distillation,
            obs_dim=self.config.game.observation_dim,
            num_actions=self.config.game.num_actions,
        )
        self.model: Optional[DistilledModel] = None
        self.ppo_trainer: Optional[PPOTrainer] = None

    def distill_from_session(self, session: SessionResult) -> DistilledModel:
        """Run distillation on a completed CL1 training session.

        Args:
            session: Completed CL1 session with recorded data

        Returns:
            Trained DistilledModel
        """
        logger.info(
            f"Starting distillation: {session.num_samples} samples, "
            f"{session.duration_seconds:.1f}s session"
        )

        self.model = self.distiller.distill(session)
        return self.model

    def setup_ppo_finetuning(self) -> PPOTrainer:
        """Set up PPO trainer for fine-tuning the distilled model.

        Returns:
            PPOTrainer ready for training
        """
        if self.model is None:
            raise RuntimeError("Must distill a model first")

        value_net = MCValueNet(
            obs_dim=self.config.game.observation_dim,
            hidden_dim=128,
        )

        self.ppo_trainer = PPOTrainer(
            policy=self.model,
            value_net=value_net,
            config=self.config.ppo,
        )

        logger.info("PPO fine-tuning trainer initialized")
        return self.ppo_trainer

    def save_model(self, path: str):
        """Save the current model (distilled or fine-tuned)."""
        if self.model is None:
            raise RuntimeError("No model to save")
        self.distiller.save_student(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> DistilledModel:
        """Load a previously saved model."""
        self.model = self.distiller.load_student(path)
        return self.model
