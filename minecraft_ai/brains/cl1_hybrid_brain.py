"""CL1/AI hybrid brain for non-hostile entities.

Two operating modes:
1. Training mode: Observations -> CL1 hardware -> spike responses -> actions
   Collects data for later distillation
2. Runtime mode: Observations -> distilled PyTorch MLP -> actions
   Uses a lightweight model distilled from CL1 training sessions
"""

import logging
import numpy as np
import torch
from typing import Any, Dict, Optional

from .base_brain import BaseBrain, BrainAction, BrainObservation
from ..config import CL1Config, DistillationConfig, EncoderConfig, DecoderConfig
from ..networks.mc_encoder import MCEncoder
from ..networks.mc_decoder import MCDecoder, ChannelGroupDecoder
from ..networks.distilled_model import DistilledModel
from ..cl1.mc_cl1_interface import MCCL1Interface
from ..cl1.channel_mapping import NUM_CHANNEL_GROUPS

logger = logging.getLogger(__name__)


class CL1HybridBrain(BaseBrain):
    """Brain combining CL1 biological hardware with PyTorch AI.

    For non-hostile entities (players, villagers, passive mobs).
    Uses CL1 hardware during short training sessions, then switches
    to a distilled PyTorch model for runtime inference.
    """

    def __init__(
        self,
        entity_type: str,
        entity_id: int,
        cl1_config: Optional[CL1Config] = None,
        encoder_config: Optional[EncoderConfig] = None,
        decoder_config: Optional[DecoderConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
    ):
        super().__init__(entity_type, entity_id)
        self.cl1_config = cl1_config or CL1Config()
        self.encoder_config = encoder_config or EncoderConfig()
        self.decoder_config = decoder_config or DecoderConfig()

        # Neural network components
        self.encoder = MCEncoder(self.encoder_config)
        self.group_decoder = ChannelGroupDecoder(
            num_groups=NUM_CHANNEL_GROUPS,
            num_actions=self.decoder_config.num_actions,
        )

        # CL1 hardware interface (None until connected)
        self.cl1_interface: Optional[MCCL1Interface] = None

        # Distilled model for runtime (None until distillation)
        self.distilled_model: Optional[DistilledModel] = None

        # Operating mode
        self._use_distilled = False
        self._training_mode = False
        self._pending_reward = 0.0
        self._last_obs: Optional[np.ndarray] = None

    def connect_cl1(self, interface: MCCL1Interface):
        """Attach CL1 hardware interface for training mode."""
        self.cl1_interface = interface
        logger.info(f"CL1 connected for {self.entity_type}:{self.entity_id}")

    def set_distilled_model(self, model: DistilledModel):
        """Set the distilled model for runtime mode."""
        self.distilled_model = model
        self._use_distilled = True
        logger.info(f"Distilled model set for {self.entity_type}:{self.entity_id}")

    def set_training_mode(self, training: bool):
        """Toggle between training (CL1) and runtime (distilled) mode."""
        self._training_mode = training
        if training:
            self._use_distilled = False

    def act(self, observation: BrainObservation) -> BrainAction:
        """Choose an action using either CL1 hardware or distilled model."""
        obs = observation.obs_vector
        self._last_obs = obs.copy()

        if self._use_distilled and self.distilled_model is not None:
            return self._act_distilled(obs)
        elif self.cl1_interface is not None and self.cl1_interface.is_connected:
            return self._act_cl1(obs)
        else:
            return self._act_encoder_decoder(obs)

    def _act_cl1(self, obs: np.ndarray) -> BrainAction:
        """Use CL1 hardware for action selection."""
        # Encode observation to stimulation params
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
        with torch.no_grad():
            frequencies, amplitudes = self.encoder(obs_tensor)

        freq_np = frequencies.squeeze(0).numpy()
        amp_np = amplitudes.squeeze(0).numpy()

        # Send to CL1 and read response
        spike_counts = self.cl1_interface.stimulate_and_read(freq_np, amp_np)

        if spike_counts is None:
            spike_counts = np.zeros(NUM_CHANNEL_GROUPS, dtype=np.float32)

        # Decode spike counts to action
        spike_tensor = torch.from_numpy(spike_counts).float().unsqueeze(0)
        with torch.no_grad():
            logits = self.group_decoder(spike_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        action_idx = int(np.argmax(probs))

        return BrainAction(
            action_idx=action_idx,
            action_probs=probs,
            metadata={"mode": "cl1", "spike_counts": spike_counts.tolist()},
        )

    def _act_distilled(self, obs: np.ndarray) -> BrainAction:
        """Use distilled PyTorch model for action selection."""
        action_idx, probs = self.distilled_model.get_action_from_numpy(obs)

        return BrainAction(
            action_idx=action_idx,
            action_probs=probs,
            metadata={"mode": "distilled"},
        )

    def _act_encoder_decoder(self, obs: np.ndarray) -> BrainAction:
        """Fallback: use encoder-decoder without CL1 hardware."""
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        with torch.no_grad():
            frequencies, amplitudes = self.encoder(obs_tensor)
            # Simulate spike response as normalized frequencies
            pseudo_spikes = frequencies / 50.0  # normalize to [0,1]
            logits = self.group_decoder(pseudo_spikes)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()

        action_idx = int(np.argmax(probs))

        return BrainAction(
            action_idx=action_idx,
            action_probs=probs,
            metadata={"mode": "encoder_decoder_fallback"},
        )

    def learn(self, reward: float) -> Dict[str, float]:
        """Accumulate reward (actual learning happens in training pipeline)."""
        self._pending_reward += reward
        self._total_reward += reward

        if self.cl1_interface is not None and self.cl1_interface.is_connected:
            self.cl1_interface.send_reward(reward)

        return {"pending_reward": self._pending_reward}

    def reset(self):
        """Reset brain state."""
        self._pending_reward = 0.0
        self._step_count = 0
        self._total_reward = 0.0
        self._last_obs = None

    def save(self, path: str):
        """Save distilled model and encoder/decoder state."""
        state = {
            "encoder": self.encoder.state_dict(),
            "group_decoder": self.group_decoder.state_dict(),
        }
        if self.distilled_model is not None:
            state["distilled_model"] = self.distilled_model.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        """Load saved state."""
        state = torch.load(path, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        self.group_decoder.load_state_dict(state["group_decoder"])
        if "distilled_model" in state and self.distilled_model is not None:
            self.distilled_model.load_state_dict(state["distilled_model"])

    def get_stats(self) -> Dict[str, Any]:
        base = super().get_stats()
        base["mode"] = "distilled" if self._use_distilled else "cl1" if self._training_mode else "fallback"
        base["has_distilled_model"] = self.distilled_model is not None
        base["cl1_connected"] = (
            self.cl1_interface is not None and self.cl1_interface.is_connected
        )
        return base
