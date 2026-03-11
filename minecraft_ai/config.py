"""Configuration dataclasses for Minecraft CL1/Izhikevich AI system."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CL1Config:
    """CL1 hardware interface configuration."""
    host: str = "127.0.0.1"
    port: int = 5001
    num_electrodes: int = 64
    usable_electrodes: int = 59
    dead_electrodes: List[int] = field(default_factory=lambda: [0, 4, 7, 56, 63])
    stimulation_rate_hz: int = 10
    max_amplitude_mv: float = 800.0  # used for UDP mode encoding
    min_amplitude_mv: float = 50.0   # used for UDP mode encoding
    max_amplitude_ua: float = 5.0    # hardware limit in microamps
    pulse_width_us: float = 200.0
    num_channel_groups: int = 10
    recording_timeout_ms: float = 100.0
    # Cloud connection settings
    cloud_host: str = "cl1-2544-144.device.cloud.corticallabs-test.com"
    cloud_read_duration_ms: float = 30.0


@dataclass
class IzhikevichConfig:
    """Izhikevich spiking neural network configuration."""
    dt: float = 0.5  # ms per simulation step
    steps_per_tick: int = 20  # simulation steps per game tick
    excitatory_ratio: float = 0.8
    base_current: float = 5.0
    noise_amplitude: float = 2.0
    synapse_delay_ms: float = 1.0
    max_weight: float = 10.0
    min_weight: float = 0.0


@dataclass
class STDPConfig:
    """Reward-modulated STDP parameters."""
    tau_plus: float = 20.0  # ms, potentiation time constant
    tau_minus: float = 20.0  # ms, depression time constant
    a_plus: float = 0.01  # potentiation amplitude
    a_minus: float = 0.012  # depression amplitude (slightly stronger)
    tau_eligibility: float = 1000.0  # ms, eligibility trace decay
    reward_learning_rate: float = 0.01
    weight_decay: float = 0.0001


@dataclass
class NetworkLayerConfig:
    """Configuration for a single SNN layer."""
    num_neurons: int = 100
    excitatory_ratio: float = 0.8
    connection_prob: float = 0.1
    initial_weight_mean: float = 0.5
    initial_weight_std: float = 0.2


@dataclass
class SNNArchConfig:
    """Multi-layer SNN architecture configuration."""
    input_layer: NetworkLayerConfig = field(
        default_factory=lambda: NetworkLayerConfig(num_neurons=59, excitatory_ratio=1.0)
    )
    hidden_layers: List[NetworkLayerConfig] = field(
        default_factory=lambda: [
            NetworkLayerConfig(num_neurons=200, excitatory_ratio=0.8, connection_prob=0.15),
            NetworkLayerConfig(num_neurons=100, excitatory_ratio=0.8, connection_prob=0.1),
        ]
    )
    output_layer: NetworkLayerConfig = field(
        default_factory=lambda: NetworkLayerConfig(num_neurons=48, excitatory_ratio=1.0)
    )
    inter_layer_conn_prob: float = 0.2


@dataclass
class EncoderConfig:
    """Observation encoder network configuration."""
    obs_dim: int = 59
    hidden_dim: int = 128
    num_channel_groups: int = 10
    electrodes_per_group: List[int] = field(
        default_factory=lambda: [8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 3, 3]
    )


@dataclass
class DecoderConfig:
    """Spike-to-action decoder configuration."""
    spike_input_dim: int = 48
    hidden_dim: int = 64
    num_actions: int = 9


@dataclass
class PPOConfig:
    """PPO training hyperparameters."""
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    num_epochs: int = 4
    batch_size: int = 64
    rollout_length: int = 128


@dataclass
class DistillationConfig:
    """CL1 -> PyTorch distillation parameters."""
    student_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    temperature: float = 2.0
    learning_rate: float = 1e-3
    num_epochs: int = 100
    batch_size: int = 32
    min_samples: int = 500
    kl_weight: float = 0.7
    ce_weight: float = 0.3


@dataclass
class BotServerConfig:
    """Node.js bot server connection config."""
    host: str = "127.0.0.1"
    http_port: int = 3002
    ws_port: int = 3002
    mc_host: str = "127.0.0.1"
    mc_port: int = 64418
    mc_version: str = "1.20.4"


@dataclass
class RewardConfig:
    """Reward shaping weights per profile."""
    survival_weight: float = 1.0
    combat_weight: float = 1.0
    resource_weight: float = 0.5
    exploration_weight: float = 0.3
    death_penalty: float = -10.0
    kill_reward: float = 5.0
    damage_dealt_scale: float = 0.5
    damage_taken_scale: float = -0.3
    food_reward: float = 1.0
    block_break_reward: float = 0.1


@dataclass
class GameConfig:
    """Main game loop configuration."""
    tick_rate_hz: float = 10.0
    max_entities: int = 32
    observation_dim: int = 59
    num_actions: int = 9
    thread_pool_size: int = 8
    entity_despawn_timeout_s: float = 30.0
    state_history_len: int = 10


@dataclass
class MinecraftCL1Config:
    """Top-level configuration combining all subsystems."""
    cl1: CL1Config = field(default_factory=CL1Config)
    izhikevich: IzhikevichConfig = field(default_factory=IzhikevichConfig)
    stdp: STDPConfig = field(default_factory=STDPConfig)
    snn_arch: SNNArchConfig = field(default_factory=SNNArchConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    bot_server: BotServerConfig = field(default_factory=BotServerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    game: GameConfig = field(default_factory=GameConfig)
