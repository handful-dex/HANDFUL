from collections import defaultdict
from dataclasses import dataclass
import os
import random
import time
from typing import Optional

import tqdm

from mani_skill.utils import gym_utils
from mani_skill.utils.registration import TimeLimitWrapper
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from envs.tasks.xarm7_leap_pick_env import XArm7TableTop
from envs.tasks.xarm7_leap_push_env import XArm7TableTopPush
from envs.tasks.xarm7_press_button import XArm7TableTopPress
from envs.tasks.xarm7_pick_randomized import XArm7TableTopPickRandomized
from envs.tasks.xarm7_pick_all import XArm7TableTopPickAll
from envs.tasks.xarm7_two_pick import XArm7TableTopTwoPick
from envs.tasks.xarm7_twist import XArm7TableTopKnobTwist
from envs.tasks.xarm7_open_drawer import XArm7CabinetDrawerEnv, XArm7CabinetDoorEnv

from envs.tasks.unified_environments.xarm7_two_pick_unified import XArm7TableTopTwoPickUnified
from envs.tasks.unified_environments.xarm7_push_unified import XArm7TableTopPushUnified
from envs.tasks.unified_environments.xarm7_open_drawer_unified import XArm7CabinetDrawerEnvUnified
from envs.tasks.unified_environments.xarm7_press_button_unified import XArm7TableTopPressUnified
from envs.tasks.unified_environments.xarm7_twist_unified import XArm7TableTopKnobTwistUnified

from envs.tasks.whole_hand_environments.xarm7_push_whole_hand import XArm7TableTopPushWholeHand
from envs.tasks.whole_hand_environments.xarm7_open_drawer_whole_hand import XArm7CabinetDrawerWholeHand
from envs.tasks.whole_hand_environments.xarm7_press_button_whole_hand import XArm7TableTopPressWholeHand
from envs.tasks.whole_hand_environments.xarm7_twist_whole_hand import XArm7TableTopKnobTwistWholeHand
from envs.tasks.whole_hand_environments.xarm7_two_pick_whole_hand import XArm7TableTopTwoPickWholeHand

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tyro

import mani_skill.envs


@dataclass
class Args:
    exp_name: Optional[str] = None
    seed: int = 11
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "ManiSkill"
    wandb_entity: Optional[str] = None
    wandb_group: str = "SAC"
    capture_video: bool = True
    save_trajectory: bool = False
    save_model: bool = True
    evaluate: bool = False
    checkpoint: Optional[str] = None
    log_freq: int = 1_000
    env_id: str = "xArm7-v1-push-unified"
    robot_uids = "xarm7_leap_right"
    env_vectorization: str = "gpu"
    num_envs: int = 512
    num_eval_envs: int = 16
    partial_reset: bool = False
    eval_partial_reset: bool = False
    num_steps: Optional[int] = None
    num_eval_steps: Optional[int] = None
    reconfiguration_freq: Optional[int] = 1
    eval_reconfiguration_freq: Optional[int] = 1
    eval_freq: int = 100000
    save_train_video_freq: Optional[int] = None
    control_mode: Optional[str] = "pd_joint_delta_pos"
    finger_selection: Optional[str] = None
    palm_use: bool = False
    num_active_fingers: int = 2
    state_file_path: Optional[str] = None
    difficulty: Optional[int] = None
    total_timesteps: int = 16_000_000
    buffer_size: int = 4_000_000
    buffer_device: str = "cuda"
    gamma: float = 0.95
    tau: float = 0.005
    batch_size: int = 1024
    learning_starts: int = 51_200
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 1
    target_network_frequency: int = 1
    alpha: float = 0.2
    autotune: bool = True
    training_freq: int = 2048
    utd: float = 0.5
    bootstrap_at_done: str = "always"
    grad_steps_per_iteration: int = 0
    steps_per_env: int = 0


@dataclass
class ReplayBufferSample:
    obs: torch.Tensor
    next_obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(self, env, num_envs, buffer_size, storage_device, sample_device):
        self.buffer_size = buffer_size
        self.pos = 0
        self.full = False
        self.num_envs = num_envs
        self.storage_device = storage_device
        self.sample_device = sample_device
        self.per_env_buffer_size = buffer_size // num_envs
        self.obs      = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        self.next_obs = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_observation_space.shape).to(storage_device)
        self.actions  = torch.zeros((self.per_env_buffer_size, num_envs) + env.single_action_space.shape).to(storage_device)
        self.logprobs = torch.zeros((self.per_env_buffer_size, num_envs)).to(storage_device)
        self.rewards  = torch.zeros((self.per_env_buffer_size, num_envs)).to(storage_device)
        self.dones    = torch.zeros((self.per_env_buffer_size, num_envs)).to(storage_device)
        self.values   = torch.zeros((self.per_env_buffer_size, num_envs)).to(storage_device)

    def add(self, obs, next_obs, action, reward, done):
        if self.storage_device == torch.device("cpu"):
            obs, next_obs, action, reward, done = obs.cpu(), next_obs.cpu(), action.cpu(), reward.cpu(), done.cpu()
        self.obs[self.pos]      = obs
        self.next_obs[self.pos] = next_obs
        self.actions[self.pos]  = action
        self.rewards[self.pos]  = reward
        self.dones[self.pos]    = done
        self.pos += 1
        if self.pos == self.per_env_buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        if self.full:
            batch_inds = torch.randint(0, self.per_env_buffer_size, size=(batch_size,), device=self.storage_device)
        else:
            batch_inds = torch.randint(0, self.pos, size=(batch_size,), device=self.storage_device)
        env_inds = torch.randint(0, self.num_envs, size=(batch_size,), device=self.storage_device)
        return ReplayBufferSample(
            obs=self.obs[batch_inds, env_inds].to(self.sample_device),
            next_obs=self.next_obs[batch_inds, env_inds].to(self.sample_device),
            actions=self.actions[batch_inds, env_inds].to(self.sample_device),
            rewards=self.rewards[batch_inds, env_inds].to(self.sample_device),
            dones=self.dones[batch_inds, env_inds].to(self.sample_device),
        )


class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 512),
            nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1),
        )
    def forward(self, x, a):
        return self.net(torch.cat([x, a], 1))


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 256),
            nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
        )
        self.fc_mean   = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",  torch.tensor((h + l) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_eval_action(self, x):
        mean = self.fc_mean(self.backbone(x))
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def get_action(self, x):
        mean, log_std = self(x)
        std    = log_std.exp()
        normal = torch.distributions.Normal(mean, std, validate_args=False)
        x_t    = normal.rsample()
        y_t    = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias  = self.action_bias.to(device)
        return super().to(device)


class Logger:
    def __init__(self, log_wandb=False, tensorboard=None):
        self.writer    = tensorboard
        self.log_wandb = log_wandb

    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)

    def close(self):
        self.writer.close()


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.grad_steps_per_iteration = int(args.training_freq * args.utd)
    args.steps_per_env = args.training_freq // args.num_envs
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(device)

    env_kwargs = dict(
        obs_mode="state", render_mode="rgb_array",
        sim_backend="gpu", robot_uids=args.robot_uids,
        reward_mode="normalized_dense",
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    if args.finger_selection is not None:
        env_kwargs["finger_selection"] = (
            [int(x) for x in args.finger_selection.split(',')]
            if ',' in args.finger_selection else args.finger_selection
        )
        env_kwargs["num_active_fingers"] = args.num_active_fingers
        env_kwargs["palm_use"] = args.palm_use
    if args.state_file_path is not None:
        env_kwargs["state_file_path"] = args.state_file_path
    if args.difficulty is not None:
        env_kwargs["difficulty"] = args.difficulty

    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1,
                    reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs,
                         reconfiguration_freq=args.eval_reconfiguration_freq,
                         human_render_camera_configs=dict(shader_pack="default"), **env_kwargs)

    if args.num_steps is None:
        args.num_steps = gym_utils.find_max_episode_steps_value(envs)
    if args.num_eval_steps is None:
        args.num_eval_steps = gym_utils.find_max_episode_steps_value(eval_envs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs      = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)

    if args.capture_video or args.save_trajectory:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval trajectories/videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x: (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos",
                                 save_trajectory=False, save_video_trigger=save_video_trigger,
                                 max_steps_per_video=args.num_steps, video_fps=30, info_on_video=True)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir,
                                  save_trajectory=args.save_trajectory, save_video=args.capture_video,
                                  trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps,
                                  video_fps=30, info_on_video=True)

    envs      = ManiSkillVectorEnv(envs,      args.num_envs,      ignore_terminations=not args.partial_reset,      record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)

    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"]      = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id,
                                          reward_mode="normalized_dense", env_horizon=max_episode_steps,
                                          partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id,
                                          reward_mode="normalized_dense", env_horizon=max_episode_steps,
                                          partial_reset=False)
            wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                       sync_tensorboard=False, config=config, name=run_name,
                       save_code=True, group=args.wandb_group, tags=["sac", "walltime_efficient"])
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text("hyperparameters",
                        "|param|value|\n|-|-|\n%s" % "\n".join(
                            [f"|{k}|{v}|" for k, v in vars(args).items()]))
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    actor      = Actor(envs).to(device)
    qf1        = SoftQNetwork(envs).to(device)
    qf2        = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint)
        actor.load_state_dict(ckpt['actor'])
        qf1.load_state_dict(ckpt['qf1'])
        qf2.load_state_dict(ckpt['qf2'])

    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer     = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, capturable=True)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=True)

    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha     = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=True)
    else:
        alpha = args.alpha
        log_alpha = None  # keep reference clean

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        env=envs, num_envs=args.num_envs, buffer_size=args.buffer_size,
        storage_device=torch.device(args.buffer_device), sample_device=device,
    )

    # ── Static tensors for CUDA graph ────────────────────────────────────────

    # CUDA graphs speedup for training. Can fall back to regular train.py if CUDA
    # graphs fails. If I remember correctly, this is around 1000it/s on my 
    # AMD 9950X3D / 5090 setup and the regular training file is around 600it/s


    # CUDA graphs require static memory inputs and outputs since graph replay 
    # executes a fixed set of GPU operations on fixed memory addresses.
    # We allocate static tensors on the target device beforehand. During 
    # the training step, we copy the dynamically sampled batch data from the 
    # replay buffer directly into these static memory spaces before calling replay().
    obs_dim = envs.single_observation_space.shape[0]
    act_dim = envs.single_action_space.shape[0]

    static_obs        = torch.zeros(args.batch_size, obs_dim, device=device)
    static_next_obs   = torch.zeros(args.batch_size, obs_dim, device=device)
    static_actions    = torch.zeros(args.batch_size, act_dim, device=device)
    static_rewards    = torch.zeros(args.batch_size, device=device)
    static_dones      = torch.zeros(args.batch_size, device=device)
    # FIX: shape (1,) so it broadcasts cleanly against (batch, 1) log_pi
    static_alpha      = torch.tensor([alpha], device=device)

    # Static targets to retrieve calculated losses from within the CUDA graph
    static_qf1_loss   = torch.zeros(1, device=device)
    static_qf2_loss   = torch.zeros(1, device=device)
    static_actor_loss = torch.zeros(1, device=device)
    static_alpha_loss = torch.zeros(1, device=device)

    # ── Update functions ──────────────────────────────────────────────────────
    # We break training into discrete functions for Q-networks, Actor, and Target parameter
    # updates. This allows us to compile them into distinct CUDA graphs rather than one monolithic
    # graph, maintaining support for asynchronous update frequencies (e.g., policy_frequency != 1).

    def update_q():
        """
        Q-network parameter update.
        Computes Bellman targets and updates both Q-networks (double Q-learning)
        using static batch inputs.
        """
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.get_action(static_next_obs)
            qf1_next_target = qf1_target(static_next_obs, next_state_actions)
            qf2_next_target = qf2_target(static_next_obs, next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - static_alpha * next_state_log_pi
            next_q_value = static_rewards + (1 - static_dones) * args.gamma * min_qf_next_target.view(-1)

        qf1_a_values = qf1(static_obs, static_actions).view(-1)
        qf2_a_values = qf2(static_obs, static_actions).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss  = qf1_loss + qf2_loss

        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        # Copy calculated losses out to static CPU-accessible memory
        static_qf1_loss.copy_(qf1_loss)
        static_qf2_loss.copy_(qf2_loss)

    def update_actor():
        """
        Actor policy parameter update.
        Computes policy gradient loss and adapts temperature alpha if autotune is enabled.
        Autotuning updates log_alpha to keep policy entropy close to the target threshold.
        """
        pi, log_pi, _ = actor.get_action(static_obs)
        qf1_pi    = qf1(static_obs, pi)
        qf2_pi    = qf2(static_obs, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        actor_loss = ((static_alpha * log_pi) - min_qf_pi).mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        static_actor_loss.copy_(actor_loss)

        if args.autotune:
            with torch.no_grad():
                _, log_pi_alpha, _ = actor.get_action(static_obs)
            alpha_loss = (-log_alpha.exp() * (log_pi_alpha + target_entropy)).mean()

            a_optimizer.zero_grad()
            alpha_loss.backward()
            a_optimizer.step()

            static_alpha_loss.copy_(alpha_loss)

    def update_target():
        """
        Target Q-network exponential moving average (EMA) soft update.
        Slowly updates target networks towards policy networks at rate tau.
        """
        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
            target_param.data.lerp_(param.data, args.tau)
        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
            target_param.data.lerp_(param.data, args.tau)

    # ── Build CUDA graphs ─────────────────────────────────────────────────────
    def build_cuda_graphs():
        """
        Warm up and capture the CUDA graph executables.
        We warm up the CUDA environment using a few dummy forward/backward updates
        to instantiate memory allocators (avoiding dynamic memory allocation inside the graph capture).
        We then record three distinct graphs to support customized update schedules.
        """
        warmup_data = rb.sample(args.batch_size)
        static_obs.copy_(warmup_data.obs)
        static_next_obs.copy_(warmup_data.next_obs)
        static_actions.copy_(warmup_data.actions)
        static_rewards.copy_(warmup_data.rewards.flatten())
        static_dones.copy_(warmup_data.dones.flatten())
        static_alpha.fill_(log_alpha.exp().item() if args.autotune else args.alpha)

        # Warmup phase: executes operations on a custom CUDA stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                update_q()
                update_actor()
                update_target()
        torch.cuda.current_stream().wait_stream(s)

        # Capture Q-network graph
        g_q = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_q):
            update_q()
            
        # Capture Actor graph
        g_actor = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_actor):
            update_actor()
            
        # Capture Target EMA soft-update graph
        g_target = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_target):
            update_target()
            
        return g_q, g_actor, g_target

    cuda_graph = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    obs, info     = envs.reset(seed=args.seed)
    eval_obs, _   = eval_envs.reset(seed=args.seed)
    global_step   = 0
    global_update = 0
    learning_has_started    = False
    global_steps_per_iteration = args.num_envs * args.steps_per_env
    pbar             = tqdm.tqdm(range(args.total_timesteps))
    cumulative_times = defaultdict(float)

    while global_step < args.total_timesteps:

        # ── Evaluation ───────────────────────────────────────────────────────
        if args.eval_freq > 0 and (global_step - args.training_freq) // args.eval_freq < global_step // args.eval_freq:
            actor.eval()
            stime     = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = \
                        eval_envs.step(actor.get_eval_action(eval_obs))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)

            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)

            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)

            if args.evaluate:
                break
            actor.train()

            if args.save_model:
                model_path = f"runs/{run_name}/ckpt_{global_step}.pt"
                torch.save({
                    'actor': actor.state_dict(),
                    'qf1':   qf1_target.state_dict(),
                    'qf2':   qf2_target.state_dict(),
                    'log_alpha': log_alpha,
                }, model_path)
                print(f"model saved to {model_path}")

        # ── Rollout ───────────────────────────────────────────────────────────
        rollout_time = time.perf_counter()
        for local_step in range(args.steps_per_env):
            global_step += args.num_envs

            if not learning_has_started:
                if args.checkpoint is not None:
                    with torch.no_grad():
                        actions, _, _ = actor.get_action(obs)
                    actions = actions.detach()
                else:
                    actions = torch.tensor(envs.action_space.sample(), dtype=torch.float32, device=device)
            else:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(obs)
                actions = actions.detach()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            logger.add_scalar("train/reward", rewards.float().mean(), global_step)

            real_next_obs = next_obs.clone()
            if args.bootstrap_at_done == 'never':
                need_final_obs = torch.ones_like(terminations, dtype=torch.bool)
                stop_bootstrap = truncations | terminations
            elif args.bootstrap_at_done == 'always':
                need_final_obs = truncations | terminations
                stop_bootstrap = torch.zeros_like(terminations, dtype=torch.bool)
            else:
                need_final_obs = truncations & (~terminations)
                stop_bootstrap = terminations

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask  = infos["_final_info"]
                real_next_obs[need_final_obs] = infos["final_observation"][need_final_obs]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

            rb.add(obs, real_next_obs, actions, rewards, stop_bootstrap)
            obs = next_obs

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        pbar.update(args.num_envs * args.steps_per_env)

        # ── Training ──────────────────────────────────────────────────────────
        if global_step < args.learning_starts:
            continue

        update_time = time.perf_counter()
        learning_has_started = True

        if cuda_graph is None:
            print("Capturing CUDA graphs...")
            g_q, g_actor, g_target = build_cuda_graphs()
            cuda_graph = True
            print("CUDA graphs captured.")

        for local_update in range(args.grad_steps_per_iteration):
            global_update += 1
            data = rb.sample(args.batch_size)

            if cuda_graph:
                # Copy fresh data into static tensors
                static_obs.copy_(data.obs)
                static_next_obs.copy_(data.next_obs)
                static_actions.copy_(data.actions)
                static_rewards.copy_(data.rewards.flatten())
                static_dones.copy_(data.dones.flatten())
                static_alpha.fill_(log_alpha.exp().item() if args.autotune else args.alpha)

                g_q.replay()

                if global_update % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        if args.autotune:
                            static_alpha.fill_(log_alpha.exp().item())
                        g_actor.replay()
                        
                if global_update % args.target_network_frequency == 0:
                    g_target.replay()

                if args.autotune:
                    alpha = log_alpha.exp().item()

            else:
                # ── Fallback non-graph update (autotune=False) ────────────────
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_obs)
                    qf1_next_target = qf1_target(data.next_obs, next_state_actions)
                    qf2_next_target = qf2_target(data.next_obs, next_state_actions)
                    min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * min_qf_next_target.view(-1)

                qf1_a_values = qf1(data.obs, data.actions).view(-1)
                qf2_a_values = qf2(data.obs, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss  = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_update % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(data.obs)
                        qf1_pi    = qf1(data.obs, pi)
                        qf2_pi    = qf2(data.obs, pi)
                        min_qf_pi = torch.min(qf1_pi, qf2_pi)
                        actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                if global_update % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.lerp_(param.data, args.tau)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.lerp_(param.data, args.tau)

                static_qf1_loss.fill_(qf1_loss.item())
                static_qf2_loss.fill_(qf2_loss.item())
                static_actor_loss.fill_(actor_loss.item())

        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time

        # ── Logging ───────────────────────────────────────────────────────────
        if (global_step - args.training_freq) // args.log_freq < global_step // args.log_freq:
            logger.add_scalar("losses/qf1_loss",   static_qf1_loss.item(),   global_step)
            logger.add_scalar("losses/qf2_loss",   static_qf2_loss.item(),   global_step)
            logger.add_scalar("losses/qf_loss",    (static_qf1_loss + static_qf2_loss).item() / 2.0, global_step)
            logger.add_scalar("losses/actor_loss", static_actor_loss.item(), global_step)
            logger.add_scalar("losses/alpha",      alpha,                    global_step)
            logger.add_scalar("time/update_time",  update_time,              global_step)
            logger.add_scalar("time/rollout_time", rollout_time,             global_step)
            logger.add_scalar("time/rollout_fps",  global_steps_per_iteration / rollout_time, global_step)
            for k, v in cumulative_times.items():
                logger.add_scalar(f"time/total_{k}", v, global_step)
            logger.add_scalar("time/total_rollout+update_time",
                              cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
            if args.autotune:
                logger.add_scalar("losses/alpha_loss", static_alpha_loss.item(), global_step)

    # ── Save final model ──────────────────────────────────────────────────────
    if not args.evaluate and args.save_model:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save({
            'actor':     actor.state_dict(),
            'qf1':       qf1_target.state_dict(),
            'qf2':       qf2_target.state_dict(),
            'log_alpha': log_alpha,
        }, model_path)
        print(f"model saved to {model_path}")
        writer.close()

    eval_envs.close()
    envs.close()