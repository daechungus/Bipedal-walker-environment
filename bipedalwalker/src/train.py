### My training script

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
import os
import sys

# Add src directory to path to import reward_wrapper
sys.path.insert(0, os.path.dirname(__file__))
from reward_wrapper import BipedalWalkerRewardWrapper

def make_env_fn(
    env_name,
    rank=0,
    render_mode=None,
    monitor_dir=None,
    use_reward_wrapper=True,
    stay_upright_bonus=0.1,
    symmetry_penalty_weight=0.1
):
    """
    Create a function that makes an environment.
    This is a top-level function to ensure it's picklable for SubprocVecEnv.
    
    Args:
        env_name: Name of the environment
        rank: Rank of the environment (for seeding)
        render_mode: Render mode for the environment
        monitor_dir: Directory for Monitor logging
        use_reward_wrapper: Whether to use reward wrapper
        stay_upright_bonus: Bonus for staying upright
        symmetry_penalty_weight: Weight for symmetry penalty
    
    Returns:
        A function that creates and returns an environment
    """
    def _init():
        """Initialize and return an environment"""
        env = gym.make(env_name, render_mode=render_mode)
        
        # Apply custom reward wrapper if enabled
        if use_reward_wrapper:
            env = BipedalWalkerRewardWrapper(
                env,
                stay_upright_bonus=stay_upright_bonus,
                symmetry_penalty_weight=symmetry_penalty_weight
            )
        
        # Apply Monitor wrapper for logging
        if monitor_dir:
            # Create unique monitor directory for each environment
            env_monitor_dir = os.path.join(monitor_dir, f"env_{rank}") if rank is not None else monitor_dir
            env = Monitor(env, env_monitor_dir)
        
        return env
    
    return _init

def train_baseline_ppo(
    env_name="BipedalWalker-v3",
    total_timesteps=1_000_000_000,
    render_during_training=False,
    render_eval=True,
    save_dir="./models",
    log_dir="./runs",
    eval_freq=10000,
    n_eval_episodes=5,
    use_reward_wrapper=True,
    stay_upright_bonus=0.1,
    symmetry_penalty_weight=0.1,
    n_envs=8,
    use_subproc=True
):
    """
    Train a baseline PPO agent on BipedalWalker environment.
    This will likely fail initially - the walker will fall over or twitch.
    
    The environment is wrapped with VecNormalize to normalize the 24 observations
    that have different scales. This helps with training stability and convergence.
    
    Advanced reward crafting is applied via BipedalWalkerRewardWrapper:
    - Stay-Upright Bonus: Rewards keeping hull upright and stable
    - Symmetry Penalty: Penalizes identical leg movements (prevents hopping)
    
    Uses parallel environments (SubprocVecEnv) for faster training by collecting
    experiences from multiple environments simultaneously.
    
    Args:
        env_name: Name of the gymnasium environment
        total_timesteps: Total number of training timesteps
        render_during_training: Whether to render during training (slower)
        render_eval: Whether to render during evaluation episodes
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Frequency of evaluation (in timesteps)
        n_eval_episodes: Number of episodes for evaluation
        use_reward_wrapper: Whether to use the custom reward wrapper
        stay_upright_bonus: Bonus reward for staying upright (per timestep)
        symmetry_penalty_weight: Weight for symmetry penalty
        n_envs: Number of parallel environments (8 or 16 recommended)
        use_subproc: Whether to use SubprocVecEnv (True) or DummyVecEnv (False)
    
    Returns:
        model: Trained PPO model
        train_env: VecNormalize-wrapped training environment (contains normalization stats)
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment with VecNormalize
    print(f"Creating training environment: {env_name}")
    vec_env_type = "SubprocVecEnv" if use_subproc else "DummyVecEnv"
    print(f"  Using {vec_env_type} with {n_envs} parallel environments")
    if use_reward_wrapper:
        print("  Wrapping with RewardWrapper -> Monitor -> VecEnv -> VecNormalize")
        print("  RewardWrapper adds: Stay-Upright Bonus, Symmetry Penalty")
    else:
        print("  Wrapping with Monitor -> VecEnv -> VecNormalize")
    print("  VecNormalize will normalize the 24 observations to have similar scales")
    
    # Create environment maker functions for each parallel environment
    env_fns = [
        make_env_fn(
            env_name=env_name,
            rank=i,
            render_mode=None,
            monitor_dir=log_dir,
            use_reward_wrapper=use_reward_wrapper,
            stay_upright_bonus=stay_upright_bonus,
            symmetry_penalty_weight=symmetry_penalty_weight
        )
        for i in range(n_envs)
    ]
    
    # Create vectorized training environment
    if use_subproc:
        train_env = SubprocVecEnv(env_fns)
        print(f"  ✓ Created {n_envs} parallel environments using SubprocVecEnv")
    else:
        train_env = DummyVecEnv(env_fns)
        print(f"  ✓ Created {n_envs} environments using DummyVecEnv (sequential)")
    
    # Wrap with VecNormalize - this normalizes observations and rewards
    # norm_obs=True: normalize observations (important for BipedalWalker's 24 different-scaled obs)
    # norm_reward=True: normalize rewards (can help with training stability)
    # training=True: update running statistics during training
    train_env = VecNormalize(
        train_env,
        norm_obs=True,      # Normalize observations (critical for BipedalWalker)
        norm_reward=True,   # Normalize rewards
        training=True,      # Update running statistics
        clip_obs=10.0,      # Clip observations to prevent extreme values
        clip_reward=10.0    # Clip rewards
    )
    
    # Create evaluation environment
    # Note: For evaluation, we use a single environment (DummyVecEnv)
    # and sync the normalization stats from training
    print(f"Creating evaluation environment: {env_name}")
    eval_env_fn = make_env_fn(
        env_name=env_name,
        rank=0,
        render_mode="human" if render_eval else None,
        monitor_dir=os.path.join(log_dir, "eval"),
        use_reward_wrapper=use_reward_wrapper,
        stay_upright_bonus=stay_upright_bonus,
        symmetry_penalty_weight=symmetry_penalty_weight
    )
    eval_env = DummyVecEnv([eval_env_fn])
    
    # Wrap eval environment with VecNormalize
    # training=False: don't update stats during eval (use existing stats)
    eval_env = VecNormalize(
        eval_env,
        norm_obs=True,
        norm_reward=False,  # Typically don't normalize rewards during eval
        training=False,     # Don't update statistics during evaluation
        clip_obs=10.0
    )
    
    # Create PPO model with standard hyperparameters
    # These are default PPO settings - likely to perform poorly initially
    print("Initializing PPO model with standard hyperparameters...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )
    
    # Setup callbacks
    # Create a custom callback to sync VecNormalize stats before evaluation
    class SyncNormalizeCallback(EvalCallback):
        """Custom callback that syncs VecNormalize stats before evaluation"""
        def __init__(self, *args, train_env=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.train_env = train_env
        
        def _on_step(self) -> bool:
            # Sync normalization statistics from training to eval environment
            if self.train_env is not None and isinstance(self.train_env, VecNormalize):
                self.eval_env.obs_rms = self.train_env.obs_rms
                self.eval_env.ret_rms = self.train_env.ret_rms
            return super()._on_step()
    
    eval_callback = SyncNormalizeCallback(
        eval_env,
        train_env=train_env,  # Pass training env to sync stats
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=eval_freq,
        deterministic=True,
        render=render_eval,
        n_eval_episodes=n_eval_episodes
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="bipedal_walker"
    )
    
    # Train the model
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print("="*60)
    print("NOTE: The walker will likely fail initially - falling over or twitching.")
    print("This is expected behavior for a baseline PPO agent.")
    print("")
    print(f"Parallel Training Configuration:")
    print(f"  - Number of parallel environments: {n_envs}")
    print(f"  - Environment type: {vec_env_type}")
    print(f"  - Expected speedup: ~{n_envs}x faster data collection")
    print("")
    print("VecNormalize is active:")
    print("  - Observations are being normalized (24 different-scaled features)")
    print("  - Rewards are being normalized")
    print("  - Statistics update during training")
    if use_reward_wrapper:
        print("")
        print("Advanced Reward Crafting is active:")
        print(f"  - Stay-Upright Bonus: {stay_upright_bonus} (rewards stable, upright posture)")
        print(f"  - Symmetry Penalty: {symmetry_penalty_weight} (penalizes identical leg movements)")
    print("="*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model and VecNormalize wrapper
    final_model_path = os.path.join(save_dir, "bipedal_walker_final")
    model.save(final_model_path)
    
    # Save the VecNormalize wrapper (contains normalization statistics)
    vec_normalize_path = os.path.join(save_dir, "vec_normalize.pkl")
    train_env.save(vec_normalize_path)
    print(f"\nTraining complete!")
    print(f"  Model saved to: {final_model_path}")
    print(f"  VecNormalize stats saved to: {vec_normalize_path}")
    print(f"  (VecNormalize stats are needed when loading the model for evaluation)")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model, train_env

def quick_test_render(env_name="BipedalWalker-v3", n_episodes=3):
    """
    Quick test to render the environment with random actions.
    Useful for verifying the environment works and seeing initial failures.
    """
    print(f"Testing environment rendering with random actions...")
    env = gym.make(env_name, render_mode="human")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        
        print(f"\nEpisode {episode + 1}:")
        print("Watch the walker - it will likely fall immediately with random actions.")
        
        while not done:
            # Random actions (will cause immediate failure)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
            
            if step % 100 == 0:
                print(f"  Step {step}, Reward: {episode_reward:.2f}")
        
        print(f"Episode {episode + 1} finished: Reward = {episode_reward:.2f}, Steps = {step}")
    
    env.close()
    print("\nTest complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a baseline PPO agent on BipedalWalker")
    parser.add_argument(
        "--env_name",
        type=str,
        default="BipedalWalker-v3",
        help="Name of the gymnasium environment"
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=10_000_000,
        help="Total number of training timesteps (default: 10M, use --total_timesteps to override)"
    )
    parser.add_argument(
        "--render_training",
        action="store_true",
        help="Render during training (slower)"
    )
    parser.add_argument(
        "--no_render_eval",
        action="store_true",
        help="Don't render during evaluation"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./runs",
        help="Directory for tensorboard logs"
    )
    parser.add_argument(
        "--test_render",
        action="store_true",
        help="Just test rendering with random actions (no training)"
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10000,
        help="Frequency of evaluation (in timesteps)"
    )
    parser.add_argument(
        "--no_reward_wrapper",
        action="store_true",
        help="Disable custom reward wrapper (use default rewards)"
    )
    parser.add_argument(
        "--stay_upright_bonus",
        type=float,
        default=0.1,
        help="Bonus reward for staying upright (per timestep)"
    )
    parser.add_argument(
        "--symmetry_penalty",
        type=float,
        default=0.1,
        help="Weight for symmetry penalty (penalizes identical leg movements)"
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        choices=[1, 2, 4, 8, 16],
        help="Number of parallel environments (8 or 16 recommended for speed)"
    )
    parser.add_argument(
        "--no_subproc",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv (sequential, slower but more stable)"
    )
    
    args = parser.parse_args()
    
    if args.test_render:
        # Just test rendering
        quick_test_render(args.env_name)
    else:
        # Train the model
        train_baseline_ppo(
            env_name=args.env_name,
            total_timesteps=args.total_timesteps,
            render_during_training=args.render_training,
            render_eval=not args.no_render_eval,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            eval_freq=args.eval_freq,
            use_reward_wrapper=not args.no_reward_wrapper,
            stay_upright_bonus=args.stay_upright_bonus,
            symmetry_penalty_weight=args.symmetry_penalty,
            n_envs=args.n_envs,
            use_subproc=not args.no_subproc
        )

