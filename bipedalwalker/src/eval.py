### My evaluation script

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import sys
import imageio
from pathlib import Path

# Add src directory to path to import reward_wrapper
sys.path.insert(0, os.path.dirname(__file__))
from reward_wrapper import BipedalWalkerRewardWrapper

# Import pygame for event pumping (prevents window freezing)
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Event pumping will be skipped.")

def evaluate_model(
    model_path,
    env_name="BipedalWalker-v3",
    n_episodes=10,
    render=True,
    save_video=True,
    video_dir="./videos",
    deterministic=True,
    vec_normalize_path=None,
    use_reward_wrapper=True,
    stay_upright_bonus=0.1,
    symmetry_penalty_weight=0.1,
    joint_velocity_bonus_weight=0.02
):
    """
    Evaluating a trained model on the BipedalWalker environment.
    
    Args:
        model_path: Path to the saved model
        env_name: Name of the gymnasium environment
        n_episodes: Number of episodes to run
        render: Whether to render the environment
        save_video: Whether to save evaluation videos
        video_dir: Directory to save videos
        deterministic: Whether to use deterministic actions
        vec_normalize_path: Path to VecNormalize stats (usually "./models/vec_normalize.pkl")
        use_reward_wrapper: Whether to use reward wrapper (should match training)
        stay_upright_bonus: Reward wrapper parameter (should match training)
        symmetry_penalty_weight: Reward wrapper parameter (should match training)
        joint_velocity_bonus_weight: Reward wrapper parameter (should match training)
    """
    # video directory
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
    
    # loading environment and model
    print(f"Loading model from {model_path}...")
    model = DDPG.load(model_path)
    
    # Create environment (matching training setup)
    def make_env():
        env = gym.make(env_name, render_mode="rgb_array" if save_video else None)
        
        # Apply reward wrapper if enabled (should match training)
        if use_reward_wrapper:
            env = BipedalWalkerRewardWrapper(
                env,
                stay_upright_bonus=stay_upright_bonus,
                symmetry_penalty_weight=symmetry_penalty_weight,
                joint_velocity_bonus_weight=joint_velocity_bonus_weight
            )
        return env
    
    # Create vectorized environment for VecNormalize
    eval_env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats if provided (critical for matching training)
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize stats from {vec_normalize_path}...")
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False  # Don't update stats during eval
        eval_env.norm_reward = False  # Don't normalize rewards during eval
        print("✓ VecNormalize stats loaded")
    else:
        if vec_normalize_path:
            print(f"Warning: VecNormalize path {vec_normalize_path} not found. Evaluation may not match training.")
        # Still wrap with VecNormalize but with default stats (not ideal)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    
    # Get unwrapped environment for rendering
    unwrapped_env = eval_env.envs[0]
    
    # evaluation 
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating model for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        # Reset through VecNormalize wrapper to get normalized observations
        obs = eval_env.reset()
        done = [False]
        episode_reward = 0
        episode_length = 0
        frames = []
        
        while not done[0]:
            # Get action from model (obs from VecNormalize is a numpy array)
            # VecNormalize returns obs as array, extract first (and only) environment
            obs_for_model = obs[0] if len(obs.shape) > 1 else obs
            action, _ = model.predict(obs_for_model, deterministic=deterministic)
            
            # Step through VecNormalize wrapper (expects action as array)
            obs, reward, done, info = eval_env.step([action])
            
            # Get actual reward (VecNormalize returns as array, norm_reward=False so it's raw)
            episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
            episode_length += 1
            
            # Save frame for video
            if save_video:
                # Render from unwrapped environment
                frame = unwrapped_env.render()
                frames.append(frame)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Save video
        if save_video and frames:
            video_path = os.path.join(video_dir, f"episode_{episode + 1}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  Video saved to {video_path}")
    
    eval_env.close()
    
    # Print statistics
    print("\n" + "="*50)
    print("Evaluation Results:")
    print("="*50)
    print(f"Number of episodes: {n_episodes}")
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print("="*50)
    
    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths)
    }

def render_episode(
    model_path,
    env_name="BipedalWalker-v3",
    deterministic=True,
    vec_normalize_path=None,
    use_reward_wrapper=True,
    stay_upright_bonus=0.1,
    symmetry_penalty_weight=0.1,
    joint_velocity_bonus_weight=0.02
):
    """
    Render a single episode with the trained model
    
    Args:
        model_path: Path to the saved model
        env_name: Name of the gymnasium environment
        deterministic: Whether to use deterministic actions
        vec_normalize_path: Path to VecNormalize stats (usually "./models/vec_normalize.pkl")
        use_reward_wrapper: Whether to use reward wrapper (should match training)
        stay_upright_bonus: Reward wrapper parameter (should match training)
        symmetry_penalty_weight: Reward wrapper parameter (should match training)
        joint_velocity_bonus_weight: Reward wrapper parameter (should match training)
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = DDPG.load(model_path)
    
    # Create environment (matching training setup)
    def make_env():
        env = gym.make(env_name, render_mode="human")
        
        # Apply reward wrapper if enabled (should match training)
        if use_reward_wrapper:
            env = BipedalWalkerRewardWrapper(
                env,
                stay_upright_bonus=stay_upright_bonus,
                symmetry_penalty_weight=symmetry_penalty_weight,
                joint_velocity_bonus_weight=joint_velocity_bonus_weight
            )
        return env
    
    # Create vectorized environment for VecNormalize
    eval_env = DummyVecEnv([make_env])
    
    # Load VecNormalize stats if provided (critical for matching training)
    if vec_normalize_path and os.path.exists(vec_normalize_path):
        print(f"Loading VecNormalize stats from {vec_normalize_path}...")
        eval_env = VecNormalize.load(vec_normalize_path, eval_env)
        eval_env.training = False  # Don't update stats during eval
        eval_env.norm_reward = False  # Don't normalize rewards during eval
        print("✓ VecNormalize stats loaded")
    else:
        if vec_normalize_path:
            print(f"Warning: VecNormalize path {vec_normalize_path} not found. Evaluation may not match training.")
        # Still wrap with VecNormalize but with default stats (not ideal)
        eval_env = VecNormalize(eval_env, training=False, norm_reward=False)
    
    # Get unwrapped environment for rendering
    unwrapped_env = eval_env.envs[0]
    
    # Reset through VecNormalize wrapper
    obs = eval_env.reset()
    done = [False]
    episode_reward = 0
    episode_length = 0
    
    print("Rendering episode... (Close window to stop)")
    print("NOTE: Event pumping is enabled to prevent window freezing.")
    
    while not done[0]:
        # Pump pygame events to prevent window freezing
        if PYGAME_AVAILABLE:
            try:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        print("\nWindow closed by user. Exiting...")
                        eval_env.close()
                        return episode_reward, episode_length
            except Exception:
                pass
        
        # Get action from model (obs from VecNormalize is a numpy array)
        obs_for_model = obs[0] if len(obs.shape) > 1 else obs
        action, _ = model.predict(obs_for_model, deterministic=deterministic)
        
        # Step through VecNormalize wrapper (expects action as array)
        obs, reward, done, info = eval_env.step([action])
        
        # Get actual reward
        episode_reward += reward[0] if isinstance(reward, (list, np.ndarray)) else reward
        episode_length += 1
    
    eval_env.close()
    
    print(f"\nEpisode completed!")
    print(f"Reward: {episode_reward:.2f}")
    print(f"Length: {episode_length}")
    
    return episode_reward, episode_length

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained BipedalWalker model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/bipedal_walker_final",
        help="Path to the saved model"
    )
    parser.add_argument(
        "--n_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes in real-time"
    )
    parser.add_argument(
        "--no_video",
        action="store_true",
        help="Don't save evaluation videos"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="./videos",
        help="Directory to save videos"
    )
    parser.add_argument(
        "--vec_normalize_path",
        type=str,
        default="./models/vec_normalize.pkl",
        help="Path to VecNormalize stats file (required for proper evaluation)"
    )
    parser.add_argument(
        "--no_reward_wrapper",
        action="store_true",
        help="Don't use reward wrapper (only if training didn't use it)"
    )
    parser.add_argument(
        "--stay_upright_bonus",
        type=float,
        default=0.1,
        help="Stay upright bonus weight (should match training)"
    )
    parser.add_argument(
        "--symmetry_penalty",
        type=float,
        default=0.1,
        help="Symmetry penalty weight (should match training)"
    )
    parser.add_argument(
        "--joint_velocity_bonus",
        type=float,
        default=0.02,
        help="Joint velocity bonus weight (should match training)"
    )
    
    args = parser.parse_args()
    
    if args.render:
        # Render a single episode
        render_episode(
            args.model_path,
            vec_normalize_path=args.vec_normalize_path,
            use_reward_wrapper=not args.no_reward_wrapper,
            stay_upright_bonus=args.stay_upright_bonus,
            symmetry_penalty_weight=args.symmetry_penalty,
            joint_velocity_bonus_weight=args.joint_velocity_bonus
        )
    else:
        # Evaluate multiple episodes
        evaluate_model(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            save_video=not args.no_video,
            video_dir=args.video_dir,
            vec_normalize_path=args.vec_normalize_path,
            use_reward_wrapper=not args.no_reward_wrapper,
            stay_upright_bonus=args.stay_upright_bonus,
            symmetry_penalty_weight=args.symmetry_penalty,
            joint_velocity_bonus_weight=args.joint_velocity_bonus
        )

