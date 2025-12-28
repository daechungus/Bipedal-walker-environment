### My evaluation script

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import os
import imageio
from pathlib import Path

def evaluate_model(
    model_path,
    env_name="BipedalWalker-v3",
    n_episodes=10,
    render=True,
    save_video=True,
    video_dir="./videos",
    deterministic=True
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
    """
    # video directory
    if save_video:
        os.makedirs(video_dir, exist_ok=True)
    
    # loading environment and model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    env = gym.make(env_name, render_mode="rgb_array" if save_video else None)
    
    # evaluation 
    episode_rewards = []
    episode_lengths = []
    
    print(f"Evaluating model for {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        frames = []
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # Save frame for video
            if save_video:
                frame = env.render()
                frames.append(frame)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Save video
        if save_video and frames:
            video_path = os.path.join(video_dir, f"episode_{episode + 1}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            print(f"  Video saved to {video_path}")
    
    env.close()
    
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
    deterministic=True
):
    """
    Render a single episode with the trained model
    
    Args:
        model_path: Path to the saved model
        env_name: Name of the gymnasium environment
        deterministic: Whether to use deterministic actions
    """
    # Load the model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = gym.make(env_name, render_mode="human")
    
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0
    
    print("Rendering episode... (Close window to stop)")
    
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
    
    env.close()
    
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
    
    args = parser.parse_args()
    
    if args.render:
        # Render a single episode
        render_episode(args.model_path)
    else:
        # Evaluate multiple episodes
        evaluate_model(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            save_video=not args.no_video,
            video_dir=args.video_dir
        )

