### My training script

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import os

def train_baseline_ppo(
    env_name="BipedalWalker-v3",
    total_timesteps=1_000_000,
    render_during_training=False,
    render_eval=True,
    save_dir="./models",
    log_dir="./runs",
    eval_freq=10000,
    n_eval_episodes=5
):
    """
    Train a baseline PPO agent on BipedalWalker environment.
    This will likely fail initially - the walker will fall over or twitch.
    
    Args:
        env_name: Name of the gymnasium environment
        total_timesteps: Total number of training timesteps
        render_during_training: Whether to render during training (slower)
        render_eval: Whether to render during evaluation episodes
        save_dir: Directory to save models
        log_dir: Directory for tensorboard logs
        eval_freq: Frequency of evaluation (in timesteps)
        n_eval_episodes: Number of episodes for evaluation
    """
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment
    print(f"Creating training environment: {env_name}")
    train_env = gym.make(env_name)
    train_env = Monitor(train_env, log_dir)
    
    # Create evaluation environment (with rendering if requested)
    print(f"Creating evaluation environment: {env_name}")
    eval_env = gym.make(
        env_name,
        render_mode="human" if render_eval else None
    )
    eval_env = Monitor(eval_env, os.path.join(log_dir, "eval"))
    
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
    eval_callback = EvalCallback(
        eval_env,
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
    print("="*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save final model
    final_model_path = os.path.join(save_dir, "bipedal_walker_final")
    model.save(final_model_path)
    print(f"\nTraining complete! Final model saved to: {final_model_path}")
    
    # Clean up
    train_env.close()
    eval_env.close()
    
    return model

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
        default=1_000_000,
        help="Total number of training timesteps"
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
            eval_freq=args.eval_freq
        )

