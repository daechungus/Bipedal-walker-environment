"""
Quick test script to verify the BipedalWalker environment renders correctly.
This will show random actions causing immediate failures - useful for baseline comparison.
"""

import gymnasium as gym
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("="*60)
    print("BipedalWalker Environment Test")
    print("="*60)
    print("\nThis will show the walker with random actions.")
    print("Expected behavior: Walker will fall immediately or twitch randomly.")
    print("Close the render window when done.\n")
    
    env_name = "BipedalWalker-v3"
    
    try:
        env = gym.make(env_name, render_mode="human")
        print(f"✓ Environment '{env_name}' created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}\n")
        
        obs, info = env.reset()
        print("✓ Environment reset successful")
        print(f"  Initial observation shape: {obs.shape}\n")
        
        print("Running episode with random actions...")
        print("(Watch the render window - the walker should fail quickly)\n")
        
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 1000:  # Limit steps for test
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            if step % 50 == 0:
                print(f"  Step {step}, Reward: {total_reward:.2f}")
        
        print(f"\n✓ Test complete!")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Episode finished: {done}")
        
        env.close()
        print("\n✓ Environment closed successfully")
        print("\n" + "="*60)
        print("Rendering test passed! You can now run training.")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure you have installed all dependencies:")
        print("  pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()

