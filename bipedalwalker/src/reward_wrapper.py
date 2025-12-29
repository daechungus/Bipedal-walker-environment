"""
Custom reward wrapper for BipedalWalker environment.
Implements advanced reward crafting to guide the walker through initial learning.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple


class BipedalWalkerRewardWrapper(gym.Wrapper):
    """
    Custom reward wrapper for BipedalWalker that adds:
    1. Stay-Upright Bonus: Reward for keeping hull above a certain height
    2. Symmetry Penalty: Penalize identical leg movements (prevents hopping)
    """
    
    def __init__(
        self,
        env: gym.Env,
        stay_upright_bonus: float = 0.1,
        min_hull_height: float = 0.5,
        symmetry_penalty_weight: float = 0.1,
        joint_velocity_bonus_weight: float = 0.02
    ):
        """
        Initialize the reward wrapper.
        
        Args:
            env: The BipedalWalker environment to wrap
            stay_upright_bonus: Bonus reward per timestep when hull is above min_hull_height
            min_hull_height: Minimum hull height to receive stay-upright bonus
            symmetry_penalty_weight: Weight for symmetry penalty (penalize identical leg movements)
            joint_velocity_bonus_weight: Weight for joint velocity bonus (rewards faster leg movement)
        """
        super().__init__(env)
        self.stay_upright_bonus = stay_upright_bonus
        self.min_hull_height = min_hull_height
        self.symmetry_penalty_weight = symmetry_penalty_weight
        self.joint_velocity_bonus_weight = joint_velocity_bonus_weight
        
        # Track stationary state for progressive penalty
        self.stationary_steps = 0
        self.stationary_threshold = 0.02  # Velocity below this is considered stationary
        
        # Track reward components for debugging
        self.last_reward_components = {
            'base_reward': 0.0,
            'stay_upright_bonus': 0.0,
            'symmetry_penalty': 0.0,
            'forward_bonus': 0.0,
            'joint_velocity_bonus': 0.0,
            'stationary_penalty': 0.0,
            'total_reward': 0.0
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment and modify the reward.
        
        Args:
            action: Action array (4D: [hip1, ankle1, hip2, ankle2])
            
        Returns:
            observation, modified_reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get forward velocity once (used in multiple reward components)
        forward_velocity = obs[2] if len(obs) > 2 else 0.0
        
        # Track stationary state (reset on episode end or when moving)
        if terminated or truncated:
            self.stationary_steps = 0
        elif abs(forward_velocity) < self.stationary_threshold:
            self.stationary_steps += 1
        else:
            self.stationary_steps = 0  # Reset when moving
        
        # Store base reward, but reduce it significantly when stationary
        # This prevents the agent from exploiting the base reward by standing still
        base_reward = reward
        if abs(forward_velocity) < self.stationary_threshold and not terminated and not truncated:
            # Reduce base reward by 80% when stationary to discourage standing still
            base_reward = base_reward * 0.2
        
        # ====================================================================
        # 1. STAY-UPRIGHT BONUS (CONDITIONAL ON MOVEMENT)
        # ====================================================================
        # Rewards the walker for keeping the hull upright and stable.
        # IMPORTANT: Only rewards when moving forward to prevent reward hacking
        # (agent staying still to farm the bonus).
        # 
        # TO TWEAK:
        # - Change stay_upright_bonus (default 0.1): Higher = more reward for staying upright
        # - Change the decay rates: 
        #   * -2.0 in angle_reward: Lower (e.g., -1.0) = more forgiving of tilt
        #   * -0.5 in stability_reward: Lower (e.g., -0.2) = more forgiving of rotation
        # - min_forward_velocity: Minimum forward velocity to receive bonus (prevents farming)
        # ====================================================================
        stay_upright_bonus = 0.0
        if not terminated and not truncated:
            # BipedalWalker observations: [hull_angle, hull_angular_vel, horizontal_vel, vertical_vel, ...]
            hull_angle = obs[0] if len(obs) > 0 else 0.0
            hull_angular_vel = obs[1] if len(obs) > 1 else 0.0
            
            # Only reward staying upright if moving forward (prevents reward hacking)
            # This prevents the agent from just standing still to farm the bonus
            # LOWERED THRESHOLD: 0.02 instead of 0.1 to reward even small movements
            min_forward_velocity = 0.02  # Minimum forward velocity to receive bonus
            if forward_velocity >= min_forward_velocity:
                # Reward for being upright (angle close to 0) and stable (low angular velocity)
                # TWEAK THESE VALUES: Lower absolute values = more forgiving
                angle_reward = np.exp(-2.0 * abs(hull_angle))  # Try: -1.0 to -3.0
                stability_reward = np.exp(-0.5 * abs(hull_angular_vel))  # Try: -0.2 to -1.0
                
                # Combined stay-upright bonus (scaled by the configured bonus amount)
                stay_upright_bonus = self.stay_upright_bonus * angle_reward * stability_reward
        
        # Symmetry penalty
        if len(action) >= 4:
            # Actions: [hip1, ankle1, hip2, ankle2]
            leg1_actions = action[:2]  # [hip1, ankle1]
            leg2_actions = action[2:4]  # [hip2, ankle2]
            
            # Calculate difference between leg actions
            leg_diff = np.abs(leg1_actions - leg2_actions)
            symmetry_score = np.mean(leg_diff)  
            symmetry_penalty = self.symmetry_penalty_weight * np.exp(-symmetry_score * 5.0)
        else:
            symmetry_penalty = 0.0
        
        # Forward reward
        forward_bonus = 0.1 * max(0, forward_velocity)  # Reward positive forward movement
        
        joint_velocity_bonus = 0.0
        if len(obs) > 11:
            joint_velocities = obs[8:12]  # Joint angular velocities
            
            # Only reward joint velocity if moving forward (prevents reward hacking)
            # This prevents the agent from just stabilizing and moving legs without progress
            # LOWERED THRESHOLD: 0.02 instead of 0.05 to reward even small movements
            min_forward_velocity = 0.02  # Minimum forward velocity to receive bonus
            if forward_velocity >= min_forward_velocity:
                # Reward the mean absolute joint velocity (faster = better)
                mean_joint_velocity = np.mean(np.abs(joint_velocities))
                joint_velocity_bonus = self.joint_velocity_bonus_weight * mean_joint_velocity
        
        # Stationary penalty to discourage standing still instead of falling.
        stationary_penalty = 0.0
        if abs(forward_velocity) < self.stationary_threshold and not terminated and not truncated:
            # Progressive penalty: starts at 0.02, increases by 0.01 per step (capped at 0.1)
            # This makes it increasingly expensive to stay still
            base_penalty = 0.02
            progressive_penalty = min(0.01 * self.stationary_steps, 0.08)  # Max additional 0.08
            stationary_penalty = base_penalty + progressive_penalty
        
        # Combined rewards
        
        modified_reward = (
            base_reward                    # Original environment reward
            + stay_upright_bonus           # Bonus for staying upright (only when moving)
            - symmetry_penalty             # Penalty for identical leg movements
            + forward_bonus                 # Bonus for forward movement
            + joint_velocity_bonus         # Bonus for faster leg movement (only when moving)
            - stationary_penalty            # Penalty for being stationary
            # ADD YOUR CUSTOM REWARDS HERE:
            # - energy_penalty
            # + contact_bonus
        )
        
        # Store components for debugging (add to info)
        self.last_reward_components = {
            'base_reward': base_reward,
            'stay_upright_bonus': stay_upright_bonus,
            'symmetry_penalty': symmetry_penalty,
            'forward_bonus': forward_bonus,
            'joint_velocity_bonus': joint_velocity_bonus,
            'stationary_penalty': stationary_penalty,
            'total_reward': modified_reward
        }
        
        # Add reward breakdown to info for logging
        info['reward_components'] = self.last_reward_components.copy()
        
        # Add diagnostic metrics for movement analysis
        # These help diagnose slow leg movement issues in ablation studies
        if len(obs) > 11:
            joint_velocities = obs[8:12]  # Joint angular velocities (hip1, ankle1, hip2, ankle2)
            mean_joint_velocity = np.mean(np.abs(joint_velocities))
            max_joint_velocity = np.max(np.abs(joint_velocities))
        else:
            mean_joint_velocity = 0.0
            max_joint_velocity = 0.0
        
        mean_action_magnitude = np.mean(np.abs(action)) if len(action) > 0 else 0.0
        forward_velocity = obs[2] if len(obs) > 2 else 0.0
        
        info['movement_diagnostics'] = {
            'mean_joint_velocity': float(mean_joint_velocity),  # How fast legs are moving (rad/s)
            'max_joint_velocity': float(max_joint_velocity),     # Fastest joint (rad/s)
            'mean_action_magnitude': float(mean_action_magnitude),  # How strong actions are
            'forward_velocity': float(forward_velocity),         # Forward movement speed (m/s)
        }
        
        return obs, modified_reward, terminated, truncated, info
    
    def get_reward_components(self) -> Dict[str, float]:
        """Get the last reward component breakdown."""
        return self.last_reward_components.copy()

