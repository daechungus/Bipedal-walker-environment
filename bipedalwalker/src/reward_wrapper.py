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
        symmetry_penalty_weight: float = 0.1
    ):
        """
        Initialize the reward wrapper.
        
        Args:
            env: The BipedalWalker environment to wrap
            stay_upright_bonus: Bonus reward per timestep when hull is above min_hull_height
            min_hull_height: Minimum hull height to receive stay-upright bonus
            symmetry_penalty_weight: Weight for symmetry penalty (penalize identical leg movements)
        """
        super().__init__(env)
        self.stay_upright_bonus = stay_upright_bonus
        self.min_hull_height = min_hull_height
        self.symmetry_penalty_weight = symmetry_penalty_weight
        
        # Track reward components for debugging
        self.last_reward_components = {
            'base_reward': 0.0,
            'stay_upright_bonus': 0.0,
            'symmetry_penalty': 0.0,
            'forward_bonus': 0.0,
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
        
        # Store base reward
        base_reward = reward
        
        # ====================================================================
        # 1. STAY-UPRIGHT BONUS
        # ====================================================================
        # Rewards the walker for keeping the hull upright and stable.
        # 
        # TO TWEAK:
        # - Change stay_upright_bonus (default 0.1): Higher = more reward for staying upright
        # - Change the decay rates: 
        #   * -2.0 in angle_reward: Lower (e.g., -1.0) = more forgiving of tilt
        #   * -0.5 in stability_reward: Lower (e.g., -0.2) = more forgiving of rotation
        # - You can also add conditions based on other observations:
        #   * obs[2] = horizontal velocity
        #   * obs[3] = vertical velocity
        #   * obs[4-7] = joint angles
        #   * obs[8-11] = joint angular velocities
        #   * obs[12-15] = leg ground contact (0 or 1)
        # ====================================================================
        stay_upright_bonus = 0.0
        if not terminated and not truncated:
            # BipedalWalker observations: [hull_angle, hull_angular_vel, horizontal_vel, vertical_vel, ...]
            hull_angle = obs[0] if len(obs) > 0 else 0.0
            hull_angular_vel = obs[1] if len(obs) > 1 else 0.0
            
            # Reward for being upright (angle close to 0) and stable (low angular velocity)
            # TWEAK THESE VALUES: Lower absolute values = more forgiving
            angle_reward = np.exp(-2.0 * abs(hull_angle))  # Try: -1.0 to -3.0
            stability_reward = np.exp(-0.5 * abs(hull_angular_vel))  # Try: -0.2 to -1.0
            
            # Combined stay-upright bonus (scaled by the configured bonus amount)
            stay_upright_bonus = self.stay_upright_bonus * angle_reward * stability_reward
        
        # ====================================================================
        # 2. SYMMETRY PENALTY
        # ====================================================================
        # Penalizes identical leg movements (prevents hopping behavior).
        #
        # TO TWEAK:
        # - Change symmetry_penalty_weight (default 0.1): Higher = stronger penalty
        # - Change the decay rate (-5.0): Lower (e.g., -3.0) = more forgiving
        # - You can modify the symmetry calculation:
        #   * Use max() instead of mean() for stricter penalty
        #   * Add separate penalties for hip vs ankle symmetry
        #   * Only penalize if both legs are moving (check if actions are non-zero)
        # ====================================================================
        if len(action) >= 4:
            # Actions: [hip1, ankle1, hip2, ankle2]
            leg1_actions = action[:2]  # [hip1, ankle1]
            leg2_actions = action[2:4]  # [hip2, ankle2]
            
            # Calculate difference between leg actions
            leg_diff = np.abs(leg1_actions - leg2_actions)
            symmetry_score = np.mean(leg_diff)  # TWEAK: Try max(leg_diff) for stricter
            
            # Penalize if legs are too similar (symmetry_score close to 0)
            # TWEAK THIS VALUE: Lower absolute value = more forgiving
            symmetry_penalty = self.symmetry_penalty_weight * np.exp(-symmetry_score * 5.0)  # Try: -3.0 to -7.0
        else:
            symmetry_penalty = 0.0
        
        # ====================================================================
        # 3. FORWARD PROGRESS BONUS
        # ====================================================================
        # Rewards the walker for moving forward (positive horizontal velocity).
        # This encourages the walker to actually walk forward instead of just
        # staying upright or moving backward.
        # ====================================================================
        forward_velocity = obs[2] if len(obs) > 2 else 0.0
        forward_bonus = 0.05 * max(0, forward_velocity)  # Reward positive forward movement
        
        # ====================================================================
        # COMBINE ALL REWARD COMPONENTS
        # ====================================================================
        # You can add your own custom reward components here!
        # Examples:
        # - Forward progress bonus: reward based on obs[2] (horizontal velocity)
        # - Energy efficiency: small penalty for large actions
        # - Contact bonus: reward when legs touch ground (obs[12-15])
        # - Smoothness bonus: reward for smooth action changes
        # ====================================================================
        modified_reward = (
            base_reward                    # Original environment reward
            + stay_upright_bonus           # Bonus for staying upright
            - symmetry_penalty             # Penalty for identical leg movements
            + forward_bonus                 # Bonus for forward movement
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
            'total_reward': modified_reward
        }
        
        # Add reward breakdown to info for logging
        info['reward_components'] = self.last_reward_components.copy()
        
        return obs, modified_reward, terminated, truncated, info
    
    def get_reward_components(self) -> Dict[str, float]:
        """Get the last reward component breakdown."""
        return self.last_reward_components.copy()

