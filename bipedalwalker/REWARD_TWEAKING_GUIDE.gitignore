# Reward Tweaking Guide

This guide explains how to customize the reward function for BipedalWalker training.

## Quick Start: Command Line Tweaking

The easiest way to tweak rewards is using command-line arguments:

```bash
# Adjust stay-upright bonus (default: 0.1)
python src/train.py --stay_upright_bonus 0.2

# Adjust symmetry penalty (default: 0.1)
python src/train.py --symmetry_penalty 0.05

# Combine both
python src/train.py --stay_upright_bonus 0.15 --symmetry_penalty 0.2
```

## Code-Level Tweaking

For more control, edit `bipedalwalker/src/reward_wrapper.py`:

### 1. Stay-Upright Bonus

**Location:** Lines 62-78 in `reward_wrapper.py`

**What it does:** Rewards the walker for keeping the hull upright and stable.

**Parameters you can tweak:**
- `stay_upright_bonus` (default: 0.1): Overall bonus multiplier
- `-2.0` in `angle_reward`: Decay rate for hull angle (lower = more forgiving)
- `-0.5` in `stability_reward`: Decay rate for angular velocity (lower = more forgiving)

**Example tweaks:**
```python
# More forgiving of tilt
angle_reward = np.exp(-1.0 * abs(hull_angle))  # Changed from -2.0

# More forgiving of rotation
stability_reward = np.exp(-0.2 * abs(hull_angular_vel))  # Changed from -0.5

# Higher overall bonus
stay_upright_bonus = self.stay_upright_bonus * angle_reward * stability_reward * 1.5
```

### 2. Symmetry Penalty

**Location:** Lines 80-95 in `reward_wrapper.py`

**What it does:** Penalizes identical leg movements to prevent hopping.

**Parameters you can tweak:**
- `symmetry_penalty_weight` (default: 0.1): Overall penalty multiplier
- `-5.0` in penalty calculation: Decay rate (lower = more forgiving)
- `np.mean()` vs `np.max()`: How to measure symmetry

**Example tweaks:**
```python
# More forgiving of symmetry
symmetry_penalty = self.symmetry_penalty_weight * np.exp(-symmetry_score * 3.0)  # Changed from 5.0

# Stricter penalty (use max instead of mean)
symmetry_score = np.max(leg_diff)  # Changed from np.mean(leg_diff)
```

### 3. Adding Custom Rewards

**Location:** Lines 97-107 in `reward_wrapper.py` (the reward combination section)

**Available observations:**
- `obs[0]`: Hull angle (radians)
- `obs[1]`: Hull angular velocity
- `obs[2]`: Horizontal velocity
- `obs[3]`: Vertical velocity
- `obs[4-7]`: Joint angles (hip1, ankle1, hip2, ankle2)
- `obs[8-11]`: Joint angular velocities
- `obs[12-15]`: Leg ground contact (0 or 1)

**Available actions:**
- `action[0]`: Hip1 torque
- `action[1]`: Ankle1 torque
- `action[2]`: Hip2 torque
- `action[3]`: Ankle2 torque

**Example custom rewards:**

```python
# Forward progress bonus
forward_velocity = obs[2] if len(obs) > 2 else 0.0
forward_bonus = 0.05 * max(0, forward_velocity)  # Reward positive forward movement

# Energy efficiency penalty
energy_penalty = 0.001 * np.sum(np.abs(action))  # Small penalty for using energy

# Contact bonus (reward when legs touch ground)
leg_contacts = obs[12:16] if len(obs) > 15 else [0, 0, 0, 0]
contact_bonus = 0.01 * np.sum(leg_contacts)  # Reward for ground contact

# Smoothness bonus (reward for smooth action changes)
# Note: You'd need to track previous action
if hasattr(self, 'prev_action'):
    action_change = np.mean(np.abs(action - self.prev_action))
    smoothness_bonus = 0.02 * np.exp(-action_change * 10.0)  # Reward smooth changes
else:
    smoothness_bonus = 0.0
self.prev_action = action.copy()

# Add to modified_reward:
modified_reward = (
    base_reward
    + stay_upright_bonus
    - symmetry_penalty
    + forward_bonus
    - energy_penalty
    + contact_bonus
    + smoothness_bonus
)
```

## Common Tweaking Strategies

### 1. **Encourage Forward Movement**
Add a forward velocity bonus:
```python
forward_velocity = obs[2] if len(obs) > 2 else 0.0
forward_bonus = 0.1 * max(0, forward_velocity)
modified_reward += forward_bonus
```

### 2. **Discourage Falling**
Add a larger penalty when the walker falls:
```python
if terminated:
    fall_penalty = -10.0
    modified_reward += fall_penalty
```

### 3. **Reward Ground Contact**
Encourage the walker to keep feet on the ground:
```python
leg_contacts = obs[12:16] if len(obs) > 15 else [0, 0, 0, 0]
contact_bonus = 0.05 * np.sum(leg_contacts)
modified_reward += contact_bonus
```

### 4. **Penalize Excessive Joint Angles**
Discourage extreme joint positions:
```python
joint_angles = obs[4:8] if len(obs) > 7 else [0, 0, 0, 0]
extreme_angle_penalty = 0.01 * np.sum(np.abs(joint_angles) > 1.0)  # Penalize angles > 1 rad
modified_reward -= extreme_angle_penalty
```

### 5. **Reward Stability**
Encourage low angular velocities:
```python
joint_velocities = obs[8:12] if len(obs) > 11 else [0, 0, 0, 0]
stability_bonus = 0.02 * np.exp(-np.mean(np.abs(joint_velocities)))
modified_reward += stability_bonus
```

## Testing Your Changes

1. **Quick test with random actions:**
   ```bash
   python quick_test.py
   ```

2. **Short training run:**
   ```bash
   python src/train.py --total_timesteps 100000 --eval_freq 5000
   ```

3. **Monitor reward components:**
   The reward components are logged in TensorBoard. Check the `info['reward_components']` dictionary to see the breakdown.

## Tips

- **Start small:** Make incremental changes (0.01-0.1 range)
- **Balance rewards:** Make sure bonuses and penalties are on similar scales
- **Test frequently:** Run short training sessions to see the effect
- **Watch TensorBoard:** Monitor how reward components change during training
- **Use negative rewards sparingly:** Too many penalties can make learning difficult

## Debugging

To see reward components in action, you can print them:

```python
# In reward_wrapper.py, add after line 113:
if step_count % 100 == 0:  # Print every 100 steps
    print(f"Rewards: base={base_reward:.3f}, upright={stay_upright_bonus:.3f}, "
          f"symmetry={symmetry_penalty:.3f}, total={modified_reward:.3f}")
```

Or access them from the environment:
```python
obs, reward, done, truncated, info = env.step(action)
if 'reward_components' in info:
    print(info['reward_components'])
```

