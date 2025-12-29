"""
Research Log System for tracking experiments, metrics, and failures.
Structured for ablation studies and documenting the research process.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np


class ResearchLog:
    """
    Research log for tracking experiments, metrics, and failures.
    Designed for ablation studies and documenting the research process.
    """
    
    def __init__(self, log_file: str = "./research_log.json"):
        """
        Initialize research log.
        
        Args:
            log_file: Path to JSON file for storing log entries
        """
        self.log_file = log_file
        self.experiments = []
        self.load()
    
    def load(self):
        """Load existing log entries from file."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.experiments = data.get('experiments', [])
            except Exception as e:
                print(f"Warning: Could not load research log: {e}")
                self.experiments = []
        else:
            self.experiments = []
    
    def save(self):
        """Save log entries to file."""
        os.makedirs(os.path.dirname(self.log_file) if os.path.dirname(self.log_file) else '.', exist_ok=True)
        with open(self.log_file, 'w') as f:
            json.dump({
                'experiments': self.experiments,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def start_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        baseline: str,
        variables: Dict[str, Any],
        description: str = ""
    ) -> Dict[str, Any]:
        """
        Start a new experiment.
        
        Args:
            experiment_id: Unique identifier (e.g., "A", "B", "C" or "EXP_001")
            experiment_name: Human-readable name
            baseline: Description of baseline (e.g., "PPO (Baseline)")
            variables: Dictionary of variables changed from baseline
            description: Additional description
        
        Returns:
            Experiment dictionary
        """
        experiment = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'baseline': baseline,
            'variables': variables,
            'description': description,
            'start_time': datetime.now().isoformat(),
            'status': 'running',
            'metrics': {
                'mean_reward': None,
                'std_reward': None,
                'max_reward': None,
                'min_reward': None,
                'mean_distance': None,
                'max_distance': None,
                'mean_episode_length': None,
                'stability_score': None,  # Can be calculated from hull angle variance
                'total_timesteps': None,
                'training_time_seconds': None
            },
            'failures': [],
            'pivots': [],
            'notes': []
        }
        
        self.experiments.append(experiment)
        self.save()
        return experiment
    
    def update_metrics(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        timesteps: Optional[int] = None
    ):
        """
        Update metrics for an experiment.
        
        Args:
            experiment_id: Experiment identifier
            metrics: Dictionary of metric values
            timesteps: Current timestep count
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment['metrics'].update(metrics)
            if timesteps:
                experiment['metrics']['total_timesteps'] = timesteps
            self.save()
    
    def record_failure(
        self,
        experiment_id: str,
        failure_description: str,
        hypothesis: str,
        reason: str,
        pivot: Optional[str] = None
    ):
        """
        Record a failure and potential pivot.
        
        Args:
            experiment_id: Experiment identifier
            failure_description: What failed (e.g., "High Speed Reward caused front flips")
            hypothesis: What was being tested
            reason: Why it failed (e.g., "Over-incentivized forward momentum at cost of balance")
            pivot: How this led to next idea (e.g., "Added hull-angle penalty to counter")
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            failure_entry = {
                'description': failure_description,
                'hypothesis': hypothesis,
                'reason': reason,
                'pivot': pivot,
                'timestamp': datetime.now().isoformat()
            }
            experiment['failures'].append(failure_entry)
            if pivot:
                experiment['pivots'].append({
                    'from_failure': failure_description,
                    'pivot_action': pivot,
                    'timestamp': datetime.now().isoformat()
                })
            self.save()
    
    def complete_experiment(
        self,
        experiment_id: str,
        final_metrics: Optional[Dict[str, float]] = None,
        success: bool = True
    ):
        """
        Mark experiment as complete.
        
        Args:
            experiment_id: Experiment identifier
            final_metrics: Final metric values
            success: Whether experiment was successful
        """
        experiment = self.get_experiment(experiment_id)
        if experiment:
            experiment['status'] = 'completed' if success else 'failed'
            experiment['end_time'] = datetime.now().isoformat()
            if final_metrics:
                experiment['metrics'].update(final_metrics)
            self.save()
    
    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment by ID."""
        for exp in self.experiments:
            if exp['experiment_id'] == experiment_id:
                return exp
        return None
    
    def get_summary(self) -> str:
        """Get a formatted summary of all experiments."""
        summary = "=" * 80 + "\n"
        summary += "RESEARCH LOG SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        for exp in self.experiments:
            summary += f"Experiment {exp['experiment_id']}: {exp['experiment_name']}\n"
            summary += f"  Baseline: {exp['baseline']}\n"
            summary += f"  Variables: {json.dumps(exp['variables'], indent=4)}\n"
            summary += f"  Status: {exp['status']}\n"
            
            if exp['metrics']['mean_reward'] is not None:
                summary += f"  Mean Reward: {exp['metrics']['mean_reward']:.2f} ± {exp['metrics'].get('std_reward', 0):.2f}\n"
                summary += f"  Max Distance: {exp['metrics'].get('max_distance', 'N/A')}\n"
                summary += f"  Stability Score: {exp['metrics'].get('stability_score', 'N/A')}\n"
            
            if exp['failures']:
                summary += f"  Failures: {len(exp['failures'])}\n"
                for failure in exp['failures']:
                    summary += f"    - {failure['description']}\n"
                    summary += f"      Reason: {failure['reason']}\n"
                    if failure.get('pivot'):
                        summary += f"      Pivot: {failure['pivot']}\n"
            
            summary += "\n"
        
        return summary
    
    def export_for_report(self, output_file: str = "./experiments_for_report.md"):
        """
        Export experiments in a format suitable for report/PDF.
        
        Args:
            output_file: Path to output markdown file
        """
        with open(output_file, 'w') as f:
            f.write("# Ablation Study Experiments\n\n")
            
            for exp in self.experiments:
                f.write(f"## Experiment {exp['experiment_id']}: {exp['experiment_name']}\n\n")
                f.write(f"**Baseline:** {exp['baseline']}\n\n")
                
                if exp['variables']:
                    f.write("**Variables Changed:**\n")
                    for key, value in exp['variables'].items():
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                if exp['description']:
                    f.write(f"**Description:** {exp['description']}\n\n")
                
                if exp['metrics']['mean_reward'] is not None:
                    f.write("**Results:**\n")
                    f.write(f"- Mean Reward: {exp['metrics']['mean_reward']:.2f} ± {exp['metrics'].get('std_reward', 0):.2f}\n")
                    if exp['metrics'].get('max_distance'):
                        f.write(f"- Max Distance: {exp['metrics']['max_distance']:.2f}\n")
                    if exp['metrics'].get('stability_score'):
                        f.write(f"- Stability Score: {exp['metrics']['stability_score']:.2f}\n")
                    f.write("\n")
                
                if exp['failures']:
                    f.write("**Failures and Pivots:**\n\n")
                    for failure in exp['failures']:
                        f.write(f"**Failure:** {failure['description']}\n")
                        f.write(f"- **Hypothesis:** {failure['hypothesis']}\n")
                        f.write(f"- **Reason:** {failure['reason']}\n")
                        if failure.get('pivot'):
                            f.write(f"- **Pivot:** {failure['pivot']}\n")
                        f.write("\n")
                
                f.write("---\n\n")
        
        print(f"Report exported to {output_file}")


def calculate_stability_score(episode_rewards: List[float], episode_lengths: List[int]) -> float:
    """
    Calculate stability score based on reward consistency and episode lengths.
    
    Higher score = more stable (consistent rewards, longer episodes)
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
    
    Returns:
        Stability score (0-1, higher is better)
    """
    if not episode_rewards or not episode_lengths:
        return 0.0
    
    # Normalize rewards (assuming rewards are typically negative to positive)
    reward_std = np.std(episode_rewards)
    reward_mean = np.mean(episode_rewards)
    
    # Normalize episode lengths
    length_mean = np.mean(episode_lengths)
    length_std = np.std(episode_lengths)
    
    # Stability = low variance in rewards + long episodes
    # Higher is better
    reward_stability = 1.0 / (1.0 + reward_std) if reward_std > 0 else 1.0
    length_stability = length_mean / (1.0 + length_std) if length_std > 0 else length_mean
    
    # Combine (normalize to 0-1 range, assuming max episode length ~1600)
    stability = (reward_stability * 0.5 + (length_stability / 1600.0) * 0.5)
    return min(1.0, max(0.0, stability))

