"""
Helper script to record failures and pivots in the research log.
Use this to document what didn't work and how you pivoted.
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.dirname(__file__))
from research_log import ResearchLog


def main():
    parser = argparse.ArgumentParser(description="Record a failure and pivot in research log")
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Experiment ID (e.g., 'A', 'B', 'EXP_001')"
    )
    parser.add_argument(
        "--failure",
        type=str,
        required=True,
        help="Description of what failed (e.g., 'High Speed Reward caused front flips')"
    )
    parser.add_argument(
        "--hypothesis",
        type=str,
        required=True,
        help="What was being tested (e.g., 'Rewarding high forward velocity would encourage walking')"
    )
    parser.add_argument(
        "--reason",
        type=str,
        required=True,
        help="Why it failed (e.g., 'Over-incentivized forward momentum at cost of balance')"
    )
    parser.add_argument(
        "--pivot",
        type=str,
        default=None,
        help="How this led to next idea (e.g., 'Added hull-angle penalty to counter front flips')"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="./runs/research_log.json",
        help="Path to research log file"
    )
    
    args = parser.parse_args()
    
    # Load research log
    research_log = ResearchLog(log_file=args.log_file)
    
    # Record failure
    research_log.record_failure(
        experiment_id=args.experiment_id,
        failure_description=args.failure,
        hypothesis=args.hypothesis,
        reason=args.reason,
        pivot=args.pivot
    )
    
    print(f"✓ Recorded failure for experiment {args.experiment_id}")
    print(f"  Failure: {args.failure}")
    print(f"  Reason: {args.reason}")
    if args.pivot:
        print(f"  Pivot: {args.pivot}")


if __name__ == "__main__":
    main()

