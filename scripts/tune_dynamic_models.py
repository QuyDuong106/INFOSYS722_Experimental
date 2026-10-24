"""Tune dynamic RFE counts for individual models; preserve existing Stacking runs.

Run from any directory using the project environment. Models and features are
selected using training-only validation; each excluded record stays untouched.
"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optuna_lopo import run_optuna_lopo

INDIVIDUAL_MODELS = ('logistic', 'random_forest', 'xgboost', 'lightgbm')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=INDIVIDUAL_MODELS,
                        default=list(INDIVIDUAL_MODELS))
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 20, 30, 38],
                        help='Source-feature counts searched separately for each model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--inner-folds', type=int, choices=[3], default=3)
    parser.add_argument('--top-k', type=int, default=20,
                        help='Initial template count; actual count comes from --k-values')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT/'results/dynamic_individual_v4')
    parser.add_argument('--max-folds', type=int,
                        help='Partial smoke run; no full-evaluation metrics are produced')
    parser.add_argument('--save-fold-models', action=argparse.BooleanOptionalAction, default=True,
                        help='Save fitted excluded-record pipelines for later XAI (default: yes)')
    parser.add_argument('--refit-final', action='store_true',
                        help='Additional full-data inference refits; not evaluation models')
    run_optuna_lopo(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
