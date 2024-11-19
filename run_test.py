import argparse
import os
import time
import random
import pandas as pd
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import data_preparation
from recbole.data.dataset.sequential_dataset import SequentialDataset
from recbole.utils import init_seed, init_logger, set_color
from utils import get_model
from trainer import SelectedUserTrainer


def get_default_config_paths(model_name, dataset_name):
    """Constructs default configuration file paths based on the provided model and dataset names."""
    return [
        f'props/{model_name}.yaml',
        f'props/{dataset_name}.yaml',
        'props/overall.yaml'
    ]


def validate_input(model_name, dataset_name, config_paths):
    """Validates model name, dataset name, and configuration files."""
    valid_models = ["RankZero", "RankAggregated", "RankFixed", "RankNearest"]
    valid_datasets = ["ml-1m", "lastfm", "Games"]

    # Validate model name
    if model_name not in valid_models:
        raise ValueError(f"Invalid model name '{model_name}'. Choose from {valid_models}.")

    # Validate dataset name
    if dataset_name not in valid_datasets:
        raise ValueError(f"Invalid dataset name '{dataset_name}'. Choose from {valid_datasets}.")

    # Validate configuration files
    for config_path in config_paths:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")


def apply_noise_to_dataset(dataset, noise_type='random', noise_ratio=0.2):
    """
    Apply noise to the dataset for robustness testing.

    Args:
        dataset (SequentialDataset): The original dataset.
        noise_type (str): Type of noise to apply ('random', 'truncate', 'duplicate').
        noise_ratio (float): The proportion of the dataset to perturb.

    Returns:
        SequentialDataset: The perturbed dataset.
    """
    data = dataset.inter_feat  # Interaction features (user, item, etc.)

    if noise_type == 'random':
        # Randomly replace items in user histories
        num_noisy = int(len(data) * noise_ratio)
        noisy_indices = random.sample(range(len(data)), num_noisy)
        unique_items = data['item_id'].unique()

        for idx in noisy_indices:
            data.loc[idx, 'item_id'] = random.choice(unique_items)  # Replace with a random item

    elif noise_type == 'truncate':
        # Truncate user histories
        max_length = 5
        data = data.groupby('user_id').head(max_length)  # Keep only the first max_length items

    elif noise_type == 'duplicate':
        # Introduce duplicate candidates
        duplicates = data.sample(frac=noise_ratio, replace=True)
        data = pd.concat([data, duplicates], ignore_index=True)

    dataset.inter_feat = data
    return dataset


def evaluate(model_name, dataset_name, pretrained_file, noise_type=None, noise_ratio=0.2, **kwargs):
    """
    Evaluates a sequential recommendation model using RecBole framework.

    Args:
        model_name (str): Name of the model to evaluate.
        dataset_name (str): Name of the dataset.
        pretrained_file (str): Path to the pre-trained model file.
        noise_type (str): Type of noise to apply (optional).
        noise_ratio (float): Proportion of the dataset to perturb (optional).
        kwargs (dict): Additional configurations.

    Returns:
        tuple: (model name, dataset name, evaluation results dictionary)
    """
    try:
        # Automatically determine the configuration file paths
        config_paths = get_default_config_paths(model_name, dataset_name)
        validate_input(model_name, dataset_name, config_paths)
        print(f"Using configuration files: {config_paths}")

        model_class = get_model(model_name)

        # Configurations initialization
        config = Config(model=model_class, dataset=dataset_name, config_file_list=config_paths, config_dict=kwargs)
        init_seed(config['seed'], config['reproducibility'])

        # Logger initialization
        init_logger(config)
        logger = getLogger()
        logger.info(config)
        logger.info(f"Seed: {config['seed']}, Reproducibility: {config['reproducibility']}")

        # Dataset initialization
        dataset = SequentialDataset(config)

        # Apply noise if specified
        if noise_type:
            logger.info(f"Applying noise: Type={noise_type}, Ratio={noise_ratio}")
            dataset = apply_noise_to_dataset(dataset, noise_type=noise_type, noise_ratio=noise_ratio)

        logger.info(dataset)

        # Dataset splitting
        train_data, valid_data, test_data = data_preparation(config, dataset)

        # Model initialization
        model = model_class(config, train_data.dataset).to(config['device'])

        # Load pre-trained model if specified
        if pretrained_file:
            checkpoint = torch.load(pretrained_file)
            logger.info(f"Loading pre-trained model from '{pretrained_file}'")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            model.load_other_parameter(checkpoint.get("other_parameter"))

        logger.info(model)

        # Trainer initialization
        trainer = SelectedUserTrainer(config, model, dataset)

        # Get embeddings and evaluate the model
        start_time = time.time()
        trainer.get_emb_multivector(train_data, valid_data, test_data, load_best_model=False, show_progress=config['show_progress'])
        test_result = trainer.evaluate(test_data, valid_data, load_best_model=False, show_progress=config['show_progress'])
        end_time = time.time()

        # Log evaluation results and execution time
        logger.info(f"Evaluation completed in {end_time - start_time:.2f} seconds")
        output_res = [f'{u} - {v}' for u, v in test_result.items()]
        logger.info(set_color('Test Results:', 'yellow') + '\t'.join(output_res))

        return config['model'], config['dataset'], {
            'valid_score_bigger': config['valid_metric_bigger'],
            'test_result': test_result
        }

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a sequential recommendation model using RecBole.")
    parser.add_argument("-m", type=str, required=True, help="Model name [RankZero, RankAggregated, RankFixed, RankNearest]")
    parser.add_argument('-d', type=str, required=True, help="Dataset name [ml-1m, lastfm, Games]")
    parser.add_argument('-p', type=str, default='', help="Path to the pre-trained model file (optional)")
    parser.add_argument('-n', type=int, default=200, help="Number of data points to evaluate")
    parser.add_argument('-pl', type=str, default="gpt-3.5-turbo", help="OpenAI engine [gpt-4, gpt-3.5-turbo]")
    parser.add_argument('-sd', type=int, default=2020, help="Seed value for reproducibility")
    parser.add_argument('--noise_type', type=str, default=None, help="Type of noise to apply: random, truncate, duplicate")
    parser.add_argument('--noise_ratio', type=float, default=0.2, help="Proportion of the dataset to perturb")

    args = parser.parse_args()

    config_dict = {
        "platform": args.pl,
        "seed": args.sd,
        'num_demo_int': 2,
        'num_demo_out': 1,
        'sim': "multivector",
        'num_data': args.n
    }

    evaluate(model_name=args.m, dataset_name=args.d, pretrained_file=args.p, noise_type=args.noise_type, noise_ratio=args.noise_ratio, **config_dict)