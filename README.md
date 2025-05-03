# Adversarial Attacks on NLP Models (Text Perturbation)

This project explores the impact of adversarial attacks on NLP models by applying text perturbation techniques to deceive sentiment analysis models.

## Project Overview

The research investigates how small, carefully crafted modifications in text can manipulate model predictions. The project focuses on:

1. Evaluating the vulnerability of different NLP models (BERT, DistilBERT) to adversarial attacks
2. Comparing various text perturbation strategies (character-level, word-level, sentence-level)

## Project Structure

```
.
├── main.py                        # Main script to run experiments
├── data_loading_preprocessing.py  # Data loading and preprocessing functions
├── baseline_model_training.py     # Model training and evaluation
├── adversarial_attacks.py         # Adversarial attack implementation
├── visualization_scripts.py       # Visualization utilities
├── requirements.txt               # Required packages
└── checkpoints/                   # Generated experiment results
    └── bert_checkpoints/          # checkpoints generated for each model and epochs
    └── distil_checkpoints/        # checkpoints generated for each model and epochs

└── experiments/                   # Generated experiment results
    └── training/                  # Each experiment run
        ├── bert_training/
        ├── distil_training/
    └── bert_char_testing/
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── bert_word_testing/              # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── bert_homoglyph_testing/                  # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── bert_deepwordbug_testing/       # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── distil_char_testing/                 # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── distil_word_testing/              # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── distil_homoglyph_testing/                  # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── distil_deepwordbug_testing/       # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── attack_results.json    # attack results
        └── config.json            # configurations
```

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

The project can be run in different modes:

### Train Baseline Models

```bash
python main.py --task train_models --dataset imdb --model_type bert --batch_size 16 --epochs 3
python main.py --task train_models --dataset imdb --model_type distil --batch_size 16 --epochs 3

```

### Run Adversarial Attacks

```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type char --num_examples 1000 --checkpoint_path checkpoints/bert_checkpoints/bert_epoch_1.pt
python main.py --task run_attacks --dataset imdb --model_type distil --attack_type char --num_examples 1000 --checkpoint_path checkpoints/distil_checkpoints/bert_epoch_1.pt
```

```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type word --num_examples 1000 --checkpoint_path checkpoints/bert_epoch_1.pt
python main.py --task run_attacks --dataset imdb --model_type distil --attack_type word --num_examples 1000 --checkpoint_path checkpoints/distil_checkpoints/bert_epoch_1.pt
```

```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type homoglyph --num_examples 1000 --checkpoint_path checkpoints/bert_epoch_1.pt
python main.py --task run_attacks --dataset imdb --model_type distil --attack_type homoglyph --num_examples 1000 --checkpoint_path checkpoints/distil_checkpoints/bert_epoch_1.pt
```

```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type deepwordbug --num_examples 1000 --checkpoint_path checkpoints/bert_epoch_1.pt
python main.py --task run_attacks --dataset imdb --model_type distil --attack_type deepwordbug --num_examples 1000 --checkpoint_path checkpoints/distil_checkpoints/bert_epoch_1.pt
```

## Command Line Arguments

- `--task`: Task to perform (`train_models`, `run_attacks`, `visualize_results`, `full_pipeline`)
- `--dataset`: Dataset to use (`imdb`)
- `--model_type`: Type of model to train/attack (`distil`, `bert`, `all`)
- `--attack_type`: Type of adversarial attack (`char`, `word`, `homoglyph`, `deepwordbug`, `all`)
- `--num_examples`: Number of examples to use for adversarial attacks (default: 1000)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of epochs for training (default: 3)

## Datasets

**IMDB Movie Reviews**: Used for sentiment analysis

## Attack Types

1. **Character-level attacks**: Misspellings, character swaps, and visually similar characters
2. **Word-level attacks**: Synonym replacement, word embedding-based substitutions
3. **Homoglyph attacks**: exploit the visual similarity between certain characters
4. **Deepwordbug attacks**: combines a sophisticated scoring function with character-level transformations

## Results

Experiment results are saved in the `experiments/attack_type_testing/` directory:

- `config`: JSON files with configurations
- `attack_results`: JSON files with model performance metrics
- `adversarial_examples/`: CSV files with original and perturbed text examples

## Requirements

```
torch>=1.8.0
transformers>=4.5.0
datasets>=1.6.0
textattack>=0.3.0
openattack>=2.0.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
pandas>=1.2.0
numpy>=1.20.0
seaborn>=0.11.0
autocorrect>=2.5.0
tensorflow>=2.5.0
tqdm>=4.60.0
nltk>=3.6.0
tensorflow-hub>=0.16.1
```

## Author

Nidhishree Talastha

---

_This project is part of research on the security and robustness of Natural Language Processing models._
