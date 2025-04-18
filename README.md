# Adversarial Attacks on NLP Models (Text Perturbation)

This project explores the impact of adversarial attacks on NLP models by applying text perturbation techniques to deceive sentiment analysis and hate speech detection models.

## Project Overview

The research investigates how small, carefully crafted modifications in text can manipulate model predictions. The project focuses on:

1. Evaluating the vulnerability of different NLP models (LSTM, CNN, BERT) to adversarial attacks
2. Comparing various text perturbation strategies (character-level, word-level, sentence-level)
3. Testing defense mechanisms to enhance model robustness

## Project Structure

```
.
├── main.py                        # Main script to run experiments
├── data_loading_preprocessing.py  # Data loading and preprocessing functions
├── baseline_model_training.py     # Model training and evaluation
├── adversarial_attacks.py         # Adversarial attack implementation
├── defense_mechanisms.py          # Defense mechanisms against attacks
├── visualization_scripts.py       # Visualization utilities
├── requirements.txt               # Required packages
└── experiments/                   # Generated experiment results
    └── training/                  # Each experiment run
        ├── models/                # Saved models
        ├── results/               # Evaluation results
    └── char_testing/                 # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── visualizations/        # Result visualizations
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── word_testing/              # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── visualizations/        # Result visualizations
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── homoglyph_testing/                  # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── visualizations/        # Result visualizations
        └── attack_results.json    # attack results
        └── config.json            # configurations
    └── deepwordbug_testing/       # Each experiment run
        ├── adversarial_examples/  # Generated adversarial examples
        └── visualizations/        # Result visualizations
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
```

### Run Adversarial Attacks
```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type char --num_examples 100 --checkpoint_path checkpoints/bert_epoch_1.pt
```
```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type word --num_examples 100 --checkpoint_path checkpoints/bert_epoch_1.pt
```
```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type homoglyph --num_examples 100 --checkpoint_path checkpoints/bert_epoch_1.pt
```
```bash
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type deepwordbug --num_examples 1000 --checkpoint_path checkpoints/bert_epoch_1.pt
```
## Command Line Arguments

- `--task`: Task to perform (`train_models`, `run_attacks`, `apply_defenses`, `visualize_results`, `full_pipeline`)
- `--dataset`: Dataset to use (`imdb`, `jigsaw`, `both`)
- `--model_type`: Type of model to train/attack (`lstm`, `cnn`, `bert`, `all`)
- `--attack_type`: Type of adversarial attack (`char`, `word`, `homoglyph`, `textfooler`, `deepwordbug`, `bert-attack`, `all`)
- `--defense_type`: Type of defense mechanism (`adversarial_training`, `input_sanitization`, `robust_embeddings`, `ensemble`, `all`)
- `--num_examples`: Number of examples to use for adversarial attacks (default: 100)
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
- `visualizations/`: PNG files with visualizations of attack success rates, model performance degradation, etc.

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
```

## Author

Nidhishree Talastha

---

*This project is part of research on the security and robustness of Natural Language Processing models.*
