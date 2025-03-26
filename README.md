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
    └── TIMESTAMP/                 # Each experiment run
        ├── config.json            # Experiment configuration
        ├── models/                # Saved models
        ├── results/               # Evaluation results
        ├── adversarial_examples/  # Generated adversarial examples
        └── visualizations/        # Result visualizations
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
python main.py --task run_attacks --dataset imdb --model_type bert --attack_type word --num_examples 100
```

### Apply Defense Mechanisms

```bash
python main.py --task apply_defenses --dataset imdb --model_type bert --defense_type adversarial_training
```

### Visualize Results

```bash
python main.py --task visualize_results
```

### Run Full Pipeline

```bash
python main.py --task full_pipeline --dataset imdb --model_type bert --attack_type all --defense_type all --num_examples 100 --batch_size 16 --epochs 3
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

1. **IMDB Movie Reviews**: Used for sentiment analysis
2. **Jigsaw Toxic Comment**: Used for hate speech detection

## Attack Types

1. **Character-level attacks**: Misspellings, character swaps, and visually similar characters
2. **Word-level attacks**: Synonym replacement, word embedding-based substitutions
3. **Textfooler**: A state-of-the-art word substitution attack
4. **DeepWordBug**: Character-level perturbations targeting key tokens
5. **BERT-Attack**: Contextualized perturbations using BERT embeddings

## Defense Mechanisms

1. **Adversarial Training**: Augment training data with adversarial examples
2. **Input Sanitization**: Clean and normalize input text to remove adversarial perturbations
3. **Robust Word Embeddings**: Modify model embeddings to be more robust
4. **Ensemble Methods**: Combine multiple models to improve robustness

## Results

Experiment results are saved in the `experiments/TIMESTAMP/` directory:

- `results/`: CSV files with model performance metrics
- `adversarial_examples/`: CSV files with original and perturbed text examples
- `visualizations/`: PNG files with visualizations of attack success rates, model performance degradation, etc.

## Extending the Project

To extend the project:

1. Add new attack types in `adversarial_attacks.py`
2. Implement additional defense mechanisms in `defense_mechanisms.py`
3. Create new visualization functions in `visualization_scripts.py`
4. Modify `main.py` to incorporate your changes

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
