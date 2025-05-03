# Main Script for Adversarial NLP Research
# Author: Nidhishree Talastha

import os
import torch
import glob
import numpy as np
import pandas as pd
import argparse
import time
import json
import sys
import traceback
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Import project modules
try:
    # Import data loading and preprocessing functions
    from data_loading_preprocessing import (
        load_imdb_dataset, load_jigsaw_dataset, preprocess_text,
        prepare_data_for_bert, create_dataloaders
    )

    # Import model training functions
    from baseline_model_training import (
        train_bert_model, evaluate_model, save_model_results
    )

    # Import attack functions (if available)
    try:
        from adversarial_attacks import (
            run_adversarial_attack, save_attack_results
        )
        attacks_available = True
    except ImportError:
        print("Warning: adversarial_attacks.py not found. Attack functionality will be disabled.")
        attacks_available = False

    # Import visualization functions (if available)
    try:
        from visualization_scripts import (
            visualize_attack_success_rates, visualize_model_performance_degradation,
            visualize_text_perturbations, visualize_perturbation_impact_by_position
        )
        visualizations_available = True
    except ImportError:
        print("Warning: visualization_scripts.py not found. Visualization functionality will be disabled.")
        visualizations_available = False

except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please make sure all required modules are in the same directory as main.py")
    print("If you're missing any modules, you can find them in the project repository.")
    sys.exit(1)

# Import transformers for BERT
try:
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, BertTokenizer, BertForSequenceClassification
except ImportError:
    print("Error: transformers library not found. Please install it with 'pip install transformers'")
    sys.exit(1)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Adversarial Attacks on NLP Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--task", type=str, default="train_models", choices=[
        "train_models", "run_attacks", "apply_defenses", "visualize_results", "full_pipeline"
    ], help="Task to perform")
    
    parser.add_argument("--dataset", type=str, default="imdb", choices=["imdb"],
                        help="Dataset to use")
    
    parser.add_argument("--model_type", type=str, default="bert", choices=["bert", "distil"],
                        help="Type of model to train/attack")
    
    parser.add_argument("--checkpoint_path", type=str, default=None,
                   help="Path to a model checkpoint to use instead of model_results.json")

    parser.add_argument("--attack_type", type=str, default="word", 
                        choices=["char", "word", "homoglyph", "textfooler", "deepwordbug", "bert-attack", "all"],
                        help="Type of adversarial attack to perform")
    
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Number of examples to use for adversarial attacks")
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs for training")
                        
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum sequence length for tokenization")
                        
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
                        
    parser.add_argument("--output_dir", type=str, default="experiments",
                        help="Directory to save output files")
    
    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment directories and configuration."""
    # Create timestamp for experiment
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Create experiment directory
    experiment_dir = f"{args.output_dir}"
    attack_type = f"{args.attack_type}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    for subdir in ["models", "results"]:
        os.makedirs(f"{experiment_dir}/training/{subdir}", exist_ok=True)
    # Create subdirectories
    for subdir in ["adversarial_examples", "visualizations"]:
        os.makedirs(f"{experiment_dir}/{attack_type}_testing/{subdir}", exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config["timestamp"] = timestamp
    config["experiment_dir"] = experiment_dir
    
    # Set seeds based on config
    if "seed" in config:
        seed = config["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        config["seed_set"] = True
    
    # Save configuration to file
    if attack_type:
        with open(f"{experiment_dir}/{attack_type}_testing/config.json", "w") as f:
            json.dump(config, f, indent=4)
    else:
        with open(f"{experiment_dir}/config.json", "w") as f:
            json.dump(config, f, indent=4)
    
    print(f"Experiment directory created: {experiment_dir}")
    print(f"Configuration saved to: {experiment_dir}/config.json")
    
    return config

def train_baseline_models(config):
    """Train baseline models based on configuration."""
    experiment_dir = config["experiment_dir"] + "/training"
    datasets = []
    
    print("\n" + "="*50)
    print("TRAINING BASELINE MODELS")
    print("="*50)
    
    # Load datasets
    try:
        if config["dataset"] in ["imdb"]:
            print("\nLoading IMDB dataset...")
            imdb_train, imdb_test = load_imdb_dataset()
            datasets.append(("imdb", imdb_train, imdb_test))
        
        if not datasets:
            print("No datasets were loaded. Please check your dataset configuration.")
            return {}
    except Exception as e:
        print(f"Error loading datasets: {e}")
        traceback.print_exc()
        return {}
    
    model_results = {}
    
    # Train models for each dataset
    for dataset_name, train_data, test_data in datasets:
        print(f"\nTraining models for {dataset_name} dataset")
        print(f"Training data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")
        
        # Use a smaller subset for faster training during development
        if config.get("debug", False):
            print("DEBUG MODE: Using smaller dataset")
            train_data = train_data[:20000]
            test_data = test_data[:10000]
        
        if config["model_type"] in ["bert", "distil", "all"]:
            try:
                print("\nTraining BERT model")
                
                # Initialize tokenizer
                if config["model_type"] == "bert":
                    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                elif config["model_type"] == "distil":
                    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                else:
                    print("Please provide a model to train.")


                
                # Prepare data for BERT
                print("Preparing data for" + config[model_type] + "...")
                max_seq_length = config.get("max_seq_length", 128)
                train_encoded_inputs, train_labels = prepare_data_for_bert(train_data, tokenizer, max_seq_length=max_seq_length)
                test_encoded_inputs, test_labels = prepare_data_for_bert(test_data, tokenizer, max_seq_length=max_seq_length)
                
                # Create dataloaders
                print("Creating dataloaders...")
                batch_size = config["batch_size"]
                train_dataloader, val_dataloader = create_dataloaders(
                    train_encoded_inputs['input_ids'],
                    train_encoded_inputs['attention_mask'],
                    train_labels,
                    batch_size=batch_size
                )
                
                test_dataloader = create_dataloaders(
                    test_encoded_inputs['input_ids'],
                    test_encoded_inputs['attention_mask'],
                    test_labels,
                    batch_size=batch_size
                )[1]  # Just use the sequential dataloader
                
                # Train BERT model
                print(f"Training BERT model for {config['epochs']} epochs...")
                bert_model = train_bert_model(config["model_type"], train_dataloader, val_dataloader, epochs=config["epochs"])
                
                # Save the model
                model_path = f"{experiment_dir}/models/bert_{dataset_name}"
                print(f"Saving model to {model_path}")
                bert_model.save_pretrained(model_path)
                tokenizer.save_pretrained(model_path)
                
                # Evaluate on test set
                print("Evaluating model on test set...")
                bert_metrics = evaluate_model(bert_model, test_dataloader)
                
                # Print evaluation metrics
                print(f"Test accuracy: {bert_metrics['accuracy']:.4f}")
                print(f"Test F1 score: {bert_metrics['f1']:.4f}")
                
                if config["model_type"] == "bert":
                    # Save results
                    save_model_results("bert", dataset_name, bert_metrics, save_dir=f"{experiment_dir}/results")
                    
                    # Store model and metrics
                    model_results[f"bert_{dataset_name}"] = {
                        "model": bert_model,
                        "tokenizer": tokenizer,
                        "metrics": bert_metrics,
                        "test_data": test_data
                    }
                elif config["model_type"] == "distil":
                    # Save results
                    save_model_results("distilbert", dataset_name, bert_metrics, save_dir=f"{experiment_dir}/results")
                    
                    # Store model and metrics
                    model_results[f"distilbert_{dataset_name}"] = {
                        "model": bert_model,
                        "tokenizer": tokenizer,
                        "metrics": bert_metrics,
                        "test_data": test_data
                    }
                
                print(f"{config["model_type"]} model for {dataset_name} completed successfully")
                
            except Exception as e:
                print(f"Error training BERT model for {dataset_name}: {e}")
                traceback.print_exc()
    
    print("\n" + "="*50)
    print(f"BASELINE MODEL TRAINING COMPLETED: {len(model_results)} models trained")
    print("="*50)
    
    return model_results

def run_adversarial_attacks(model_results, config):
    """Run adversarial attacks on trained models."""
    if not attacks_available:
        print("\n" + "="*50)
        print("ADVERSARIAL ATTACKS FUNCTIONALITY NOT AVAILABLE")
        print("Please make sure adversarial_attacks.py is in the same directory")
        print("="*50)
        return {}
    
    experiment_dir = config["experiment_dir"] + "/"+config["attack_type"] + "_testing"
    attack_results = {}
    
    print("\n" + "="*50)
    print("RUNNING ADVERSARIAL ATTACKS")
    print("="*50)
    
    for model_key, model_data in model_results.items():
        model = model_data["model"]
        tokenizer = model_data["tokenizer"]
        test_data = model_data["test_data"]
        
        model_name, dataset_name = model_key.split("_", 1)
        
        print(f"\nRunning adversarial attacks on {model_name} for {dataset_name} dataset")
        
        if config["attack_type"] == "all":
            attack_types = ["char", "word", "homoglyph", "deepwordbug"]
        else:
            attack_types = [config["attack_type"]]
        
        for attack_type in attack_types:
            try:
                print(f"  Running {attack_type} attack")
                
                # Run attack
                attack_result = run_adversarial_attack(
                    model, tokenizer, test_data,
                    attack_type=attack_type,
                    num_examples=config["num_examples"]
                )
                
                # Save attack results
                save_attack_results(
                    model_name, dataset_name, attack_result,
                    save_dir=f"{experiment_dir}/adversarial_examples"
                )
                
                # Store results
                attack_results[f"{model_key}_{attack_type}"] = attack_result
                
                # Visualize examples if available
                if visualizations_available:
                    example_file = f"{experiment_dir}/adversarial_examples/{model_name}_{dataset_name}_{attack_type}_examples.csv"
                    if os.path.exists(example_file):
                        visualize_text_perturbations(
                            example_file, 
                            save_dir=f"{experiment_dir}/visualizations",
                            max_examples=5
                        )
                
                print(f"  {attack_type} attack completed: Success rate = {attack_result['attack_success_rate']:.2f}%")
                
            except Exception as e:
                print(f"  Error running {attack_type} attack: {e}")
                traceback.print_exc()
    
    print("\n" + "="*50)
    print(f"ADVERSARIAL ATTACKS COMPLETED: {len(attack_results)} attacks run")
    print("="*50)
    
    return attack_results

def visualize_all_results(experiment_dir):
    """Generate visualizations for all results in the experiment."""
    if not visualizations_available:
        print("\n" + "="*50)
        print("VISUALIZATION FUNCTIONALITY NOT AVAILABLE")
        print("Please make sure visualization_scripts.py is in the same directory")
        print("="*50)
        return
    
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    try:
        # Visualize attack success rates
        print("\nVisualizing attack success rates...")
        attack_metrics_files = glob.glob(f"{experiment_dir}/adversarial_examples/*_metrics.csv")
        if attack_metrics_files:
            visualize_attack_success_rates(
                results_dir=f"{experiment_dir}/adversarial_examples",
                save_dir=f"{experiment_dir}/visualizations"
            )
            print("Attack success rate visualization completed")
        else:
            print("No attack metrics files found")
        
        # Visualize model performance degradation
        print("\nVisualizing model performance degradation...")
        baseline_files = glob.glob(f"{experiment_dir}/results/*_results.csv")
        if baseline_files and attack_metrics_files:
            visualize_model_performance_degradation(
                baseline_dir=f"{experiment_dir}/results",
                attack_dir=f"{experiment_dir}/adversarial_examples",
                save_dir=f"{experiment_dir}/visualizations"
            )
            print("Model performance degradation visualization completed")
        else:
            print("Missing files for performance degradation visualization")
        
        # Visualize perturbation impact by position
        print("\nVisualizing perturbation impact by position...")
        example_files = glob.glob(f"{experiment_dir}/adversarial_examples/*_examples.csv")
        if example_files:
            visualize_perturbation_impact_by_position(
                example_files,
                save_dir=f"{experiment_dir}/visualizations"
            )
            print("Perturbation impact visualization completed")
        else:
            print("No example files found")
    
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        traceback.print_exc()
    
    print("\n" + "="*50)
    print("VISUALIZATION COMPLETED")
    print("="*50)

def main():
    """Main function to run the experiment."""
       
    # Parse command line arguments
    args = parse_args()
    
    # Setup experiment
    try:
        config = setup_experiment(args)
    except Exception as e:
        print(f"Error setting up experiment: {e}")
        traceback.print_exc()
        return
    
    experiment_dir = config["experiment_dir"] + "/" + config["attack_type"] + "_testing"
    training_experiment_dir = config["experiment_dir"] + "/training"
    
    print(f"\nStarting experiment: {config['timestamp']}")
    print(f"Configuration: {json.dumps(config, indent=2)}")
    
    model_results = {}
    attack_results = {}
    defense_results = {}

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return super(NpEncoder, self).default(obj)

    # Execute requested task
    if config["task"] in ["train_models", "full_pipeline"]:
        try:
            # Train baseline models
            model_results = train_baseline_models(config)
            
            # Save model_results metadata
            if model_results:
                model_results_meta = {}
                for key, value in model_results.items():
                    model_results_meta[key] = {
                        "metrics": value["metrics"],
                        "model_path": f"{training_experiment_dir}/models/{key}"
                    }
                print(model_results_meta)
                with open(f"{training_experiment_dir}/model_results.json", "w") as f:
                    json.dumps(model_results_meta,indent=4,cls=NpEncoder)
                
                print(f"Model results metadata saved to {training_experiment_dir}/model_results.json")
            else:
                print("No models were trained")
        except Exception as e:
            print(f"Error in model training: {e}")
            traceback.print_exc()
    
    elif config["task"] in ["run_attacks", "visualize_results"]:
        # Load previously trained models
        try:
            if config.get("checkpoint_path"):
                print(f"Loading model from checkpoint: {config['checkpoint_path']}")

                if config["model_type"] == "bert":
                    # Load base model
                    model = BertForSequenceClassification.from_pretrained(f"{training_experiment_dir}/models/bert_imdb", num_labels=2)
                    # Load tokenizer
                    tokenizer = BertTokenizer.from_pretrained(f"{training_experiment_dir}/models/bert_imdb")
                elif config["model_type"] == "distil":
                    # Load base model
                    model = DistilBertForSequenceClassification.from_pretrained(f"{training_experiment_dir}/models/distil_imdb", num_labels=2)
                    # Load tokenizer
                    tokenizer = DistilBertTokenizer.from_pretrained(f"{training_experiment_dir}/models/distil_imdb")

                # Load checkpoint
                model.load_state_dict(torch.load(config['checkpoint_path'], map_location=device))
                
                # Load test data
                _, test_data = load_imdb_dataset()
                
                # Create model_results dict with just this model
                model_results = {
                    "bert_imdb": {
                        "model": model,
                        "tokenizer": tokenizer,
                        "metrics": {},  # Empty metrics since we're loading from checkpoint
                        "test_data": test_data
                    }
                }
            
            print(len(model_results["bert_imdb"]["test_data"]))
            
            if not model_results:
                print("No previously trained models found. Please run with --task train_models first.")
                return
            
        # except FileNotFoundError:
        #     print(f"No model_results.json file found in {training_experiment_dir}")
        #     print("Please run with --task train_models first.")
        #     return
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            return
    
    if config["task"] in ["run_attacks", "full_pipeline"]:
        try:
            # Run adversarial attacks
            attack_results = run_adversarial_attacks(model_results, config)
            
            # Save attack_results metadata
            if attack_results:
                attack_results_meta = {}
                for key, value in attack_results.items():
                    attack_results_meta[key] = {
                        "attack_type": value["attack_type"],
                        "attack_success_rate": value["attack_success_rate"],
                        "num_examples": value["num_examples"],
                        "successful_attacks": value["successful_attacks"]
                    }
                
                with open(f"{experiment_dir}/attack_results.json", "w") as f:
                    json.dump(attack_results_meta, f, indent=4)
                
                print(f"Attack results metadata saved to {experiment_dir}/attack_results.json")
            else:
                print("No attacks were run")
        except Exception as e:
            print(f"Error in running attacks: {e}")
            traceback.print_exc()
    
    elif config["task"] in ["apply_defenses", "visualize_results"]:
        # Load previously run attacks
        try:
            print("Loading previously run attack results...")            
            with open(f"{experiment_dir}/attack_results.json", "r") as f:
                attack_results_meta = json.load(f)
            
            for key, meta in attack_results_meta.items():
                model_key, attack_type = key.rsplit("_", 1)
                model_name, dataset_name = model_key.split("_", 1)
                
                # Load the attack examples
                examples_file = f"{experiment_dir}/adversarial_examples/{model_name}_{dataset_name}_{attack_type}_examples.csv"
                if os.path.exists(examples_file):
                    examples_df = pd.read_csv(examples_file)
                    
                    attack_results[key] = {
                        "attack_type": meta["attack_type"],
                        "attack_success_rate": meta["attack_success_rate"],
                        "num_examples": meta["num_examples"],
                        "successful_attacks": meta["successful_attacks"],
                        "original_texts": examples_df["original_text"].tolist(),
                        "perturbed_texts": examples_df["perturbed_text"].tolist(),
                        "original_labels": examples_df["original_label"].tolist(),
                        "perturbed_labels": examples_df["perturbed_label"].tolist()
                    }
                    
                    print(f"Attack results for {key} loaded successfully")
            
            if not attack_results:
                print("No previously run attacks found. Please run with --task run_attacks first.")
                if config["task"] == "apply_defenses":
                    return
            
        except FileNotFoundError:
            print(f"No attack_results.json file found in {experiment_dir}")
            print("Please run with --task run_attacks first.")
            if config["task"] == "apply_defenses":
                return
        except Exception as e:
            print(f"Error loading attack results: {e}")
            traceback.print_exc()
            if config["task"] == "apply_defenses":
                return
    
    
    if config["task"] in ["visualize_results", "full_pipeline"]:
        try:
            # Generate visualizations
            visualize_all_results(experiment_dir)
        except Exception as e:
            print(f"Error in generating visualizations: {e}")
            traceback.print_exc()
    
    print(f"\n" + "="*50)
    print(f"EXPERIMENT COMPLETED: {config['timestamp']}")
    print(f"Results saved to: {experiment_dir}")
    print("="*50)

if __name__ == "__main__":
    # Add support for visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Warning: Visualization libraries not found. Please install matplotlib and seaborn.")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        traceback.print_exc()
        print("\nPlease check the error message and fix the issue before retrying.")

            