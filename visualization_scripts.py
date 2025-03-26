# Visualization Scripts
import re
import matplotlib.pyplot as plt
import seaborn as sns
import difflib
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.colors as mcolors
from matplotlib.ticker import PercentFormatter

def visualize_attack_success_rates(results_dir="adversarial_examples", save_dir="visualizations"):
    """
    Visualize attack success rates across different models and attack types.
    
    Args:
        results_dir: Directory containing attack metrics CSV files
        save_dir: Directory to save visualizations
    """
    # Load all metrics files
    metrics_files = glob.glob(f"{results_dir}/*_metrics.csv")
    
    all_metrics = []
    for file in metrics_files:
        df = pd.read_csv(file)
        all_metrics.append(df)
    
    if not all_metrics:
        print("No metrics files found.")
        return
    
    metrics_df = pd.concat(all_metrics)
    
    # Plot attack success rates by model and attack type
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(
        x='model',
        y='attack_success_rate',
        hue='attack_type',
        data=metrics_df
    )
    chart.set_title('Attack Success Rate by Model and Attack Type', fontsize=16)
    chart.set_xlabel('Model', fontsize=14)
    chart.set_ylabel('Attack Success Rate (%)', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/attack_success_rates.png")
    plt.close()
    
    # Plot attack success rates by dataset
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(
        x='dataset',
        y='attack_success_rate',
        hue='model',
        data=metrics_df
    )
    chart.set_title('Attack Success Rate by Dataset and Model', fontsize=16)
    chart.set_xlabel('Dataset', fontsize=14)
    chart.set_ylabel('Attack Success Rate (%)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dataset_success_rates.png")
    plt.close()

def visualize_model_performance_degradation(baseline_dir="results", attack_dir="adversarial_examples", save_dir="visualizations"):
    """
    Visualize model performance degradation after adversarial attacks.
    
    Args:
        baseline_dir: Directory containing baseline model results
        attack_dir: Directory containing attack results
        save_dir: Directory to save visualizations
    """
    # Load baseline results
    baseline_files = glob.glob(f"{baseline_dir}/*_results.csv")
    
    baseline_results = []
    for file in baseline_files:
        df = pd.read_csv(file)
        baseline_results.append(df)
    
    if not baseline_results:
        print("No baseline results found.")
        return
    
    baseline_df = pd.concat(baseline_results)
    baseline_df['condition'] = 'baseline'
    
    # Load attack metrics to get performance after attack
    attack_files = glob.glob(f"{attack_dir}/*_metrics.csv")
    
    # Calculate accuracy after attack (100 - attack_success_rate)
    attack_results = []
    for file in attack_files:
        df = pd.read_csv(file)
        # Create a copy of metrics with adjusted accuracy
        adjusted_df = df.copy()
        
        # Find corresponding baseline accuracy
        for _, row in df.iterrows():
            model = row['model']
            dataset = row['dataset']
            baseline_acc = baseline_df[(baseline_df['model'] == model) & 
                                      (baseline_df['dataset'] == dataset)]['accuracy'].values
            
            if len(baseline_acc) > 0:
                # Calculate new accuracy after attack
                adjusted_df['accuracy'] = baseline_acc[0] * (1 - row['attack_success_rate']/100)
                adjusted_df['condition'] = f"after_{row['attack_type']}"
                attack_results.append(adjusted_df)
    
    if not attack_results:
        print("No attack results found that match baseline models.")
        return
    
    attack_df = pd.concat(attack_results)
    
    # Combine baseline and attack results
    combined_df = pd.concat([baseline_df, attack_df])
    
    # Plot accuracy degradation by model
    plt.figure(figsize=(14, 10))
    chart = sns.barplot(
        x='model',
        y='accuracy',
        hue='condition',
        data=combined_df
    )
    chart.set_title('Model Accuracy Before and After Attacks', fontsize=16)
    chart.set_xlabel('Model', fontsize=14)
    chart.set_ylabel('Accuracy', fontsize=14)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/accuracy_degradation.png")
    plt.close()
    
    # Plot accuracy degradation by dataset
    plt.figure(figsize=(12, 8))
    chart = sns.barplot(
        x='dataset',
        y='accuracy',
        hue='condition',
        data=combined_df
    )
    chart.set_title('Accuracy by Dataset Before and After Attacks', fontsize=16)
    chart.set_xlabel('Dataset', fontsize=14)
    chart.set_ylabel('Accuracy', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/dataset_accuracy_degradation.png")
    plt.close()

def visualize_text_perturbations(example_file, save_dir="visualizations", max_examples=5):
    """
    Visualize text perturbations from adversarial attacks.
    
    Args:
        example_file: CSV file containing original and perturbed text examples
        save_dir: Directory to save visualizations
        max_examples: Maximum number of examples to visualize
    """
    
    # Load examples
    examples_df = pd.read_csv(example_file)
    
    if len(examples_df) == 0:
        print("No examples found in file.")
        return
    
    # Limit number of examples
    examples_df = examples_df.head(max_examples)
    
    # Create a figure for each example
    for i, row in examples_df.iterrows():
        original_text = row['original_text']
        perturbed_text = row['perturbed_text']
        original_label = row['original_label']
        perturbed_label = row['perturbed_label']
        
        # Create a diff between original and perturbed text
        diff = difflib.ndiff(original_text.split(), perturbed_text.split())
        
        # Extract perturbed words
        diff_list = list(diff)
        perturbed_words = []
        for j, s in enumerate(diff_list):
            if s.startswith('- '):
                original_word = s[2:]
                # Look for replacement
                if j+1 < len(diff_list) and diff_list[j+1].startswith('+ '):
                    replacement = diff_list[j+1][2:]
                    perturbed_words.append((original_word, replacement))
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.axis('off')
        
        # Add title with labels
        title = f"Example {i+1}: Original Label = {original_label}, Perturbed Label = {perturbed_label}"
        plt.title(title, fontsize=14)
        
        # Display original text
        original_display = f"Original: {original_text}"
        plt.text(0.05, 0.7, original_display, wrap=True, fontsize=12)
        
        # Display perturbed text
        perturbed_display = f"Perturbed: {perturbed_text}"
        plt.text(0.05, 0.4, perturbed_display, wrap=True, fontsize=12)
        
        # Display perturbation details
        if perturbed_words:
            perturbation_text = "Perturbations:\n"
            for original, replacement in perturbed_words:
                perturbation_text += f"'{original}' → '{replacement}'\n"
            plt.text(0.05, 0.1, perturbation_text, fontsize=12)
        
        # Save figure
        filename = f"{save_dir}/example_{i+1}_perturbation.png"
        plt.savefig(filename, bbox_inches='tight')
        plt.close()
    
    print(f"{len(examples_df)} example visualizations saved to {save_dir}")

def visualize_perturbation_impact_by_position(example_files, save_dir="visualizations"):
    """
    Analyze and visualize the impact of perturbations based on their position in text.
    
    Args:
        example_files: List of CSV files containing perturbation examples
        save_dir: Directory to save visualizations
    """
    all_examples = []
    
    # Load all example files
    for file in example_files:
        df = pd.read_csv(file)
        # Extract model and attack type from filename
        match = re.search(r'(.+?)_(.+?)_(.+?)_examples.csv', file)
        if match:
            model = match.group(1)
            dataset = match.group(2)
            attack_type = match.group(3)
            
            df['model'] = model
            df['dataset'] = dataset
            df['attack_type'] = attack_type
            
            all_examples.append(df)
    
    if not all_examples:
        print("No example files found.")
        return
    
    examples_df = pd.concat(all_examples)
    
    # Analyze position of perturbations
    position_data = []
    
    for _, row in examples_df.iterrows():
        original_words = row['original_text'].split()
        perturbed_words = row['perturbed_text'].split()
        
        # Skip if word counts don't match (for simplicity)
        if len(original_words) != len(perturbed_words):
            continue
        
        # Find positions of perturbations
        for i, (orig, pert) in enumerate(zip(original_words, perturbed_words)):
            if orig != pert:
                # Calculate relative position (0-1)
                rel_position = i / len(original_words)
                
                position_data.append({
                    'model': row['model'],
                    'attack_type': row['attack_type'],
                    'rel_position': rel_position,
                    'position_bucket': int(rel_position * 10) / 10  # 0.0, 0.1, 0.2, etc.
                })
    
    if not position_data:
        print("No position data could be extracted.")
        return
    
    position_df = pd.DataFrame(position_data)
    
    # Plot perturbation distribution by position
    plt.figure(figsize=(12, 8))
    chart = sns.histplot(
        data=position_df,
        x='position_bucket',
        hue='attack_type',
        multiple='stack',
        discrete=True
    )
    chart.set_title('Distribution of Perturbations by Position in Text', fontsize=16)
    chart.set_xlabel('Relative Position (0=start, 1=end)', fontsize=14)
    chart.set_ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/perturbation_position_distribution.png")
    plt.close()
    
    # Plot position distribution by model
    plt.figure(figsize=(12, 8))
    chart = sns.histplot(
        data=position_df,
        x='position_bucket',
        hue='model',
        multiple='dodge',
        discrete=True
    )
    chart.set_title('Distribution of Perturbations by Model and Position', fontsize=16)
    chart.set_xlabel('Relative Position (0=start, 1=end)', fontsize=14)
    chart.set_ylabel('Frequency', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/model_position_distribution.png")
    plt.close()
