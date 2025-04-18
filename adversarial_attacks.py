# Adversarial Attack Implementation

import textattack
import torch
import pandas as pd
from textattack.attack_recipes import TextFoolerJin2019, DeepWordBugGao2018, BERTAttackLi2020
from textattack.datasets import Dataset
from textattack.models.wrappers import PyTorchModelWrapper, HuggingFaceModelWrapper
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.goal_functions import UntargetedClassification
from textattack.transformations import WordSwapEmbedding, WordSwapHomoglyphSwap, WordSwapNeighboringCharacterSwap
from textattack.search_methods import GreedyWordSwapWIR
from textattack.attack_results import SuccessfulAttackResult


# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device for adversarial attacks: {device}")

class CustomModelWrapper(PyTorchModelWrapper):
    """
    Wrapper for PyTorch models to use with TextAttack.
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, text_inputs):
        # Tokenize inputs
        encoded_inputs = self.tokenizer(
            text_inputs, 
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded_inputs["input_ids"].to(device)
        attention_mask = encoded_inputs["attention_mask"].to(device)
        
        # Get model predictions
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
        
        return logits.cpu().detach().numpy()

def create_custom_attack(attack_type, model_wrapper):
    """
    Create a custom adversarial attack based on attack type.
    
    Args:
        attack_type: Type of attack ('char', 'word', or 'sentence')
        model_wrapper: TextAttack model wrapper
    
    Returns:
        TextAttack Attack object
    """
    # Goal function - untargeted classification
    goal_function = UntargetedClassification(model_wrapper)
    
    # Common constraints
    constraints = [
        RepeatModification(),  # Don't modify the same word twice
        StopwordModification()  # Don't modify stopwords
    ]
    
    # Choose transformation based on attack type
    if attack_type == 'char':
        # Character-level attack
        transformation = WordSwapNeighboringCharacterSwap()
    elif attack_type == 'word':
        # Word-level attack (synonym replacement)
        transformation = WordSwapEmbedding()
    elif attack_type == 'homoglyph':
        # Homoglyph attack (visually similar characters)
        transformation = WordSwapHomoglyphSwap()
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    # Search method
    search_method = GreedyWordSwapWIR(wir_method="delete")
    
    # Create attack
    attack = textattack.Attack(
        goal_function=goal_function,
        constraints=constraints,
        transformation=transformation,
        search_method=search_method
    )
    
    return attack

def run_adversarial_attack(model, tokenizer, test_data, attack_type='word', num_examples=1000):
    """
    Run adversarial attack on the specified model.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        test_data: List of (text, label) tuples
        attack_type: Type of attack ('char', 'word', or 'homoglyph')
        num_examples: Number of test examples to attack
    
    Returns:
        Dictionary with attack results
    """
    # Create model wrapper
    model = model.to(device)
    model_wrapper = CustomModelWrapper(model, tokenizer)
    
    # Create dataset with subset of test data
    limited_test_data = test_data[:num_examples]
    dataset = Dataset(
    [(text, label) for text, label in limited_test_data]
    )

    print(len(dataset))
    
    # Create attack
    if attack_type == 'textfooler':
        attack = TextFoolerJin2019.build(model_wrapper)
    elif attack_type == 'deepwordbug':
        attack = DeepWordBugGao2018.build(model_wrapper)
    elif attack_type == 'bert-attack':
        attack = BERTAttackLi2020.build(model_wrapper)
    else:
        attack = create_custom_attack(attack_type, model_wrapper)
    
    attack_args = textattack.AttackArgs(
    num_examples=len(dataset),
)

    # Run attack
    attacker = textattack.Attacker(attack, dataset, attack_args)
    results = attacker.attack_dataset()
    
    # Calculate attack success rate
    successful_attacks = len([r for r in results if isinstance(r, SuccessfulAttackResult)])
    attack_success_rate = successful_attacks / len(results) * 100
    
    # Collect original and adversarial examples
    original_texts = []
    perturbed_texts = []
    original_labels = []
    perturbed_labels = []
    
    for result in results:
        if isinstance(result, SuccessfulAttackResult):
            original_texts.append(result.original_result.attacked_text.text)
            perturbed_texts.append(result.perturbed_result.attacked_text.text)
            original_labels.append(result.original_result.output)
            perturbed_labels.append(result.perturbed_result.output)
    
    return {
        'attack_type': attack_type,
        'num_examples': len(results),
        'successful_attacks': successful_attacks,
        'attack_success_rate': attack_success_rate,
        'original_texts': original_texts,
        'perturbed_texts': perturbed_texts,
        'original_labels': original_labels,
        'perturbed_labels': perturbed_labels
    }

def save_attack_results(model_name, dataset_name, attack_results, save_dir="adversarial_examples"):
    """
    Save adversarial attack results to files.
    """
    attack_type = attack_results['attack_type']
    
    # Save attack success metrics
    metrics = {
        'model': model_name,
        'dataset': dataset_name,
        'attack_type': attack_type,
        'num_examples': attack_results['num_examples'],
        'successful_attacks': attack_results['successful_attacks'],
        'attack_success_rate': attack_results['attack_success_rate']
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_filename = f"{save_dir}/{model_name}_{dataset_name}_{attack_type}_metrics.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    
    # Save examples
    examples = []
    for i in range(len(attack_results['original_texts'])):
        example = {
            'original_text': attack_results['original_texts'][i],
            'perturbed_text': attack_results['perturbed_texts'][i],
            'original_label': attack_results['original_labels'][i],
            'perturbed_label': attack_results['perturbed_labels'][i]
        }
        examples.append(example)
    
    examples_df = pd.DataFrame(examples)
    examples_filename = f"{save_dir}/{model_name}_{dataset_name}_{attack_type}_examples.csv"
    examples_df.to_csv(examples_filename, index=False)
    
    print(f"Attack results saved to {metrics_filename} and {examples_filename}")
