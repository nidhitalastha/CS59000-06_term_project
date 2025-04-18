# Data Loading and Preprocessing

import re
import torch
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

def load_imdb_dataset():
    """
    Load the IMDB Movie Reviews dataset for sentiment analysis.
    Returns train and test datasets.
    """
    print("Loading IMDB dataset...")
    dataset = load_dataset("imdb")
    train_data = [(text, label) for text, label in zip(dataset["train"]["text"], dataset["train"]["label"])]
    test_data = [(text, label) for text, label in zip(dataset["test"]["text"], dataset["test"]["label"])]
    
    print(f"IMDB dataset loaded: {len(train_data)} training samples, {len(test_data)} test samples")
    return train_data, test_data

def load_jigsaw_dataset(max_samples=25000):
    """
    Load the Jigsaw Toxic Comment dataset for hate speech detection.
    Limit samples to manage computational resources.
    Returns train and test datasets.
    """
    print("Loading Jigsaw Toxic Comment dataset...")
    try:
        # Use the correct dataset name
        dataset = load_dataset("jigsaw_toxicity_pred")
        
        # Filter to binary toxicity classification
        toxicity_threshold = 0.5
        train_texts = dataset["train"]["comment_text"][:max_samples]
        train_labels = [1 if toxicity > toxicity_threshold else 0 
                       for toxicity in dataset["train"]["toxicity"][:max_samples]]
        
        # Create train/test split
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            train_texts, train_labels, test_size=0.2, random_state=RANDOM_SEED, stratify=train_labels
        )
        
        train_data = list(zip(train_texts, train_labels))
        test_data = list(zip(test_texts, test_labels))
        
        print(f"Jigsaw dataset loaded: {len(train_data)} training samples, {len(test_data)} test samples")
        return train_data, test_data
    except Exception as e:
        print(f"Error loading Jigsaw dataset: {e}")
        # Handle the error gracefully
        print("Using a placeholder dataset instead. Please check your internet connection or dataset availability.")
        # Return a small placeholder dataset
        return [("This is toxic content", 1), ("This is normal content", 0)], [("More toxic content", 1), ("More normal content", 0)]

def preprocess_text(text, remove_stopwords=False):
    """
    Basic text preprocessing function.
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
    Returns:
        Preprocessed text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        text = ' '.join([word for word in word_tokens if word not in stop_words])
    
    return text

def prepare_data_for_bert(data, tokenizer, max_seq_length=128):
    """
    Prepare data for BERT model.
    Args:
        data: List of (text, label) tuples
        tokenizer: BERT tokenizer
        max_seq_length: Maximum sequence length
    Returns:
        Encoded input features and labels as tensors
    """
    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    
    # Tokenize and encode sequences
    encoded_inputs = tokenizer(
        texts,
        max_length=max_seq_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # Convert labels to tensor
    labels_tensor = torch.tensor(labels)
    
    return encoded_inputs, labels_tensor

def create_dataloaders(input_ids, attention_mask, labels, batch_size=8):
    """
    Create PyTorch DataLoaders for training and evaluation.
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Labels
        batch_size: Batch size
    Returns:
        Train and validation DataLoaders
    """

    # Set random seed for reproducibility
    RANDOM_SEED = 42

    # Split data into train and validation sets
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        input_ids, attention_mask, labels, 
        test_size=0.1, 
        random_state=RANDOM_SEED, 
        stratify=labels
    )
    
    # Create DataLoaders
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    
    return train_dataloader, val_dataloader
