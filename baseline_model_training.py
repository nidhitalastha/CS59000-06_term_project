# Baseline Model Training (LSTM, CNN, BERT)

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device for model training: {device}")

# LSTM Model for Sentiment Analysis and Hate Speech Detection
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

# CNN Model for Sentiment Analysis and Hate Speech Detection
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, 
                      kernel_size=(fs, embedding_dim)) 
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def train_lstm_model_pytorch(train_dataloader, val_dataloader, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1, n_layers=2, epochs=5):
    """
    Train an LSTM model using PyTorch.
    """
    # Initialize model
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers)
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                text, labels = batch
                text, labels = text.to(device), labels.to(device)
                
                predictions = model(text).squeeze(1)
                loss = criterion(predictions, labels.float())
                val_loss += loss.item()
                
                preds = torch.round(torch.sigmoid(predictions))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Print epoch statistics
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
    
    return model

def train_cnn_model_pytorch(train_dataloader, val_dataloader, vocab_size, embedding_dim=100, n_filters=100, filter_sizes=[3, 4, 5], output_dim=1, epochs=5):
    """
    Train a CNN model using PyTorch.
    """
    # Initialize model
    model = CNNModel(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim)
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    
    # Training loop (similar to LSTM)
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Training)"):
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            
            optimizer.zero_grad()
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation phase (similar to LSTM)
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{epochs} (Validation)"):
                text, labels = batch
                text, labels = text.to(device), labels.to(device)
                
                predictions = model(text).squeeze(1)
                loss = criterion(predictions, labels.float())
                val_loss += loss.item()
                
                preds = torch.round(torch.sigmoid(predictions))
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)
        
        # Print epoch statistics
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
    
    return model

def train_bert_model(train_dataloader, val_dataloader, num_labels=2, epochs=4):
    """
    Train a BERT model for sequence classification.
    Args:
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
        num_labels: Number of output labels
        epochs: Number of training epochs
    Returns:
        Trained model and evaluation metrics
    """
    # Print GPU info if available
    if torch.cuda.is_available():
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Load pre-trained BERT model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=num_labels
    )
    model.to(device)
    
    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    
    # Total number of training steps
    total_steps = len(train_dataloader) * epochs
    
    # Create learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_dataloader, desc="Training"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                     'attention_mask': batch[1],
                     'labels': batch[2]}
            
            # Clear gradients
            model.zero_grad()
            
            # Forward pass
            outputs = model(**inputs)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradient norm to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        
        # Validation phase
        model.eval()
        val_accuracy = 0
        val_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(val_dataloader, desc="Validation"):
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                     'attention_mask': batch[1],
                     'labels': batch[2]}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            loss = outputs.loss
            logits = outputs.logits
            
            val_loss += loss.item()
            
            # Convert logits to predictions
            preds = torch.argmax(logits, dim=1).flatten()
            labels = inputs['labels'].flatten()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            accuracy = (preds == labels).cpu().numpy().mean() * 100
            val_accuracy += accuracy
        
        avg_val_accuracy = val_accuracy / len(val_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Validation Accuracy: {avg_val_accuracy:.2f}%")
        print(f"Validation Loss: {avg_val_loss}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        
        # Save checkpoint after each epoch
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{checkpoint_dir}/bert_epoch_{epoch+1}.pt")
        print(f"Checkpoint saved to {checkpoint_dir}/bert_epoch_{epoch+1}.pt")
    
    return model

def evaluate_model(model, dataloader):
    """
    Evaluate a PyTorch model on a dataloader.
    Returns accuracy, precision, recall, F1-score.
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids': batch[0],
                 'attention_mask': batch[1]}
        labels = batch[2]
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).flatten()
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_preds,
        'labels': all_labels
    }

def save_model_results(model_name, dataset_name, metrics, save_dir="results"):
    """
    Save model evaluation results to CSV file and create visualization.
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    results = {
        'model': model_name,
        'dataset': dataset_name,
        'accuracy': metrics['accuracy'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }
    
    df = pd.DataFrame([results])
    filename = f"{save_dir}/{model_name}_{dataset_name}_results.csv"
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
    # Save confusion matrix    
    cm = confusion_matrix(metrics['labels'], metrics['predictions'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name} on {dataset_name}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_filename = f"{save_dir}/{model_name}_{dataset_name}_confusion_matrix.png"
    plt.savefig(cm_filename)
    plt.close()
    print(f"Confusion matrix saved to {cm_filename}")
    
    # Save metrics visualization
    plt.figure(figsize=(10, 6))
    metrics_values = [results['accuracy'], results['precision'], results['recall'], results['f1']]
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    plt.bar(metrics_names, metrics_values, color='skyblue')
    plt.ylim(0, 1.0)
    plt.title(f"{model_name} Performance on {dataset_name}")
    plt.ylabel('Score')
    
    for i, v in enumerate(metrics_values):
        plt.text(i, v + 0.05, f"{v:.3f}", ha='center')
    
    metrics_filename = f"{save_dir}/{model_name}_{dataset_name}_metrics.png"
    plt.savefig(metrics_filename)
    plt.close()
    print(f"Metrics visualization saved to {metrics_filename}")
    
    return results

# Test function to make sure everything works
def test_model_training():
    """Test a small model training run to ensure functionality."""
    
    print("Testing model training with tiny dataset...")
    
    # Create tiny fake dataset
    input_ids = torch.randint(0, 1000, (100, 32))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, 2, (100,))
    
    # Create DataLoader
    dataset = TensorDataset(input_ids, attention_mask, labels)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=8
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        sampler=RandomSampler(val_dataset),
        batch_size=8
    )
    
    # Initialize tiny BERT model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=2,
        hidden_dropout_prob=0.1
    )
    
    # Train for 1 mini-epoch
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    # Single batch training test
    model.train()
    batch = next(iter(train_dataloader))
    batch = tuple(t.to(device) for t in batch)
    
    inputs = {
        'input_ids': batch[0],
        'attention_mask': batch[1],
        'labels': batch[2]
    }
    
    optimizer.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    
    print(f"✓ Test training successful. Loss: {loss.item()}")
    
    # Test evaluation
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_dataloader))
        batch = tuple(t.to(device) for t in batch)
        
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1]
        }
        
        labels = batch[2]
        outputs = model(**inputs)
        logits = outputs.logits
        
    print(f"✓ Test evaluation successful. Logits shape: {logits.shape}")
    
    return True

if __name__ == "__main__":
    # Run tests if this file is executed directly
    test_model_training()
