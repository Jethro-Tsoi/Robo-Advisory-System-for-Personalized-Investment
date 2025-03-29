#!/usr/bin/env python
# coding: utf-8

# Gemma 3 Model Training with LoRA
# 
# This script implements the training pipeline for Google's Gemma 3 model using LoRA (Low-Rank Adaptation) for efficient fine-tuning.
# 
# Features:
# 1. LoRA implementation
# 2. Multi-metric early stopping
# 3. Evaluation metrics tracking
# 4. Multimodal capabilities (text classification)

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from transformers.modeling_utils import PreTrainedModel
from torch import nn
from huggingface_hub import login

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Dataset Preparation
class FinancialTweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Early Stopping Implementation
class EarlyStoppingCallback:
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metrics = None
        self.early_stop = False

    def __call__(self, metrics):
        if self.best_metrics is None:
            self.best_metrics = metrics
            return False

        # Check if any metric improved by min_delta
        improved = False
        for metric, value in metrics.items():
            if value > self.best_metrics[metric] + self.min_delta:
                improved = True
                self.best_metrics = metrics
                break

        if not improved:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0

        return self.early_stop

def calculate_metrics(predictions, labels):
    """Calculate multiple evaluation metrics"""
    pred_labels = np.argmax(predictions, axis=1)

    # Basic metrics
    accuracy = accuracy_score(labels, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred_labels, average='weighted'
    )

    # Additional metrics
    kappa = cohen_kappa_score(labels, pred_labels)
    mcc = matthews_corrcoef(labels, pred_labels)

    # ROC-AUC (multi-class)
    try:
        roc_auc = roc_auc_score(labels, predictions, multi_class='ovr')
    except:
        roc_auc = 0.0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'kappa': kappa,
        'mcc': mcc,
        'roc_auc': roc_auc
    }

# Add a sequence classification head on top
class GemmaForSequenceClassification(nn.Module):
    def __init__(self, base_model, num_labels=5):
        nn.Module.__init__(self)
        self.base_model = base_model
        self.num_labels = num_labels
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Get the hidden states from the base model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # Use the first token representation for classification
        cls_output = hidden_states[:, 0, :]
        
        # Apply the classification head
        logits = self.classifier(cls_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        # Standardized output structure
        class SequenceClassifierOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        return SequenceClassifierOutput(loss, logits)

# Function to load a fine-tuned model
def load_finetuned_model(adapter_path, base_model, hf_token=None):
    from peft import PeftModel, PeftConfig
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    
    # Use our custom classification wrapper
    model = GemmaForSequenceClassification(base_model, num_labels=5)
    
    # Load the PEFT adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    
    return model

# Example of loading and using the model
def predict_sentiment(text, model, tokenizer, label_map_reverse):
    model.eval()
    encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(model.device)

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()

    return {
        "sentiment": label_map_reverse[predicted_class],
        "probabilities": {label_map_reverse[i]: prob.item() for i, prob in enumerate(probabilities[0])}
    }

def main():
    # Set up data paths - use Vertex AI env vars when available 
    data_dir = os.environ.get("AIP_DATA_DIR", "./data")
    model_dir = os.environ.get("AIP_MODEL_DIR", "./models/gemma3")
    
    # Create output directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load labeled data - adjust path for Vertex AI or Cloud Storage if needed
    data_file = os.path.join(data_dir, "all_labeled_tweets.csv")
    
    # Check if file exists and is accessible
    if os.path.exists(data_file):
        print(f"Loading data from {data_file}")
        df = pd.read_csv(data_file)
    else:
        # Try Cloud Storage path if local file not found
        print(f"Data file {data_file} not found, trying Cloud Storage")
        try:
            # Replace with your GCS bucket path if using Cloud Storage
            gcs_path = "gs://your-bucket-name/data/all_labeled_tweets.csv"
            df = pd.read_csv(gcs_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return
    
    # Convert labels to numeric
    label_map = {
        'STRONGLY_POSITIVE': 0,
        'POSITIVE': 1,
        'NEUTRAL': 2,
        'NEGATIVE': 3,
        'STRONGLY_NEGATIVE': 4
    }
    df['label'] = df['sentiment'].map(label_map)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['description'].values, df['label'].values,
        test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer for Gemma 3
    model_name = "google/gemma-3-12b-pt"
    
    # Add Hugging Face authentication
    # Get token from environment variables
    hf_token = os.environ.get("HF_TOKEN", None)
    if hf_token:
        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub")
    else:
        print("⚠️ Hugging Face token not found! Please set the HF_TOKEN environment variable.")
        print("This will likely cause authentication errors when accessing Gemma 3 models.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    # Create datasets
    train_dataset = FinancialTweetDataset(train_texts, train_labels, tokenizer)
    val_dataset = FinancialTweetDataset(val_texts, val_labels, tokenizer)
    
    # Initialize model with LoRA config
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    
    # Wrap the model with our classification head
    model = GemmaForSequenceClassification(model, num_labels=5)  # 5 classes for the sentiment labels

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"], # Target the attention layers
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM  # Changed from SEQ_CLS to CAUSAL_LM for Gemma 3
    )
    
    # Prepare model for LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters percentage
    model.print_trainable_parameters()
    
    # Training parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Training hyperparameters
    num_epochs = 5
    batch_size = 8  # Smaller batch size due to model size
    learning_rate = 1e-5  # Lower learning rate for stability
    weight_decay = 0.01
    warmup_steps = 100
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Total steps for scheduler
    total_steps = len(train_loader) * num_epochs
    
    # Initialize optimizer and scheduler
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize early stopping
    early_stopping = EarlyStoppingCallback(patience=3)
    
    # Training loop
    best_metrics = None
    best_model_state = None
    training_history = {"loss": [], "val_metrics": []}
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0
        epoch_steps = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            epoch_steps += 1
        
        avg_loss = total_loss / epoch_steps
        training_history["loss"].append(avg_loss)
        print(f"\nAverage training loss: {avg_loss:.4f}")
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_preds.append(outputs.logits.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        predictions = np.vstack(all_preds)
        true_labels = np.concatenate(all_labels)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, true_labels)
        training_history["val_metrics"].append(metrics)
        
        print("\nValidation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Early stopping check
        if early_stopping(metrics):
            print("\nEarly stopping triggered!")
            break
        
        # Save best model
        if best_metrics is None or metrics['f1'] > best_metrics['f1']:
            best_metrics = metrics
            # For PEFT models, we save the state_dict of the adapter instead of the entire model
            best_model_state = {k: v.clone() for k, v in model.state_dict().items() if "lora" in k}
            print("New best model saved!")
    
    # Save final model and metrics
    model.save_pretrained(os.path.join(model_dir, "gemma3_lora_adapter"))
    tokenizer.save_pretrained(os.path.join(model_dir, "gemma3_lora_adapter"))
    
    # Save training history
    pd.DataFrame([{
        "epoch": i+1, 
        "loss": loss, 
        **metrics
    } for i, (loss, metrics) in enumerate(zip(training_history["loss"], training_history["val_metrics"]))])\
        .to_csv(os.path.join(model_dir, 'training_history.csv'), index=False)
    
    # Save final performance metrics
    metrics_df = pd.DataFrame([best_metrics])
    metrics_df.to_csv(os.path.join(model_dir, 'metrics.csv'), index=False)
    
    print(f"Training complete. Model saved to {model_dir}")

if __name__ == "__main__":
    main()

