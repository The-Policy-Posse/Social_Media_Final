# -*- coding: utf-8 -*-
"""
Fine-tuning RoBERTa for Multi-Label Classification on Reddit Posts with Cross-Validation

This script fine-tunes a RoBERTa model for a multi-label text classification task 
using a dataset of labeled Reddit comments. It includes data preprocessing, 
k-fold cross-validation, training, and evaluation on a separate test set.

Key Features:
- Preprocessing of labeled Reddit Posts, including topic normalization and filtering.
- Custom Dataset and DataLoader implementations for handling multi-label data.
- Focal Loss implementation to address class imbalance.
- Dynamic threshold optimization for maximizing evaluation metrics.
- Layer-wise learning rate scheduling for RoBERTa model fine-tuning.
- Early stopping based on validation performance to prevent overfitting.
- k-Fold Cross-Validation with aggregation of metrics.
- Final model evaluation on a separate test set.

Author: dforc
Created on: November 19, 2024
"""

##############
## Imports ##
##############

## Standard Libraries
import os
import ast
import pickle
import logging
import random
from tqdm import tqdm

## Data Handling
import numpy as np
import pandas as pd

## PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

## Transformers and Optimization
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)

## Scikit-Learn for Metrics and Preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    jaccard_score
)

## Stratified K-Fold for Multi-Label Data
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit



##############################
## Configuration and Setup ##
##############################

## Configuration Dictionary
CONFIG = {
    'seed': 47,                               # Seed for reproducibility
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),  # GPU if available
    'max_len': 128,                           # Maximum token length for inputs
    'batch_size': 32,                         # Batch size for training and evaluation
    'epochs': 10,                             # Number of training epochs
    'test_size': 0.15,                        # Proportion of data reserved for testing
    'k_folds': 5,                             # Number of folds for cross-validation
    'early_stopping_patience': 3,             # Early stopping patience
    'fusion_method': 'concat',                # Placeholder for potential future use
    'mlb_save_path': './artifacts/mlb_multi_label.pkl',  # Path to save MultiLabelBinarizer
    'model_save_dir': './artifacts/models',   # Directory to save trained models
    'thresholds_save_dir': './artifacts/thresholds',  # Directory to save thresholds
    'metrics_save_dir': './artifacts/metrics',  # Directory to save metrics
    'labels_csv': '../data/processed/labeled_reddit_posts_processed.csv',  # Path to labeled data
    'primary_model_name': 'roberta-base',     # Pretrained model name
}

## Create necessary directories
for dir_path in [CONFIG['model_save_dir'], CONFIG['thresholds_save_dir'], CONFIG['metrics_save_dir']]:
    os.makedirs(dir_path, exist_ok=True)

## Setup Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


## Prevent adding multiple handlers
if not logger.handlers:
    
    ## File Handler
    file_handler = logging.FileHandler('training.log', mode='a')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    ## Console Handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)


##########################
### Utility Functions  ###
##########################

## Set Seed (Set in Config)
def set_seed(seed=None):
    """
    Sets the seed for reproducibility.
    """
    seed = seed if seed is not None else CONFIG['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Random seeds set to {seed}.")


## Quick Preprocessing
## Note: This should have already been completed in preprocessing
##        Here as a -just in case-
def normalize_topic(topic):
    """
    Normalize topic strings to ensure consistency.
    """
    topic = topic.strip()              # Remove leading/trailing whitespace
    topic = topic.replace("\\/", "/")  # Replace escaped slashes
    topic = topic.lower()              # Convert to lowercase for uniformity
    return topic



############################
###   Dataset Definition ###
############################

class RedditCommentsDataset(Dataset):
    """
    Custom Dataset class for Reddit comments, handling tokenization and label formatting.
    """
    
    ## Initialize the dataset
    def __init__(self, texts, labels, tokenizer, max_len):
        """
        Initializes the dataset with texts and labels.
        
        Args:
            texts (List[str]): List of comment texts.
            labels (np.ndarray): Multi-label binary matrix.
            tokenizer (RobertaTokenizer): Tokenizer for encoding texts.
            max_len (int): Maximum length for tokenization.
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    
    
    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.texts)
    
    
    
    def __getitem__(self, idx):
        """
        Retrieves the tokenized input and corresponding labels for a given index.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            dict: Dictionary containing input_ids, attention_mask, and labels.
        """
        try:
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.float)
            }
        except Exception as e:
            logging.error(f"Error in __getitem__ for index {idx}: {e}")
            raise e

############################
###     Loss Function    ###
############################

class FocalLoss(nn.Module):
    """
    Implementation of Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, gamma=2, alpha=0.25):
        """
        Initializes the FocalLoss with specified gamma and alpha values.
        
        Args:
            gamma (float): Focusing parameter for modulating factor (1 - p).
            alpha (float): Weighting factor for the rare class.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    
    def forward(self, logits, labels):
        """
        Computes the focal loss between logits and labels.
        
        Args:
            logits (torch.Tensor): Predicted unnormalized scores.
            labels (torch.Tensor): Ground truth binary labels.
        
        Returns:
            torch.Tensor: Computed focal loss.
        """
        probs = torch.sigmoid(logits)                                     # Convert logits to probabilities
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)  # Binary cross-entropy loss
        pt = torch.where(labels == 1, probs, 1 - probs)                   # Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss        # Apply focal loss formula
        return focal_loss.mean()                                          # Return mean loss over the batch


##########################
## Helper Functions ##
##########################

#### Set Metric to Optimize Here (F1 or Precision) #####


def find_optimal_thresholds(y_true, y_probs, metric='f1'):
    """
    Finds the optimal threshold for each class to maximize a specified metric (F1 or Precision).
    
    Args:
        y_true (np.ndarray): Ground truth binary labels.
        y_probs (np.ndarray): Predicted probabilities.
        metric (str): Metric to optimize ('f1' or 'precision').
    
    Returns:
        List[float]: List of optimal thresholds for each class.
    """
    thresholds = []
    for i in range(y_true.shape[1]):
        best_threshold = 0.5
        best_metric = 0
        
        ## Iterate over possible thresholds 
        ###### Set linspace Thresholds Here ######  <--->
        for threshold in np.linspace(0.1, 0.9, 81):
            
            ## Apply threshold
            preds = (y_probs[:, i] >= threshold).astype(int)
            if metric == 'f1':
                current_metric = f1_score(y_true[:, i], preds, zero_division=0)
            elif metric == 'precision':
                current_metric = precision_score(y_true[:, i], preds, zero_division=0)
            else:
                current_metric = f1_score(y_true[:, i], preds, zero_division=0)
            if current_metric > best_metric:
                best_metric = current_metric
                best_threshold = threshold
                
        ## Store best threshold for class        
        thresholds.append(best_threshold)
    return thresholds


#### Set Max number of labels per instance here ####

def apply_thresholds_with_limit_dynamic(y_probs, thresholds, max_labels=3):
    """
    Applies thresholds to predicted probabilities to obtain binary predictions,
    ensuring a maximum number of labels per instance.
    
    Args:
        y_probs (np.ndarray): Predicted probabilities.
        thresholds (List[float]): Thresholds for each class.
        max_labels (int): Maximum number of labels to assign per instance.
    
    Returns:
        np.ndarray: Binary predictions after applying thresholds.
    """
    binary_preds = np.zeros_like(y_probs)
    for i in range(y_probs.shape[0]):
        probs = y_probs[i]
        
        ## Get indices sorted by probability in descending order
        sorted_indices = np.argsort(probs)[::-1]
        selected_indices = []
        for idx in sorted_indices:
            if probs[idx] >= thresholds[idx]:
                selected_indices.append(idx)
            if len(selected_indices) == max_labels:
                break
            
        ## Assign label if conditions met    
        binary_preds[i, selected_indices] = 1
    return binary_preds




def eval_model(model, data_loader, device, criterion, max_labels=3, metric_to_optimize='f1'):
    """
    Evaluates the model and computes various metrics on the validation set.
    
    Args:
        model (RobertaForSequenceClassification): The trained model.
        data_loader (DataLoader): DataLoader for the validation set.
        device (torch.device): Device to perform computations on.
        criterion (nn.Module): Loss function.
        max_labels (int): Maximum number of labels to assign per instance.
        metric_to_optimize (str): Metric to optimize when finding thresholds.
    
    Returns:
        dict: Dictionary containing various evaluation metrics.
    """
    
    ## Set model to evaluation mode
    model.eval()
    losses, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            ## Extract logits
            logits = outputs.logits
            loss = criterion(logits, labels)
            losses.append(loss.item())
            
            ## Convert logits to probabilities
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    
    ## Find optimal thresholds
    thresholds = find_optimal_thresholds(all_labels, all_preds, metric=metric_to_optimize)
    
    ## Apply thresholds with limit
    binary_preds = apply_thresholds_with_limit_dynamic(all_preds, thresholds, max_labels=max_labels)
    
    
    ############################# Evaluation Metrics
    #####################
    micro_f1 = f1_score(all_labels, binary_preds, average='micro', zero_division=0)
    macro_f1 = f1_score(all_labels, binary_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, binary_preds, average='weighted', zero_division=0)
    ham_loss = hamming_loss(all_labels, binary_preds)
    jaccard = jaccard_score(all_labels, binary_preds, average='samples', zero_division=0)
    class_f1 = f1_score(all_labels, binary_preds, average=None, zero_division=0)
    class_precision = precision_score(all_labels, binary_preds, average=None, zero_division=0)
    class_recall = recall_score(all_labels, binary_preds, average=None, zero_division=0)
    #####################



    ###################################
    ## Create a formatted metrics table
    metrics_table = f"""
########## Validation Metrics ##########
Validation Loss: {np.mean(losses):.4f}
Micro F1 Score: {micro_f1:.4f}
Macro F1 Score: {macro_f1:.4f}
Weighted F1 Score: {weighted_f1:.4f}
Hamming Loss: {ham_loss:.4f}
Jaccard Index: {jaccard:.4f}

Per-Class Metrics:
{"Class":<10} {"F1":<10} {"Precision":<10} {"Recall":<10}
{"-"*40}
"""
    for i in range(len(class_f1)):
        metrics_table += f"Class {i:<7} {class_f1[i]:<10.4f} {class_precision[i]:<10.4f} {class_recall[i]:<10.4f}\n"
    
    metrics_table += f"Optimal Thresholds: {', '.join([f'{t:.2f}' for t in thresholds])}\n"
    metrics_table += "########################################\n"

    logging.info(metrics_table)

    return {
        'loss': np.mean(losses),
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'hamming_loss': ham_loss,
        'jaccard_index': jaccard,
        'thresholds': thresholds
    }
    ###################################


def ensemble_predict_weighted(ensemble_models, fold_weights, data_loader, device):
    """
    Aggregates predictions from ensemble models using weighted averaging.
    
    Args:
        ensemble_models (List[RobertaForSequenceClassification]): List of trained models.
        fold_weights (List[float]): Corresponding weights for each model based on validation performance.
        data_loader (DataLoader): DataLoader for the dataset to predict on.
        device (torch.device): Device to perform computations on.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Aggregated probabilities and true labels.
    """
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Ensembling Predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            batch_probs = np.zeros((input_ids.size(0), ensemble_models[0].config.num_labels))
            
            
            for model, weight in zip(ensemble_models, fold_weights):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # Extract logits
                probs = torch.sigmoid(logits).cpu().numpy()
                batch_probs += weight * probs
            all_probs.append(batch_probs)
            all_labels.append(labels)
    all_probs = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    return all_probs, all_labels




#############################
## Main Training Function  ##
#############################

def main():
    """
   Main function to execute the training pipeline, including data loading,
   model training with k-fold cross-validation, evaluation, and saving artifacts.
   """   
   
    try:
        ## Set seeds for reproducibility
        set_seed(CONFIG['seed'])
        
        logging.info(f"Using device: {CONFIG['device']}")
        
        ## Load and preprocess data
        logging.info("Loading labeled data...")
        labeled_df = pd.read_csv(CONFIG['labels_csv'])
        
        ## Convert 'parsed_topics' from string to list if necessary
        if labeled_df['parsed_topics'].dtype == object:
            labeled_df['parsed_topics'] = labeled_df['parsed_topics'].apply(ast.literal_eval)
        
        ## Normalize topics
        labeled_df['parsed_topics'] = labeled_df['parsed_topics'].apply(
            lambda topics: [normalize_topic(topic) for topic in topics]
        )
        
        
        ## Remove Classes with Fewer Than Two Samples
        all_topics = [topic for sublist in labeled_df['parsed_topics'] for topic in sublist]
        topic_counts = pd.Series(all_topics).value_counts()
        valid_topics = topic_counts[topic_counts >= 2].index.tolist()
        logging.info(f"Valid topics (>=2 samples): {valid_topics}")
        
        
        ## Filter out topics that don't meet the minimum sample requirement
        labeled_df['parsed_topics'] = labeled_df['parsed_topics'].apply(
            lambda topics: [topic for topic in topics if topic in valid_topics]
        )
        labeled_df = labeled_df[labeled_df['parsed_topics'].apply(len) > 0]
        logging.info(f"Number of samples after filtering: {len(labeled_df)}")
        
        
        ## Initialize Tokenizer
        tokenizer = RobertaTokenizer.from_pretrained(CONFIG['primary_model_name'])
        
        
        ## Initialize and Fit MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        mlb.fit(labeled_df['parsed_topics'])
        labeled_labels = mlb.transform(labeled_df['parsed_topics'])
        num_labels = len(mlb.classes_)
        logging.info(f"Number of labels: {num_labels}")
        
        
        ## Save MultiLabelBinarizer
        with open(CONFIG['mlb_save_path'], 'wb') as f:
            pickle.dump(mlb, f)
        logging.info(f"MultiLabelBinarizer saved at: {CONFIG['mlb_save_path']}")
        
        
        ## Split data into Train+Val and Test
        msss_initial = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=CONFIG['test_size'], random_state=CONFIG['seed'])
        for train_val_idx, test_idx in msss_initial.split(labeled_df['combined_text'].values, labeled_labels):
            train_val_texts, test_texts = labeled_df['combined_text'].values[train_val_idx], labeled_df['combined_text'].values[test_idx]
            train_val_labels, test_labels = labeled_labels[train_val_idx], labeled_labels[test_idx]
        
        logging.info(f"Training + Validation samples: {len(train_val_texts)}, Test samples: {len(test_texts)}")
        
        
        ## Initialize Multilabel Stratified K-Fold
        kfold = MultilabelStratifiedKFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['seed'])
        
        
        ## Lists to store ensemble models and their corresponding weights
        ensemble_models = []
        fold_weights = []
        
        ## Initialize criterion
        criterion = FocalLoss(gamma=2, alpha=0.25)
        
        ## Initialize a DataFrame to store per-fold thresholds
        all_fold_thresholds = []
        
        ## Iterate over each fold
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_texts, train_val_labels)):
            logging.info(f"\n=== Fold {fold + 1}/{CONFIG['k_folds']} ===")
            
            ## Split data
            fold_train_texts, fold_val_texts = train_val_texts[train_idx], train_val_texts[val_idx]
            fold_train_labels, fold_val_labels = train_val_labels[train_idx], train_val_labels[val_idx]
            
            ## Create Datasets and DataLoaders
            train_dataset = RedditCommentsDataset(fold_train_texts, fold_train_labels, tokenizer, CONFIG['max_len'])
            val_dataset = RedditCommentsDataset(fold_val_texts, fold_val_labels, tokenizer, CONFIG['max_len'])
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
            
            logging.info("Datasets and DataLoaders created for this fold.")
            
            ## Initialize a fresh model for each fold
            model = RobertaForSequenceClassification.from_pretrained(
                CONFIG['primary_model_name'],
                num_labels=num_labels,
                problem_type="multi_label_classification",
                output_attentions=False,
                output_hidden_states=False
            ).to(CONFIG['device'])
            
            logging.info("Model initialized for this fold.")
            
            
            ## Define optimizer parameters with different learning rates for different layers
            ############ Set Learning Rates Here #############
            
            optimizer_grouped_parameters = [
                {'params': model.roberta.embeddings.parameters(), 'lr': 1e-5},
                {'params': model.roberta.encoder.layer[:6].parameters(), 'lr': 1e-5},
                {'params': model.roberta.encoder.layer[6:].parameters(), 'lr': 2e-5},
                {'params': model.classifier.parameters(), 'lr': 3e-5},
            ]
            
            optimizer = AdamW(optimizer_grouped_parameters, correct_bias=False)
            total_steps = len(train_loader) * CONFIG['epochs']
            
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                ## 10% of steps for warm-up
                ###### Change Steps Here ######
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )
            logging.info("Optimizer and scheduler initialized for this fold.")
            
            ## Training variables for early stopping
            best_f1 = 0
            epochs_no_improve = 0
            best_thresholds = None
            
            ## Training Loop
            for epoch in range(CONFIG['epochs']):
                logging.info(f'\nEpoch {epoch + 1}/{CONFIG["epochs"]}')
                logging.info('-----------------------------------')
                
                ## Train the Model
                model.train()
                train_loss = 0
                for batch in tqdm(train_loader, desc="Training"):
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(CONFIG['device'])
                    attention_mask = batch['attention_mask'].to(CONFIG['device'])
                    labels = batch['labels'].to(CONFIG['device'])
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                logging.info(f"Train Loss: {train_loss:.4f}")
                
                ## Validate the Model
                metrics = eval_model(model, val_loader, CONFIG['device'], criterion, max_labels=3, metric_to_optimize='f1')
                
                ## Early Stopping and Model Saving
                if metrics['micro_f1'] > best_f1:
                    best_f1 = metrics['micro_f1']
                    epochs_no_improve = 0
                    best_thresholds = metrics['thresholds']
                    logging.info(f"New best model found at epoch {epoch + 1} with Micro F1: {best_f1:.4f}")
                    
                    ## Save the model for this fold
                    fold_model_save_path = os.path.join(CONFIG['model_save_dir'], f'fold_{fold + 1}_best_model')
                    model.save_pretrained(fold_model_save_path)
                    tokenizer.save_pretrained(fold_model_save_path)
                    logging.info(f"Model saved to: {fold_model_save_path}")
                    
                    ## Save optimal thresholds for this fold
                    fold_thresholds_save_path = os.path.join(CONFIG['thresholds_save_dir'], f'fold_{fold + 1}_optimal_thresholds.pkl')
                    with open(fold_thresholds_save_path, 'wb') as f:
                        pickle.dump(best_thresholds, f)
                    logging.info(f"Optimal thresholds for fold {fold + 1} saved to: {fold_thresholds_save_path}")
                else:
                    epochs_no_improve += 1
                    logging.info(f"No improvement in Micro F1. ({epochs_no_improve}/{CONFIG['early_stopping_patience']})")
                    if epochs_no_improve >= CONFIG['early_stopping_patience']:
                        logging.info("Early stopping triggered.")
                        break
                logging.info('-----------------------------------\n')
            
            logging.info(f"Training complete for Fold {fold + 1}.")
            
            ## Load the best model for this fold
            best_fold_model = RobertaForSequenceClassification.from_pretrained(fold_model_save_path)
            best_fold_model.to(CONFIG['device'])
            best_fold_model.eval()
            
            ## Collect validation predictions and calculate fold weight
            fold_val_probs = []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Collecting Validation Predictions for Fold {fold + 1}"):
                    input_ids = batch['input_ids'].to(CONFIG['device'])
                    attention_mask = batch['attention_mask'].to(CONFIG['device'])
                    
                    ## Access the logits
                    outputs = best_fold_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits  # Extract logits from the output object
                    probs = torch.sigmoid(logits).cpu().numpy()
                    fold_val_probs.append(probs)
            fold_val_probs = np.vstack(fold_val_probs)
            
            ## Determine fold weight based on validation micro F1
            fold_weight = best_f1
            fold_weights.append(fold_weight)
            logging.info(f"Fold {fold + 1} weight based on Micro F1: {fold_weight:.4f}")
            
            ## Add the model to the ensemble
            ensemble_models.append(best_fold_model)
            logging.info(f"Fold {fold + 1} model added to ensemble.")
            
            ## Store thresholds
            all_fold_thresholds.append(best_thresholds)
        
        logging.info("\nK-Fold Cross-Validation Training Complete.\n")
        
        ## Normalize fold weights
        total_f1 = sum(fold_weights)
        normalized_fold_weights = [f1 / total_f1 for f1 in fold_weights]
        logging.info(f"Normalized fold weights: {normalized_fold_weights}")
        
        ## Save fold weights
        fold_weights_save_path = os.path.join(CONFIG['thresholds_save_dir'], 'fold_weights.pkl')
        with open(fold_weights_save_path, 'wb') as f:
            pickle.dump(normalized_fold_weights, f)
        logging.info(f"Fold weights saved to: {fold_weights_save_path}")
        
        ## Create Test Dataset and DataLoader
        test_dataset = RedditCommentsDataset(test_texts, test_labels, tokenizer, CONFIG['max_len'])
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        
        ## Perform weighted ensemble prediction on test set
        ensemble_probs, ensemble_test_labels = ensemble_predict_weighted(
            ensemble_models, normalized_fold_weights, test_loader, CONFIG['device']
        )
        
        logging.info("Ensemble predictions on test set obtained.")
        
        ## Aggregate thresholds by averaging
        ensemble_thresholds = np.mean(all_fold_thresholds, axis=0).tolist()
        logging.info("\nOptimal thresholds for the ensemble determined by averaging fold thresholds:")
        for idx, threshold in enumerate(ensemble_thresholds):
            logging.info(f"Class {idx}: Threshold = {threshold:.2f}")
        
        ## Save ensemble thresholds
        ensemble_thresholds_save_path = os.path.join(CONFIG['thresholds_save_dir'], 'ensemble_optimal_thresholds.pkl')
        with open(ensemble_thresholds_save_path, 'wb') as f:
            pickle.dump(ensemble_thresholds, f)
        logging.info(f"\nEnsemble optimal thresholds saved to: {ensemble_thresholds_save_path}")
        
        ## Apply ensemble thresholds to test set predictions
        ensemble_binary_preds = apply_thresholds_with_limit_dynamic(ensemble_probs, ensemble_thresholds, max_labels=3)
        
        ## Calculate Metrics
        test_micro_f1 = f1_score(ensemble_test_labels, ensemble_binary_preds, average='micro', zero_division=0)
        test_macro_f1 = f1_score(ensemble_test_labels, ensemble_binary_preds, average='macro', zero_division=0)
        test_weighted_f1 = f1_score(ensemble_test_labels, ensemble_binary_preds, average='weighted', zero_division=0)
        test_hamming_loss = hamming_loss(ensemble_test_labels, ensemble_binary_preds)
        test_jaccard = jaccard_score(ensemble_test_labels, ensemble_binary_preds, average='samples', zero_division=0)
        
        logging.info("\n########## Ensemble Test Set Metrics ##########")
        logging.info(f"Micro F1: {test_micro_f1:.4f}")
        logging.info(f"Macro F1: {test_macro_f1:.4f}")
        logging.info(f"Weighted F1: {test_weighted_f1:.4f}")
        logging.info(f"Hamming Loss: {test_hamming_loss:.4f}")
        logging.info(f"Jaccard Index: {test_jaccard:.4f}")
        logging.info(f"Optimal Thresholds: {', '.join([f'{t:.2f}' for t in ensemble_thresholds])}")
        logging.info("###############################################\n")
        
        ## Save test metrics
        test_metrics = {
            'test_micro_f1': test_micro_f1,
            'test_macro_f1': test_macro_f1,
            'test_weighted_f1': test_weighted_f1,
            'test_hamming_loss': test_hamming_loss,
            'test_jaccard_index': test_jaccard,
            'optimal_thresholds': ensemble_thresholds
        }
        
        test_metrics_save_path = os.path.join(CONFIG['metrics_save_dir'], 'ensemble_test_metrics.pkl')
        with open(test_metrics_save_path, 'wb') as f:
            pickle.dump(test_metrics, f)
        logging.info(f"Ensemble test metrics saved to: {test_metrics_save_path}")
        
        ## TODO: implement a final training on the entire training set using ensemble strategies or advanced ensembling techniques here
        
    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise e

if __name__ == "__main__":
    main()



