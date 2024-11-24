# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:59:07 2024

@author: dforc
"""


##############
## Imports ##
##############

import os
import re
import pickle
import logging
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification

from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS




##############################
## Configuration and Setup ##
##############################

## Configuration Dictionary
CONFIG = {
    'seed': 47,
    'device': torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
    'max_len': 128,
    'batch_size': 128,
    'num_folds': 5,
    'top_percentage': 0.10,  # 10%
    'max_labels': 3,
    'artifacts_dir': './artifacts/',
    'models_dir': './artifacts/models',
    'thresholds_dir': './artifacts/thresholds',
    'metrics_dir': './artifacts/metrics',
    'mlb_path': './artifacts/mlb_multi_label.pkl',
    'ensemble_thresholds_path': './artifacts/thresholds/ensemble_optimal_thresholds.pkl',
    'fold_weights_path': './artifacts/thresholds/fold_weights.pkl',
    'unlabeled_csv': '../data/processed/unlabeled_reddit_posts_processed.csv',
    'output_csv': '../data/model_predictions/nonFusion_predictions.csv',
    'top_10_percent_csv': './artifacts/top_10_percent_predictions.csv',
    'tsne_csv': 'labeled_with_tsne.csv',
    'wordcloud_dir': './artifacts/wordclouds',
    'tsne_plots_dir': './artifacts/tsne_plots',
}

## Create necessary directories
for dir_path in [
    CONFIG['models_dir'],
    CONFIG['thresholds_dir'],
    CONFIG['metrics_dir'],
    CONFIG['artifacts_dir'],
    CONFIG['wordcloud_dir'],
    CONFIG['tsne_plots_dir']
]:
    os.makedirs(dir_path, exist_ok=True)

## Setup Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

## Prevent adding multiple handlers
if not logger.handlers:
    
    ## File Handler
    file_handler = logging.FileHandler('prediction.log', mode='a')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    ## Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

logger.info("Prediction Script Initialized.")



##############################
##    Datast Definition     ##
##############################


class RedditCommentsDataset(Dataset):
    """
    PyTorch Dataset for Reddit Comments.

    Attributes:
        texts (list): List of text samples.
        tokenizer (RobertaTokenizer): Tokenizer for encoding the texts.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,  # RoBERTa does not use token type ids
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }



##############################
##     Helper Functions     ##
##############################

def sanitize_filename(name):
    """
    Sanitize filenames by removing or replacing invalid characters.

    Args:
        name (str): Original filename.

    Returns:
        str: Sanitized filename.
    """
    return re.sub(r'[\\/:"*?<>|]+', '_', name)



def compute_overall_confidence(probabilities, method='average'):
    """
    Compute an overall confidence score for each sample.

    Args:
        probabilities (np.ndarray): Array of shape (num_samples, num_labels) with predicted probabilities.
        method (str): Method to compute confidence ('average' or 'max').

    Returns:
        np.ndarray: Array of confidence scores.
    """
    
    
    if method == 'average':
        return probabilities.mean(axis=1)
    elif method == 'max':
        return probabilities.max(axis=1)
    else:
        raise ValueError("Invalid method. Choose 'average' or 'max'.")
        
        
        

def select_top_percentage(df, confidence_scores, top_percentage=0.10):
    """
    Select the top percentage of samples based on confidence scores.

    Args:
        df (pd.DataFrame): The original DataFrame containing the data.
        confidence_scores (np.ndarray): Array of confidence scores.
        top_percentage (float): The top percentage to select (e.g., 0.10 for top 10%).

    Returns:
        pd.DataFrame: DataFrame containing the top percentage samples.
    """
    if not 0 < top_percentage < 1:
        raise ValueError("top_percentage must be between 0 and 1.")

    threshold = np.percentile(confidence_scores, 100 * (1 - top_percentage))
    top_indices = np.where(confidence_scores >= threshold)[0]
    top_df = df.iloc[top_indices].copy()
    top_df['confidence_score'] = confidence_scores[top_indices]
    return top_df




def load_ensemble_models(models_dir, num_folds, device):
    """
    Load ensemble models from the specified directory.

    Args:
        models_dir (str): Directory where models are saved.
        num_folds (int): Number of folds/models to load.
        device (torch.device): Device to map models to.

    Returns:
        list: List of loaded models.
    """
    ensemble_models = []


    for fold in range(1, num_folds + 1):
        model_path = os.path.join(models_dir, f'fold_{fold}_best_model')
        if not os.path.exists(model_path):
            logging.warning(f"Model path {model_path} does not exist. Skipping.")
            continue
        try:
            model = RobertaForSequenceClassification.from_pretrained(
                model_path,
                # Remove num_labels and problem_type to prevent overriding the config
                output_attentions=False,
                output_hidden_states=False
            )
            model.to(device)
            model.eval()
            ensemble_models.append(model)
            logging.info(f"Successfully loaded model from: {model_path}")
        except Exception as e:
            logging.error(f"Failed to load model from {model_path}: {e}")

    return ensemble_models



def ensemble_predict_weighted(ensemble_models, fold_weights, data_loader, device):
    """
    Aggregates predictions from ensemble models using weighted averaging.

    Args:
        ensemble_models (list): List of loaded models.
        fold_weights (list): List of weights corresponding to each model.
        data_loader (DataLoader): DataLoader for the input data.
        device (torch.device): Device to perform computations on.

    Returns:
        np.ndarray: Aggregated probabilities.
    """
    all_probs = []

    for batch in tqdm(data_loader, desc="Ensembling Predictions"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_probs = np.zeros((input_ids.size(0), ensemble_models[0].config.num_labels))

        for model, weight in zip(ensemble_models, fold_weights):
            with torch.no_grad():                       ## Ensure no gradients are being tracked
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits                 ## Extract logits
                probs = torch.sigmoid(logits).cpu().detach().numpy()  ## Detach before converting
                batch_probs += weight * probs

        all_probs.append(batch_probs)

    all_probs = np.vstack(all_probs)
    return all_probs



def apply_thresholds_with_limit_dynamic(y_probs, thresholds, max_labels=3):
    """
    Apply class-specific thresholds and limit the number of predicted labels to `max_labels` per instance.

    Args:
        y_probs (np.ndarray): Array of shape (num_samples, num_labels) with predicted probabilities.
        thresholds (list or np.ndarray): List of thresholds for each class.
        max_labels (int): Maximum number of labels to predict per instance.

    Returns:
        np.ndarray: Binary array of predictions with shape (num_samples, num_labels).
    """
    binary_preds = np.zeros_like(y_probs)
    for i in range(y_probs.shape[0]):
        probs = y_probs[i]
        sorted_indices = np.argsort(probs)[::-1]
        selected_indices = []
        for idx in sorted_indices:
            if probs[idx] >= thresholds[idx]:
                selected_indices.append(idx)
            if len(selected_indices) == max_labels:
                break
        binary_preds[i, selected_indices] = 1
    return binary_preds



def eval_model(model, data_loader, device, mlb, thresholds, max_labels=3):
    """
    Assigns labels based on thresholds.

    Args:
        model (RobertaForSequenceClassification): The model to evaluate.
        data_loader (DataLoader): DataLoader for the data.
        device (torch.device): Device to run computations on.
        mlb (MultiLabelBinarizer): MultiLabelBinarizer instance.
        thresholds (list): Optimal thresholds for each class.
        max_labels (int): Maximum number of labels per instance.

    Returns:
        tuple: (assigned_labels, all_probs)
    """
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    
    
    ## Apply thresholds and limit to top `max_labels`
    all_preds = apply_thresholds_with_limit_dynamic(all_probs, thresholds, max_labels=max_labels)
    
    
    ## Inverse transform to get label names
    assigned_labels = mlb.inverse_transform(all_preds)
    return assigned_labels, all_probs



def compute_contrastive_tfidf(topic_texts, reference_texts, stopwords_list):
    """
    Compute contrastive TF-IDF scores for word cloud generation.

    Args:
        topic_texts (list): Texts assigned to a specific topic.
        reference_texts (list): Texts not assigned to the specific topic.
        stopwords_list (list): List of stopwords.

    Returns:
        dict: Contrastive TF-IDF scores.
    """
    
    
    ## Vectorize topic-specific corpus
    vectorizer_topic = TfidfVectorizer(
        max_features=1000,
        stop_words=stopwords_list,
        lowercase=True,
        ngram_range=(1, 2)
    )
    tfidf_topic_matrix = vectorizer_topic.fit_transform(topic_texts)
    topic_feature_names = vectorizer_topic.get_feature_names_out()
    topic_scores = np.mean(tfidf_topic_matrix.toarray(), axis=0)

    ## Vectorize reference corpus
    vectorizer_ref = TfidfVectorizer(
        max_features=1000,
        stop_words=stopwords_list,
        lowercase=True,
        ngram_range=(1, 2),
        vocabulary=topic_feature_names  # Use the same vocabulary for comparability
    )
    tfidf_ref_matrix = vectorizer_ref.fit_transform(reference_texts)
    ref_scores = np.mean(tfidf_ref_matrix.toarray(), axis=0)

    ## Compute contrastive scores (difference between topic and reference TF-IDF)
    contrastive_scores = topic_scores - ref_scores
    return dict(zip(topic_feature_names, contrastive_scores))




###############################
##   Visualization Functions ##
###############################


def plot_tsne(df, mlb, tsne_plots_dir):
    """
    Generate t-SNE plots for each topic and a combined plot.

    Args:
        df (pd.DataFrame): DataFrame containing t-SNE coordinates and assigned topics.
        mlb (MultiLabelBinarizer): MultiLabelBinarizer instance.
        tsne_plots_dir (str): Directory to save t-SNE plots.
    """
    logging.info("Generating t-SNE plots...")

    ## Define a colormap for topics
    num_topics = len(mlb.classes_)
    palette = sns.color_palette("husl", num_topics)  # Unique colors for each topic
    topic_colors = {topic: palette[idx] for idx, topic in enumerate(mlb.classes_)}

    ## Plot individual t-SNEs for each topic
    for idx, topic in enumerate(mlb.classes_):
        mask = df['assigned_topics'].apply(lambda x: topic in x if isinstance(x, tuple) else False)
        if not mask.any():
            logging.info(f"No samples for topic '{topic}'. Skipping t-SNE plot.")
            continue

        plt.figure(figsize=(8, 6))
        plt.scatter(
            df.loc[mask, 'tsne_x'],
            df.loc[mask, 'tsne_y'],
            c=[topic_colors[topic]] * mask.sum(),
            label=topic,
            alpha=0.6,
            s=10
        )
        plt.title(f"t-SNE Visualization for Topic: {topic}")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend()
        plt.tight_layout()

        # Save the plot
        plot_filename = sanitize_filename(f"tsne_topic_{topic}.png")
        plot_path = os.path.join(tsne_plots_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
        logging.info(f"Saved t-SNE plot for topic '{topic}' to '{plot_path}'.")

    ## Combined t-SNE Plot with All Topics
    plt.figure(figsize=(10, 8))
    for idx, topic in enumerate(mlb.classes_):
        mask = df['assigned_topics'].apply(lambda x: topic in x if isinstance(x, tuple) else False)
        if not mask.any():
            continue
        plt.scatter(
            df.loc[mask, 'tsne_x'],
            df.loc[mask, 'tsne_y'],
            c=[topic_colors[topic]] * mask.sum(),
            label=topic,
            alpha=0.6,
            s=10
        )
    plt.title("Combined t-SNE Visualization with All Topics")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(loc='best', markerscale=2, fontsize='small')  # Legend for topics
    plt.tight_layout()

    ## Save the combined plot
    combined_plot_path = os.path.join(tsne_plots_dir, "tsne_combined_all_topics.png")
    plt.savefig(combined_plot_path)
    plt.close()
    logging.info(f"Saved combined t-SNE plot with all topics to '{combined_plot_path}'.")



def create_wordclouds(df, mlb, wordcloud_dir):
    """
    Generate contrastive word clouds for each topic.

    Args:
        df (pd.DataFrame): DataFrame containing assigned topics and texts.
        mlb (MultiLabelBinarizer): MultiLabelBinarizer instance.
        wordcloud_dir (str): Directory to save word clouds.
    """
    logging.info("Creating word clouds for each topic...")
    stopwords_list = list(STOPWORDS)

    for idx, topic in enumerate(mlb.classes_):
        
        ## Filter texts assigned to the current topic
        mask = df['assigned_topics'].apply(lambda x: topic in x if isinstance(x, tuple) else False)
        topic_texts = df.loc[mask, 'combined_text'].tolist()
        reference_texts = df.loc[~mask, 'combined_text'].tolist()

        if not topic_texts or not reference_texts:
            logging.info(f"Skipping topic '{topic}' due to insufficient data for word cloud.")
            continue

        ## Compute contrastive TF-IDF scores
        contrastive_scores = compute_contrastive_tfidf(topic_texts, reference_texts, stopwords_list)

        ## Generate Word Cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=stopwords_list,
            max_words=100
        ).generate_from_frequencies(contrastive_scores)

        ## Plot the word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Contrastive Word Cloud for Topic: {topic}")
        plt.tight_layout()

        ## Save the word cloud
        wc_filename = sanitize_filename(f"wordcloud_topic_{topic}.png")
        wc_path = os.path.join(wordcloud_dir, wc_filename)
        plt.savefig(wc_path)
        plt.close()
        logging.info(f"Saved word cloud for topic '{topic}' to '{wc_path}'.")




###############################
##        Main Func          ##
###############################

def main():
    """
    Main function to execute the prediction workflow.
    """
    try:
        # -----------------------------------------------------------------------------------
        ## Load MultiLabelBinarizer
        # -----------------------------------------------------------------------------------
        with open(CONFIG['mlb_path'], 'rb') as f:
            mlb = pickle.load(f)
        logging.info(f"Loaded MultiLabelBinarizer from: {CONFIG['mlb_path']}")

        # -----------------------------------------------------------------------------------
        ## Load Ensemble Thresholds and Fold Weights
        # -----------------------------------------------------------------------------------
        with open(CONFIG['ensemble_thresholds_path'], 'rb') as f:
            ensemble_thresholds = pickle.load(f)
        logging.info(f"Loaded ensemble optimal thresholds from: {CONFIG['ensemble_thresholds_path']}")

        with open(CONFIG['fold_weights_path'], 'rb') as f:
            fold_weights = pickle.load(f)
        logging.info(f"Loaded fold weights from: {CONFIG['fold_weights_path']}")

        # -----------------------------------------------------------------------------------
        ## Load Tokenizer and Ensemble Models
        # -----------------------------------------------------------------------------------
        
        ## Load the tokenizer directly from 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        logging.info("Loaded tokenizer from 'roberta-base'.")

        ## Load ensemble models
        ensemble_models = load_ensemble_models(CONFIG['models_dir'], CONFIG['num_folds'], CONFIG['device'])
        if not ensemble_models:
            logging.error("No ensemble models loaded. Exiting.")
            return
        logging.info(f"Loaded {len(ensemble_models)} ensemble models.")

        # -----------------------------------------------------------------------------------
        ## Load Unlabeled Data
        # -----------------------------------------------------------------------------------
        unlabeled_df = pd.read_csv(CONFIG['unlabeled_csv'])
        logging.info(f"Loaded unlabeled data from: {CONFIG['unlabeled_csv']}")

        ## Check for 'combined_text' column
        if 'combined_text' not in unlabeled_df.columns:
            logging.error("The CSV file must contain a 'combined_text' column.")
            return

        ## Handle missing 'combined_text' entries
        missing_text = unlabeled_df['combined_text'].isnull().sum()
        logging.info(f"Missing 'combined_text' entries: {missing_text}")

        if missing_text > 0:
            unlabeled_df.dropna(subset=['combined_text'], inplace=True)
            logging.info(f"Dropped {missing_text} samples with missing 'combined_text'.")

        ## Extract texts
        unlabeled_texts = unlabeled_df['combined_text'].values

        # -----------------------------------------------------------------------------------
        ## Create Dataset and DataLoader
        # -----------------------------------------------------------------------------------
        unlabeled_dataset = RedditCommentsDataset(
            texts=unlabeled_texts,
            tokenizer=tokenizer,
            max_len=CONFIG['max_len']
        )

        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=False
        )
        logging.info("Created Dataset and DataLoader for unlabeled data.")

        # -----------------------------------------------------------------------------------
        ## Perform Ensemble Predictions
        # -----------------------------------------------------------------------------------
        logging.info("Starting ensemble predictions...")
        ensemble_probs = ensemble_predict_weighted(
            ensemble_models=ensemble_models,
            fold_weights=fold_weights,
            data_loader=unlabeled_loader,
            device=CONFIG['device']
        )
        logging.info("Ensemble predictions completed.")

        # -----------------------------------------------------------------------------------
        ## Assign Labels Based on Ensemble Thresholds with Limiting to Top N
        # -----------------------------------------------------------------------------------
        binary_preds = apply_thresholds_with_limit_dynamic(
            y_probs=ensemble_probs,
            thresholds=ensemble_thresholds,
            max_labels=CONFIG['max_labels']
        )
        assigned_labels = mlb.inverse_transform(binary_preds)
        unlabeled_df['assigned_topics'] = assigned_labels
        logging.info("Assigned labels to unlabeled data based on thresholds.")

        # -----------------------------------------------------------------------------------
        ## Compute Overall Confidence Scores
        # -----------------------------------------------------------------------------------
        unlabeled_df['confidence_score'] = compute_overall_confidence(
            probabilities=ensemble_probs,
            method='average'  # Options: 'average', 'max'
        )
        logging.info("Computed overall confidence scores using 'average' method.")

        # -----------------------------------------------------------------------------------
        ## Select Top Percentage High-Confidence Samples
        # -----------------------------------------------------------------------------------
        top_percentage_df = select_top_percentage(
            df=unlabeled_df,
            confidence_scores=unlabeled_df['confidence_score'].values,
            top_percentage=CONFIG['top_percentage']
        )
        logging.info(f"Selected top {int(CONFIG['top_percentage'] * 100)}% high-confidence samples: {len(top_percentage_df)} entries.")

        # -----------------------------------------------------------------------------------
        ## Save Top Percentage Samples to CSV
        # -----------------------------------------------------------------------------------
        top_percentage_csv = CONFIG['top_10_percent_csv']
        top_percentage_df.to_csv(top_percentage_csv, index=False)
        logging.info(f"Top {int(CONFIG['top_percentage'] * 100)}% high-confidence predictions saved to '{top_percentage_csv}'.")

        # -----------------------------------------------------------------------------------
        ## Save All Predictions to CSV
        # -----------------------------------------------------------------------------------
        unlabeled_df.to_csv(CONFIG['output_csv'], index=False)
        logging.info(f"All predictions saved to '{CONFIG['output_csv']}'.")

        # -----------------------------------------------------------------------------------
        ## Apply t-SNE on Ensemble Probabilities
        # -----------------------------------------------------------------------------------
        logging.info("Applying t-SNE on ensemble probabilities...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_results = tsne.fit_transform(ensemble_probs)
        unlabeled_df['tsne_x'] = tsne_results[:, 0]
        unlabeled_df['tsne_y'] = tsne_results[:, 1]
        tsne_csv = CONFIG['tsne_csv']
        unlabeled_df.to_csv(tsne_csv, index=False)
        logging.info(f"t-SNE coordinates saved to '{tsne_csv}'.")

        # -----------------------------------------------------------------------------------
        ## Plot t-SNE Visualizations
        # -----------------------------------------------------------------------------------
        plot_tsne(unlabeled_df, mlb, CONFIG['tsne_plots_dir'])

        # -----------------------------------------------------------------------------------
        ## Create Word Clouds for Each Topic
        # -----------------------------------------------------------------------------------
        create_wordclouds(unlabeled_df, mlb, CONFIG['wordcloud_dir'])

        logging.info("Prediction workflow completed successfully.")

    except Exception as e:
        logging.exception("An error occurred during the prediction workflow.")

if __name__ == "__main__":
    main()
