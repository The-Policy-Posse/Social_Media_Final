# scripts/preprocess_unlabeled.py

# -*- coding: utf-8 -*-
"""
Script for preprocessing unlabeled Reddit posts.

- Removes posts already present in the labeled dataset.
- Cleans text data (titles and selftext).
- Prepares the dataset for future labeling.

@author: dforc
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scripts.utils import clean_text, normalize_topics, assign_single_label

##################################
# Preprocess Unlabeled Data      #
##################################

def preprocess_unlabeled(file_path, output_path, labeled_post_ids, topics):
    """
    Preprocesses the unlabeled dataset by filtering out labeled posts and cleaning text data.
    """
    print("Loading unlabeled dataset...")
    df = pd.read_csv(file_path)

    print("Filtering out labeled posts...")
    df = df[~df['post_id'].isin(labeled_post_ids)].reset_index(drop=True)
    
    
    ##################################
    #       Clean Text Fields        #
    ##################################

    print("Cleaning text fields...")
    df['cleaned_title'] = df['title'].apply(clean_text)
    df['cleaned_selftext'] = df['selftext'].apply(clean_text)
    df['combined_text'] = df['cleaned_title'] + " " + df['cleaned_selftext']
    
    
    ##################################
    #    Initialize Empty Labels     #
    ##################################

    print("Initializing empty labels...")
    df['parsed_topics'] = [[] for _ in range(len(df))]
    mlb = MultiLabelBinarizer(classes=topics)
    empty_labels = mlb.fit_transform(df['parsed_topics'])
    empty_label_df = pd.DataFrame(empty_labels, columns=topics)
    df = pd.concat([df, empty_label_df], axis=1)
    df['primary_label'] = ""
    
    
    ##################################
    #    Save Processed Data         #
    ##################################

    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)

    return df
