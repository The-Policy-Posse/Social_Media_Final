# scripts/preprocess_labeled.py

# -*- coding: utf-8 -*-
"""
Script for preprocessing labeled Reddit posts.

- Parses topic labels.
- Cleans and preprocesses the text fields.
- Converts topics to binary multilabel format.
- Assigns a primary topic label to each post.

@author: dforc
"""

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scripts.utils import clean_text, assign_single_label
import ast
from tabulate import tabulate

##################################
# Parse and Normalize Topics     #
##################################

def parse_topics(topic_entry):
    """
    Safely parse topics from JSON-like strings or handle plain strings.
    """
    if pd.isnull(topic_entry):  # Handle NaN entries
        return []
    try:
        # Attempt to parse as a dictionary if it's a JSON-like string
        parsed = ast.literal_eval(topic_entry)
        if isinstance(parsed, dict) and "choices" in parsed:
            return parsed["choices"]  # Return the list of choices
        return [topic_entry.strip()] if isinstance(topic_entry, str) else []
    except (ValueError, SyntaxError):
        # Fall back to treating the entry as a plain string
        return [topic_entry.strip()] if isinstance(topic_entry, str) else []

    ##################################
    #          Normalize             #
    ##################################

def normalize_topics(topics):
    """
    Ensures consistent labeling for topics.
    Combines 'Defense and National Security' and 'International Affairs and Trade' into 
    'National Security and International Affairs'.
    Also fixes variations like 'Other / Uncategorized' and 'Other \/ Uncategorized'.
    """
    normalized = []
    for topic in topics:
        if topic in ["Other / Uncategorized", "Other \\/ Uncategorized"]:
            normalized_topic = "Other / Uncategorized"
        elif topic in ["Defense and National Security", "International Affairs and Trade"]:
            normalized_topic = "National Security and International Affairs"
        else:
            normalized_topic = topic
        if normalized_topic not in normalized:
            normalized.append(normalized_topic)
    return normalized


##################################
# Preprocess Labeled Data        #
##################################

def preprocess_labeled(file_path, output_path, topics):
    """
    Preprocesses the labeled dataset and saves the processed output.
    """
    
    ## Load Data
    print("Loading labeled dataset...")
    df = pd.read_csv(file_path)
    
    ## Parse and Normalize
    print("Parsing and normalizing topics...")
    df['parsed_topics'] = df['topic'].apply(parse_topics).apply(normalize_topics)
    
    ## Cleaning Title and Selftext, Creating combined_text
    print("Cleaning text fields...")
    df['cleaned_title'] = df['title'].apply(clean_text)
    df['cleaned_selftext'] = df['selftext'].apply(clean_text)
    df['combined_text'] = df['cleaned_title'] + " " + df['cleaned_selftext']


    ##################################
    #      Convert to Binary Labels  #
    ##################################

    print("Converting topics to binary labels...")
    mlb = MultiLabelBinarizer(classes=topics)
    binary_labels = mlb.fit_transform(df['parsed_topics'])
    binary_label_df = pd.DataFrame(binary_labels, columns=topics)
    df = pd.concat([df, binary_label_df], axis=1)


    ##################################
    #     Assign Primary Labels      #
    ##################################
    print("Assigning primary labels...")
    df['primary_label'] = [
        assign_single_label(row, topics) for row in binary_labels
    ]

    print(f"Saving processed data to {output_path}...")
    df.to_csv(output_path, index=False)

    return df


##################################
#    Topic Distribution          #
##################################

## Primary Printout
def print_primary_label_distribution(df):
    """
    Prints the distribution of primary labels using tabulate for formatting.
    """
    primary_label_counts = df['primary_label'].value_counts().reset_index()
    primary_label_counts.columns = ['Label', 'Count']
    print("\n*** Primary Label Distribution ***\n")
    print(tabulate(primary_label_counts, headers='keys', tablefmt='grid'))



## Multi-Label Prinout
def print_total_topic_occurrences(df):
    """
    Prints the total occurrences of each topic using tabulate for formatting.
    """
    all_topics = [topic for sublist in df['parsed_topics'] for topic in sublist]
    topic_counts = pd.Series(all_topics).value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    print("\n*** Total Topic Occurrences Across All Posts ***\n")
    print(tabulate(topic_counts, headers='keys', tablefmt='grid'))
    
## Save Mutil-Label Topic Distribution
def save_total_topic_occurrences(df, output_path):
    """
    Saves the total topic occurrences to a CSV file.
    """
    all_topics = [topic for sublist in df['parsed_topics'] for topic in sublist]
    topic_counts = pd.Series(all_topics).value_counts().reset_index()
    topic_counts.columns = ['Topic', 'Count']
    topic_counts.to_csv(output_path, index=False)
    print(f"\nTotal topic occurrences saved to {output_path}\n")
        
