## scripts/utils.py

# -*- coding: utf-8 -*-
"""
Utility functions for text cleaning, topic normalization, and label assignment.

- Includes shared functionality used across preprocessing scripts.
- Centralizes common logic for better maintainability and reusability.

@author: dforc
"""

import re
import pandas as pd

##################################
#          Clean Text            #
##################################

def clean_text(text):
    """
    Cleans text data by removing URLs, special characters, and extra spaces.
    Converts text to lowercase for consistency.

    Args:
        text (str): The text string to clean.

    Returns:
        str: Cleaned text.
    """
    if pd.isnull(text):                         ## Handle missing text data
        return ""
    text = re.sub(r"http\S+", "", text)         ## Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  ## Remove special characters
    text = text.lower()                         ## Convert to lowercase
    return text.strip()

##################################
#      Normalize Topics          #
##################################

def normalize_topics(topics):
    """
    Ensures consistent labeling for topics.

    Fixes variations like:
    - 'Other / Uncategorized'
    - 'Other \/ Uncategorized'

    Args:
        topics (list): List of topic labels.

    Returns:
        list: Normalized topic labels.
    """
    return [
        "Other / Uncategorized" if topic in ["Other / Uncategorized", "Other \\/ Uncategorized"] else topic
        for topic in topics
    ]

##################################
#     Assign Primary Label       #
##################################

def assign_single_label(row, topics, gov_label="Government Operations and Politics"):
    """
    Assigns a single primary topic label to a row based on binary topic indicators.

    Prioritizes non-government labels if multiple labels are present.
    Defaults to "Uncategorized" if no labels are assigned.

    Args:
        row (list): Binary list indicating topic assignments.
        topics (list): List of topic names.
        gov_label (str): Label for "Government Operations and Politics".

    Returns:
        str: Assigned primary label.
    """
    
    ## Get indices of assigned labels
    label_indices = [i for i, val in enumerate(row) if val == 1]

    ## Handle rows with no assigned labels
    if not label_indices:
        return "Uncategorized"

    ## Get topic names for assigned labels
    labels = [topics[i] for i in label_indices]

    ## Prioritize non-government labels if present
    if gov_label in labels:
        for label in labels:
            if label != gov_label:
                return label
        return gov_label

    ## Default to the first label
    return labels[0]