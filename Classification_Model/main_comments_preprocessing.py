# -*- coding: utf-8 -*-
"""
This script processes a dataset of Reddit comments, removes invalid or empty entries, 
cleans the text by removing URLs and special characters, and saves the cleaned data 
to a CSV file for further analysis.


@author: dforc
"""

import pandas as pd
import re


#####################
## Load the dataset
#####################
file_path = "data/raw/reddit_comments_raw.csv"
output_path = "data/processed/reddit_comments_processed.csv"
df = pd.read_csv(file_path)



#####################
## Handle missing and invalid entries
#####################

## Convert na to ""
df['body'] = df['body'].fillna("").astype(str)

## Remove [removed] and [deleted] comments
removed_entries_df = df[df['body'].isin(['[removed]', '[deleted]'])]

## Save Empty values for printout
empty_entries_df = df[df['body'] == ""]

## Create final Dataframe with removed, deleted, and empty values taken out
cleaned_entries_df = df[~df['body'].isin(['[removed]', '[deleted]', ""])].copy()



#####################
## Preprocess valid entries
#####################

def preprocess_text(text):
    text = re.sub(r"http\S+", "", text)         ## Remove URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  ## Remove special characters
    text = text.strip()                         ## Remove leading and trailing whitespace
    return text

cleaned_entries_df.loc[:, 'body'] = cleaned_entries_df['body'].apply(preprocess_text)
cleaned_entries_df = cleaned_entries_df[cleaned_entries_df['body'] != ""]



#####################
## Count and save results
#####################

removed_count = len(removed_entries_df)
empty_count = len(empty_entries_df)
final_cleaned_count = len(cleaned_entries_df)

print("####### Summary #######")
print(f"Removed entries count: {removed_count}")
print(f"Empty entries count: {empty_count}")
print(f"Cleaned entries count (non-empty): {final_cleaned_count}")
print("########################")

cleaned_entries_df.to_csv(output_path, index=False)
print(f"Final cleaned DataFrame saved to {output_path}")

