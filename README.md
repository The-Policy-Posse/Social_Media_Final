# Social Media Classification Modeling for Policy Topics

Welcome to the **Social Media Classification Modeling for Policy Topics** repository! This project focuses on collecting, processing, and classifying Reddit data to analyze discussions around various policy topics across different U.S. states. By leveraging advanced data scraping techniques, exploratory data analysis (EDA), manual labeling, and sophisticated classification models, this project aims to provide insightful classifications of social media conversations related to policy issues.

## Table of Contents

1. [Introduction](#introduction)
2. [Reddit Data Collection](#reddit-data-collection)
   - [Files in the Folder](#files-in-the-folder)
   - [Requirements](#requirements)
   - [Step 1: Run `redditPostPull.py`](#step-1-run-redditpostpullpy)
   - [Step 2: Run `redditCommentPull.py`](#step-2-run-redditcommentpullpy)
3. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   - [Overview](#overview)
   - [EDA Allotaxonometry](#eda-allotaxonometry)
   - [Topic Modeling](#topic-modeling)
   - [Statistical Tests](#statistical-tests)
4. [Image Handling and Sampling for Label Studio](#image-handling-and-sampling-for-label-studio)
   - [Image Download Script](#image-download-script)
   - [Data Sampling for Label Studio](#data-sampling-for-label-studio)
   - [Sampling Methodology](#sampling-methodology)
   - [How to Run the Sampling Process](#how-to-run-the-sampling-process)
5. [Labeling Process with Label Studio](#labeling-process-with-label-studio)
   - [Overview of the Labeling Process](#overview-of-the-labeling-process)
   - [Task Types](#task-types)
   - [Label Studio Setup](#label-studio-setup)
   - [Quality Assurance](#quality-assurance)
   - [Outputs](#outputs)
6. [Classification Model Preprocessing](#classification-model-preprocessing)
   - [Overview of Preprocessing Pipeline](#overview-of-preprocessing-pipeline)
   - [Key Features](#key-features)
   - [Script Breakdown](#script-breakdown)
   - [Outputs](#outputs-1)
7. [Classification Model](#classification-model)
   - [Overview](#overview-1)
   - [Data Preprocessing](#data-preprocessing)
   - [Model Architecture](#model-architecture)
   - [Training Procedure](#training-procedure)
   - [Evaluation Metrics](#evaluation-metrics)
   - [Results](#results)
   - [Outputs from Fine-Tuning RoBERTa](#outputs-from-fine-tuning-roberta)
   - [Prediction Script](#prediction-script)
8. [Results](#results-1)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)
12. [Acknowledgements](#acknowledgements)

## Introduction

In the age of digital information, social media platforms like Reddit serve as rich sources of public opinion and discourse on a myriad of topics, including policy issues. Understanding and categorizing these discussions can provide valuable insights for policymakers, researchers, and the general public.

This project aims to systematically collect and analyze Reddit posts and comments from state-specific subreddits across the United States to classify discussions into predefined policy topics. The workflow encompasses several stages:

1. **Data Collection**: Utilizing custom scripts to efficiently gather large-scale Reddit data while managing API rate limits.
2. **Exploratory Data Analysis (EDA)**: Conducting in-depth analysis to assess the suitability of the data for modeling, including trend detection and topic exploration.
3. **Image Handling and Sampling**: Preprocessing images associated with Reddit posts and creating a representative sample for manual labeling using Label Studio.
4. **Labeling Process**: Implementing a structured labeling workflow to categorize posts accurately, ensuring high-quality annotations through team collaboration and quality assurance measures.
5. **Classification Model Preprocessing**: Preparing both labeled and unlabeled datasets for training machine learning models, involving text cleaning, normalization, and label encoding.
6. **Model Development**: Building and fine-tuning multi-label classification models based on transformer architectures (RoBERTa) to accurately categorize posts into multiple policy areas.
7. **Evaluation and Results**: Assessing model performance using robust metrics and presenting comprehensive results to validate the effectiveness of the classification approach.

By integrating these components, the project not only facilitates the classification of social media content but also provides a scalable framework for analyzing policy-related discussions across various platforms. Whether you're a data scientist, policy analyst, or researcher, this repository offers valuable tools and insights to explore the intersection of social media and policy discourse.

---

#### License

This project is licensed under the MIT License

#### Contact
For any questions or feedback, please reach out to dforcade@gatech.edu).

#### Acknowledgements
- Special thanks to all of our annotators!


### Key Objectives:
- **Data Collection**: Automate the large-scale collection of Reddit data, including posts and comments.
- **Exploratory Data Analysis (EDA)**: Apply advanced techniques to validate and prepare the dataset for classification.
- **Model Development**: Create and fine-tune multi-label classification models to categorize posts into predefined policy areas.
- **Human-in-the-Loop**: Incorporate manual labeling and review processes to ensure high-quality datasets and nuanced model outputs.




## Reddit Data Collection

The `Reddit_Data_Scrapers` folder contains scripts designed for efficient and large-scale collection of Reddit posts and comments from state-specific subreddits. These scripts utilize multiple Reddit API keys to manage rate limits and optimize asynchronous data fetching.  They are configured to pull the top 600 threads/posts from the past year for the 50 state subreddits, and all of the comments (including nested comments) from those threads (around 4.5 million comments).  The comments script takes around 16 hours to run, posts is much faster.

---

### Files in the Folder
1. **`redditPostPull.py`**  
   - **Purpose**: Retrieves the top posts from specified state subreddits over the past year.  
   - **Output**: Saves collected posts to a CSV file (`reddit_posts.csv`).

2. **`redditCommentPull.py`**  
   - **Purpose**: Fetches all comments for posts collected by `redditPostPull.py`.  
   - **Output**: Saves comments to a CSV file (`reddit_comments.csv`), grouped by state.
   - (New York had an issue in data collection and has two specific scripts to append to the created dataframes)
---

### Requirements
#### Reddit API Credentials
1. **Create Reddit Accounts**: 
   - Sign up for multiple Reddit accounts to obtain multiple API keys.

2. **Register Applications**: 
   - Log in to each Reddit account and navigate to Reddit Apps.
   - Click "Create App" or "Create Another App".
   - Fill in the application name and select "script" as the type.
   - Set the redirect URI to `http://localhost`.
   - Note down the client ID and client secret (API key).

3. **Organize API Keys**:
   - Create a JSON file named `reddit_api_keys.json` in the `Reddit_Data_Scrapers` folder.
   - Structure the JSON file as follows:
     ```json
     {
       "group1": [
         {
           "client_id": "your_client_id_1",
           "api_key": "your_api_secret_1"
         },
         {
           "client_id": "your_client_id_2",
           "api_key": "your_api_secret_2"
         }
       ],
       "group2": [
         {
           "client_id": "your_client_id_3",
           "api_key": "your_api_secret_3"
         },
         {
           "client_id": "your_client_id_4",
           "api_key": "your_api_secret_4"
         }
       ]
     }
4.  **Environment**:
   - Python 3.8+.
   - Install dependencies via:
     ```bash
     pip install -r requirements.txt
     ```


#### Step 1: Run redditPostPull.py must be run first, as redditCommentsPull.py utilizies the post_ids created
- **Script**: `redditPostPull.py`  
- **Description**:
  - Collects the top posts from state subreddits over the last year.
  - Rotates between multiple API key groups for rate-limited, asynchronous scraping.
- **Output**:
  - Saves posts to `reddit_posts.csv`

### `reddit_posts.csv`

| **Column**       | **Description**                                             |
|-------------------|-------------------------------------------------------------|
| `post_id`        | Unique identifier of the Reddit post                        |
| `state`          | Name of the subreddit (state)                               |
| `title`          | Title of the post                                           |
| `selftext`       | Body text of the post                                       |
| `created_utc`    | UTC timestamp of when the post was created                  |
| `score`          | Score (upvotes - downvotes) of the post                     |
| `url`            | URL of the post                                             |
| `num_comments`   | Number of comments on the post                              |
| `author`         | Username of the post's author                               |

---

### Step 2: Run 
- **Script**: `redditCommentPull.py`  
- **Description**:
  - Collects all the comments from the top posts produced by redditPostPull.py
  - Rotates between multiple API key groups for rate-limited, asynchronous scraping.
- **Output**:
  - Saves comments to `reddit_comments.csv`
    
### `reddit_comments.csv`  

| **Column**       | **Description**                                             |
|-------------------|-------------------------------------------------------------|
| `post_id`        | Identifier of the post to which the comment belongs         |
| `state`          | Name of the subreddit (state)                               |
| `comment_id`     | Unique identifier of the comment                            |
| `body`           | Text content of the comment                                 |
| `created_utc`    | UTC timestamp of when the comment was created               |
| `score`          | Score of the comment                                        |
| `author`         | Username of the comment's author                            |


## EDA

### Quick Overview on **Why** we decided on on investing in manual labeling and advanced Classification Models  

Once the data successfully scraped and validated, extensive EDA was run using several exploratory methods to determine if this data would be a good candidate for modeling.  

The first step was employing Allotaxonometry-Style graphs on several test states to determine if rough trends and differences could be detected in the data, or if it was simply too noisy to be worth the trouble.  With our EDA Allotaxonemtry, we were able to detect a Marijunana Legalization Trend downtick due to a legislative event that was losing steam, and in Vermont we were able to detect Foliage-related terms trending going into the Fall:

<p>
  <img src="images/kentucky-5-01-23.png" alt="EDA Allotaxonometry of Kentucky" style="width: 45%;">
  <img src="images/vermont_7-30.png" alt="EDA Allotaxonometry of Vermont" style="width: 45%;">
</p>


From there, we used exploratory Topic Modeling with BeRTopic and KMeans clustering.  When converting embeddings to t-SNE, we saw some promising results -- but not directly usable for our policy classification/modeling task.  K-Means was strugging to differentiate in a meaningful way - and when BeRTopic clusters were individually investigated, they were too fragemented for usable downstream analysis.

<p>
  <img src="images/tsneKmeans.png" alt="EDA Allotaxonometry of Kentucky" style="width: 45%;">
  <img src="images/tsneBertTopic.png" alt="EDA Allotaxonometry of Vermont" style="width: 45%;">
</p>  


We also ran additional statistical tests on simple sentiment analysis between clusters and groups to determine if there was validitity to our intuiton, and the results were statistically significant.  With these (and a few more metrics/analysis), we made the decision that this data was a good candidate for manual labeling and transformer based classification for our goal of identifying political topic discussion.

Full EDA Modeling report can be found here: [EDA Modeling Report](pdfs/Sentiment_Report_1.pdf)  





## Image Handling and Sampling for Label Studio
To prepare to for manual labeling of the social media data, several preprocessing steps need to be completed, namely image handling and sampling.  In our dataset, 35% of reddit posts 
contained images, many of which were crucial for determining context.  To addresss this, the script `reddit_post_image_handling.py` uses reddit_posts.csv urls to search for image 
extensions and retrieves images from Reddit post URLs and saves them locally.  Warning: This will easily be over 20GB of data, and the script will take several hours to run.  We moved these to the cloud for hosting, but if you're working local that's fine too.

### Image Download Script

`reddit_post_image_handling.py`

**Key Features:**
- **Normalization of URLs**: The script cleans and normalizes the URLs to address inconsistencies such as backslashes or whitespace.
- **Filtering Valid Image Links**: Only URLs pointing to supported image formats (`.jpg`, `.png`, `.gif`, etc.) are retained.
- **Concurrent Downloading**: Uses `asyncio` and `aiohttp` to download multiple images simultaneously, significantly reducing runtime.
- **Retry Logic with Exponential Backoff**: Handles rate limits and transient errors by retrying failed downloads with increasing delays.

**Output**:  
Images are saved in the directory `post_images/`, with filenames corresponding to their respective `post_id` (e.g., `abc123.jpg`).

---

### Data Sampling for Label Studio
Once the images have been downloaded, you can proceed to using `labelStudioSampleCreation.rmd`, which handles Label Studio preprocessing as well as generates a stratified,
proportional, and constrained sample with prioritization based on engagement metrics within each State.  It also performs lemmatization on a dummy column to facilitate a keyword search for each policy topic of interest, to further drive representative sampling for all classes into the manual phase.  


#### Sampling Methodology

1. **Policy Area Classification**  
   Each post is classified into one of several predefined policy areas using a keyword-based matching approach.  
   - **Text Preprocessing**: Titles and body text are lemmatized for better keyword matching using the `textstem` R library.  
   - **Keyword Matching**: Policy areas are defined by a set of keywords, such as:
     - *Health*: `health`, `medicine`, `hospital`, `insurance`, etc.
     - *Environment*: `climate`, `pollution`, `wildlife`, etc.  
     Posts without a match are classified as *Other / Uncategorized*.

2. **State-Specific Sampling**  
   The sampling ensures a balanced representation across U.S. states while prioritizing relevance:
   - **Minimum and Maximum Constraints**: Each state contributes at least 90 posts but no more than 350.
   - **Weighting by Engagement**: Sampling is limited-weight proportional to the total comments per state.
   - **Stratification by Policy Area**: Posts are distributed across policy areas to maintain diversity in content.

3. **Post Selection Criteria**  
   Posts are prioritized based on engagement:
   - **80th Percentile Thresholds**: Posts in the top 20% for each State by `num_comments` or `score` are prioritized for selection.
   - **Random Sampling for Remaining Posts**: To fill gaps, additional posts are randomly sampled within states, excluding duplicates.

4. **"Other / Uncategorized" Posts**  
   An additional 1,000 posts classified as "Other / Uncategorized" are included in the final dataset to ensure representation of general or miscellaneous topics.

#### Final Dataset Characteristics

- **Total Sample Size**: 6,000 posts, plus 1,000 *Other / Uncategorized* posts.
- **Balanced Distribution**: Ensures proportional representation of states and policy areas while maintaining diversity.
- **Output File**: The final dataset is saved as `final_sample.csv`.

---

#### Visualization and Quality Assurance

To verify the dataset's representativeness:
- **State Distribution**: The number of posts per state is visualized in a bar chart.
- **Policy Area Distribution**: Policy areas are similarly analyzed to confirm proportional representation.
- **Comparison to Original Data**: Distributions of the final sample are compared to the original dataset to highlight differences and ensure sampling goals are met.

---

**Visualization Examples**:  
Graphs comparing the distribution of states and policy areas in the sampled dataset are included to validate the sampling process.  

<p align="center">
<img src="images/sample_distribution_state.png" alt="Weighted Sample Distribution for State Activity" style="width: 50%;">  
</p>  


---

### How to Run the Sampling Process

1. **Prepare the Input Data**  
   Ensure `reddit_posts.csv` and `reddit_comments.csv` are available in the working directory.

2. **Run Image Download**  
   Use `download_images.py` to fetch images linked in posts. Save them in `post_images/`.

3. **Execute the Sampling Script**  
   Run `labelStudioSampleCreation.Rmd` to generate the balanced ready to label dataset
   - Output: `final_sample.csv`
  
#### `final_sample.csv`

| **Column**       | **Description**                                             |
|-------------------|-------------------------------------------------------------|
| `post_id`        | Unique identifier of the Reddit post                        |
| `state`          | Name of the subreddit (state)                               |
| `title`          | Title of the post                                           |
| `selftext`       | Body text of the post                                       |
| `policy_area`    | Classified policy area of the post                          |
| `num_comments`   | Number of comments on the post                              |
| `score`          | Score (upvotes - downvotes) of the post                     |
| `image_url`      | URL of the image associated with the post                   |




## Labeling Process with Label Studio

To prepare the dataset for analysis, we used **[Label Studio](https://labelstud.io/)** for labeling Reddit posts. This process involved both **single-label** and **multi-label classification** tasks.  We set ours up on a virtual machine on Google Cloud, and uploaded the reddit images downloaded from the previous script to a bucket on the Cloud.  Those images were then efficently fed into the Label Studio setup so that our annotators could have full context while labeling.  


Ours can be seen here: http://34.23.190.214:8080/projects/

<p align="center">
  <img src="images/label_studio_example.jpg" alt="Our Label Studio UI" style="width: 40%;">
</p>



### Overview of the Labeling Process

1. **Team Collaboration**:
   - We recruited additional annotators and provided training to ensure consistent and high-quality annotations.
   - We provided a setup guide, training, and reference guide for all labelers (Including members of our team), which were displayed everytime someone entered a project
   - Reference guide provided in-depth category definitions and explainations to keep labeling between annotators consistent.
        - [Starting Guide PDF](pdfs/Labeling_Getting_Started.pdf)
        - [Reference Guide PDF](pdfs/LAbeling_Reference_Guide.pdf)
   - Label Studio Info: [Label Studio Documentation](https://labelstud.io/guide/)
   
2. **Task Types**:
   - **Multi-Label Classification**: 2,500 posts were labeled with one or more categories, allowing for posts to belong to multiple policy areas or topics.
   - **Single-Label Classification**: 1,000 posts were labeled with exactly one category, simplifying the classification process.

---

### Label Studio Setup

1. **Data Preparation**:
   - The sampled dataset (`final_sample.csv`) was uploaded to Label Studio.
   - Each record included:
     - **State**: State subreddit the post was made in
     - **Post Title**: Title of the Post
     - **Image URL**: (if applicable): Visual content automatically displayed with posts from Google Cloud bucket
     - **Post Contents**: Text the author posted along with the title, if any


3. **Label Studio Interface**:
   - Each labeler was assigned tasks directly in Label Studio.
   - The interface included:
      - Large color annotation buttons for each category, to assist in speed and comfortability of annotators
      - Automated queue on Submit or Skip
      - Full random delivery to keep things interesting for annotators and ensure distribution of class balance

4. **Quality Assurance**:
   - An initial training phase allowed labelers to familiarize themselves with the task.
   - Randomly selected posts were reviewed to ensure labeling consistency.
   - Individual annotator results for Cohen's Kappa statistics

---

### Outputs

- **Labeled Dataset**:
  - After completion, the labeled data was exported from Label Studio in .CSV format.
  - The final dataset was processed into CSV format for further analysis.

- **Label Summary**:
  - A summary of labeled categories, including frequency and distribution, was generated for exploratory analysis.
  - Save this file as Classification_Model/data/raw/labeled_reddit_posts_raw.csv
  - Bring in an additional copy of reddit_posts.csv and place into Classification_Model/data/raw/reddit_posts_raw.csv

---



## Classification Model Preprocessing

The **classification model preprocessing pipeline** is implemented in Python to prepare Reddit post and comment data for classification tasks. This includes handling both labeled and unlabeled datasets, ensuring they are cleaned, normalized, and formatted for multi-label and single-label classification.  Only Posts will be covered here due to the large file size and computing requirements for handling the 4.5 million comments.

---

### Overview of Preprocessing Pipeline

The preprocessing process is orchestrated by the script `Classification_Model/main_preprocessing.py`, which utilizes helper modules to automate and modularize tasks.

#### Key Features:
1. **Labeled Data Preprocessing**:
   - Parses multi-label topics from raw data fields from Label Studio Export.
   - Cleans and combines text fields (title + body text).
   - Converts multi-label topics into a **binary label matrix** for machine learning.
   - Assigns a **primary label** to each post for single-label classification using a prioritization strategy (Note: No Single-Label models were moved to production).
   - Outputs a processed dataset for labeled posts (`Classification_Model/data/processed/labeled_reddit_posts_processed.csv`).

2. **Unlabeled Data Preprocessing**:
   - Filters out posts already labeled to avoid duplication.
   - Cleans and combines text fields (title + body text).
   - Initializes placeholders for topic and label fields.
   - Outputs a processed dataset for unlabeled posts (`Classification_Model/data/processed/unlabeled_reddit_posts_processed.csv`).

3. **Topic Distribution Analysis**:
   - Computes and prints the distribution of primary labels.
   - Counts the occurrences of all topics across the dataset (multi-label).
   - Outputs summary statistics to a CSV (`Classification_Model/data/processed/total_topic_occurrences.csv`).

---

### Script Breakdown

#### **`main_preprocessing.py`**  
The entry point for the preprocessing pipeline:
- Calls labeled and unlabeled data preprocessing functions.
- Outputs processed datasets and summary statistics.

#### **`utils.py`**  
Utility functions used throughout the pipeline:
- **`clean_text()`**: Cleans text by removing URLs, special characters, and unnecessary spaces, and converts it to lowercase.
- **`assign_single_label()`**: Assigns a single primary label to a post, prioritizing non-governmental labels if multiple are present.
- **`normalize_topics()`**: Ensures consistent formatting for topic labels.

#### **`preprocess_labeled.py`**  
Handles preprocessing for the labeled dataset:
- Parses and normalizes multi-label topics.
- Converts topics to a binary matrix using `MultiLabelBinarizer`.
- Cleans text fields and combines title and body.
- Assigns a primary label to each post for single-label classification.

#### **`preprocess_unlabeled.py`**  
Handles preprocessing for the unlabeled dataset:
- Filters out posts already present in the labeled dataset.
- Cleans text fields and combines title and body.
- Initializes empty labels for future annotation.

---

### Outputs

1. **Processed Labeled Dataset (`labeled_reddit_posts_processed.csv`)**:
   - Includes cleaned text fields, binary topic labels, and primary labels.

2. **Processed Unlabeled Dataset (`unlabeled_reddit_posts_processed.csv`)**:
   - Includes cleaned text fields and placeholders for future labeling.

3. **Summary Statistics**:
   - **Total Topic Occurrences (`total_topic_occurrences.csv`)**:
     - Provides the count of each topic across all posts.
   - **Primary Label Distribution**:
     - Displays the count of posts per primary label (printed to console).

---

### Topics and Labels

---

### How to Run the Preprocessing Pipeline

1. **Prepare the Raw Data**:
   - Place the raw labeled and unlabeled datasets in the `Classification_Model/data/raw/` directory:
     - `labeled_reddit_posts_raw.csv`
     - `reddit_posts_raw.csv`

2. **Run the Main Preprocessing Script**:
   ```bash
   python main_preprocessing.py
3. **Outputs**:
  - Both processed .csvs are put into the Classification_Model/data/processed/ directory:
    - `labeled_reddit_posts_processed.csv`
    - `unlabeled_reddit_posts_processed.csv`



## Classification Model

### Overview

We developed two multi-label text classification models to categorize Reddit posts into multiple policy areas -- an ensemble roberta-base and an ensemble fusion roberta-large.  The ensemble fusion model is currently still in development and not ready for release at this time.  The nonFusion model is performing well on the multi-label policy classification task and is provided here.  

To run the model: 
- Make sure you have `labeled_reddit_posts_processed.csv` in the `Classification_Model/data/processed/` directory.
- Go to `Classification_Model/nonFusion`
- Install `requirements.txt`
- Run `nonFusion.py`


### Data Preprocessing

#### Label Binarization

- **MultiLabelBinarizer**: We utilized scikit-learn's `MultiLabelBinarizer` to convert the list of policy area labels for each post into a binary matrix. This matrix is suitable for multi-label classification tasks, where each post can belong to multiple classes.

#### Text Preprocessing

- **Text Cleaning**: Combined the post's title and body text into a single string. We performed cleaning steps such as:
  - Removing URLs
  - Removing special characters
- **Tokenization**: Used `RobertaTokenizer` with a maximum sequence length of 128 tokens to tokenize the cleaned text.

### Model Architecture

We fine-tuned the pre-trained `roberta-base` model from Hugging Face's Transformers library for our classification task.

- **Model Modification**:
  - Set `problem_type="multi_label_classification"` to adapt the model for multi-label outputs.
  - Adjusted the output layer to match the number of policy area classes.

### Loss Function: Focal Loss

Due to class imbalance in our dataset, we implemented the **Focal Loss** function to focus training on hard-to-classify examples.

The Focal Loss is defined as:

$$
\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- `p_t` is the model's estimated probability for the true class.
- `alpha_t` is the weighting factor for the class (we set `alpha = 0.25`).
- `gamma` is the focusing parameter (we set  `gamma = 2`).

We customized the Focal Loss to handle the class imbalance in our dataset. The implementation in PyTorch is as follows:

```{python}
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, labels):
        probs = torch.sigmoid(logits)                                     ## Convert logits to probabilities
        ce_loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)  ## Binary cross-entropy loss
        pt = torch.where(labels == 1, probs, 1 - probs)                   ## Probability of the true class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss        ## Apply focal loss formula
        return focal_loss.mean()                                          ## Return mean loss over the batch
```


This loss function reduces the relative loss for well-classified examples and focuses on those that the model struggles with.

### Training Procedure

#### Cross-Validation

- **5-Fold Multilabel Stratified Cross-Validation**: Ensured that each fold has a similar distribution of labels, crucial for multi-label datasets.
- **Early Stopping**: Monitored the validation Micro F1 score to prevent overfitting. Training stops if the performance doesn't improve for a specified number of epochs (patience of 3 epochs).

#### Optimization

- **Optimizer**: Used `AdamW` with weight decay for regularization.
- **Layer-wise Learning Rates**: Applied different learning rates to different layers:
  - Embeddings and lower layers: `1e-5`
  - Middle layers: `1e-5`
  - Higher layers: `2e-5`
  - Classification head: `3e-5`
- **Learning Rate Scheduler**: Employed a linear learning rate scheduler with a warm-up phase (10% of total steps).

### Threshold Optimization

In multi-label classification, it's essential to determine the optimal threshold to convert predicted probabilities into binary labels.

#### Finding Optimal Thresholds

For each class `c`, we determined the threshold `τ_c` that maximizes the chosen metric (e.g., F1 or Micro F1 score) on the validation set.

The process involved:

1. For each class, iterate over possible thresholds in the range [0.1, 0.9] with steps of 0.01.
2. For each threshold, compute the metric (e.g., F1 score) using the validation data.
3. Select the threshold that yields the highest metric value.

```{python}
def apply_thresholds_with_limit_dynamic(y_probs, thresholds, max_labels=3):

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
```

#### Applying Thresholds with Label Limit

To ensure that each instance is assigned a realistic number of labels, we applied the thresholds while limiting the maximum number of labels per instance to `k` (we used `k`=3).

The procedure is:

1. **For each instance**:
   - Obtain the predicted probabilities for all classes.
   - Sort the classes by their predicted probabilities in descending order.
2. **Assign labels**:
   - Iterate through the sorted classes.
   - Assign a label if the predicted probability exceeds the class threshold `c`.
   - Stop assigning labels once `k` labels have been assigned.

This method ensures that the most confident predictions (above the threshold) are selected, up to the maximum number of labels per instance.

### Evaluation Metrics

We evaluated our model using several metrics suitable for multi-label classification:

- **Micro F1 Score**: Measures the F1 score globally by counting the total true positives, false negatives, and false positives.
- **Macro F1 Score**: Averages the F1 score per class without considering class imbalance.
- **Weighted F1 Score**: Averages the F1 score per class, weighted by the number of true instances per class.
- **Hamming Loss**: Fraction of labels that are incorrectly predicted.
- **Jaccard Index**: Also known as Intersection over Union, measures the similarity between the predicted and true label sets.
- **Per-Class Precision, Recall, and F1 Scores**: Provides insight into the model's performance on individual classes.

### Results

#### Cross-Validation Performance

- The best model per fold was saved based on the highest validation Micro F1 score.
- Optimal thresholds per class were saved for each fold.
- The fold models were added to an ensemble for final evaluation.

#### Test Set Evaluation

- **Ensemble Model**: Combined the models from each fold using weighted averaging based on their validation performance.
- **Threshold Aggregation**: The optimal thresholds from each fold were averaged to obtain ensemble thresholds.
- **Test Metrics**:

## Ensemble Test Set Metrics

| Metric            | Score  |
|--------------------|--------|
| Micro F1          | 0.7359 |
| Macro F1          | 0.7042 |
| Weighted F1       | 0.7359 |
| Hamming Loss      | 0.0637 |
| Jaccard Index     | 0.6711 |


### Outputs from Fine-Tuning RoBERTa for Multi-Label Classification

This section details the various outputs generated by the model:

---

### **Saved Models**
- **Location**: `./artifacts/models/`
- **Content**: 
  - Best-performing model for each fold during k-fold cross-validation.
  - Saved using Hugging Face’s `save_pretrained` method, including both the model weights and tokenizer.
3
---

### **Optimal Thresholds**
- **Location**: `./artifacts/thresholds/`
- **Content**: 
  - **Per-Fold Thresholds**:
    - Files: `fold_1_optimal_thresholds.pkl`, `fold_2_optimal_thresholds.pkl`, etc.
    - Description: Optimal thresholds for each class obtained during cross-validation.
  - **Ensemble Thresholds**:
    - File: `ensemble_optimal_thresholds.pkl`
    - Description: Average thresholds across all folds, used for final test evaluation.

---

### **Metrics**
- **Location**: `./artifacts/metrics/`
- **Content**:
  - **Per-Fold Metrics**:
    - Includes metrics like Micro F1, Macro F1, Weighted F1, Hamming Loss, and Jaccard Index for each fold.
  - **Test Set Metrics**:
    - File: `ensemble_test_metrics.pkl`
    - Description: Metrics calculated on the test set using the ensemble of models.

---

### **MultiLabelBinarizer**
- **Location**: `./artifacts/mlb_multi_label.pkl`
- **Content**:
  - A `MultiLabelBinarizer` object fitted on the training dataset.
  - Used for encoding and decoding multi-label targets.

---

### **Log Files**
- **Location**: `training.log`
- **Content**:
  - Detailed logs of the training process, including:
    - Data preprocessing steps.
    - Training and validation metrics for each epoch and fold.
    - Final test set evaluation metrics.

---

### **Ensemble Predictions**
- **Content**:
  - Aggregated probabilities and binary predictions from the ensemble models.
  - Predictions stored in memory for evaluation metrics and threshold application.




## **Prediction Script for Multi-Label Classification Using RoBERTa Ensemble**

This script uses the ensemble of fine-tuned RoBERTa models to perform multi-label classification on unlabeled Reddit posts and comments. It assigns labels based on confidence thresholds, assignes confidence score for Human-In-The-Loop review, selects high-confidence samples, and visualizes the results.

---

#### **Key Features**
- **Ensemble Predictions**: Combines predictions from multiple models using weighted averaging.
- **Label Assignment**: Assigns multi-label classifications using class-specific thresholds while limiting the number of labels per instance.
- **Confidence Scoring**: Computes overall confidence scores for predictions using methods such as average or maximum probability.
- **High-Confidence Sample Selection**: Selects the top percentage of samples based on confidence scores for further analysis or manual labeling.
- **t-SNE Visualizations**: Reduces prediction probabilities to two dimensions for visualization, including both topic-specific and combined plots.
- **Contrastive Word Clouds**: Generates word clouds for each predicted topic, highlighting distinctive terms using contrastive TF-IDF scores.

---

#### **Outputs**
1. **CSV Files**:
   - Predictions with assigned labels and confidence scores.
   - High-confidence samples.
   - Data with t-SNE coordinates.
2. **Visualizations**:
   - Topic-specific and combined t-SNE plots.
   - Contrastive word clouds for each topic.
3. **Log File**:
   - Detailed logs of the prediction workflow.

---

#### **Requirements:**
- Pre-trained ensemble models, thresholds, and `MultiLabelBinarizer` saved during training.
- Unlabeled data in `/data/processed/unlabeled_reddit_posts_processed` (From the Classification Preprocessing Step)
 **Usage:**
- Configure the paths and settings in the `CONFIG` dictionary within the script.
-  Run the script:
   `python nonFusion_Predict.py`



# Results


<p align="center">
<img src="images/tsne_combined_all_topics.png" alt="Weighted Sample Distribution for State Activity" style="width: 50%;">  
</p>  







https://policy-ensemble-social-posts-classification.streamlit.app/










## Policy Area Categories

---

### 🔴 Health and Healthcare 🔴
**Description:**  
Topics related to health, healthcare services, public health initiatives, and medical research.

**Example:**  
*"Another public hospital closes in Montana, the third this year."*

---

### 🟠 Defense and National Security 🟠
**Description:**  
Covers armed forces, national defense, homeland security, and military policies.


**Example:**  
*"I’m worried that China may come and steal my goats in the night, is that possible? Do they like goats?"*

---

### 🔵 Crime and Law Enforcement 🔵
**Description:**  
Includes crime prevention, law enforcement, policing, and emergency management.
 

**Example:**  
*"Third officer arrested in New York this week on corruption charges."*

---

### 🌍 International Affairs and Trade 🌍
**Description:**  
Focuses on international relations, foreign trade, diplomacy, and international finance.


**Example:**  
*"Vermont tightens border regulations with Canada, will maple syrup prices go up?"*

---

### 🟢 Government Operations and Politics 🟢
**Description:**  
Topics on government operations, legislation, law, political processes, and congressional matters.


**Example:**  
*"State congress motions for unlimited snack budget."*

---

### 🟠 Economy and Finance 🟠
**Description:**  
Encompasses topics related to financial stability, economic growth, labor policies, and trade practices that impact citizens’ day-to-day lives and the overall economy.


**Example:**  
*"If our property taxes go up again this year, I’m moving to the moon. I mean it this time, Elon is really making progress on the moon."*

---

### 🌱 Environment and Natural Resources 🌱
**Description:**  
Covers environmental protection, natural resources, energy, and water resource management.


**Example:**  
*"Historic flood washes away brand new solar panel installations."*

---

### 📚 Education and Social Services 📚
**Description:**  
Covers education, social welfare, housing, family support, and social sciences.


**Example:**  
*"Affordable housing is impossible to find right now in our state!"*

---

### 🌾 Agriculture and Food 🌾
**Description:**  
Includes agriculture, farming policies, food production, and food safety.


**Example:**  
*"Organic farming takes a big hit this year, due to the wow-crop-delicious insect boom."*

---

### 🔬 Science, Technology, and Communications 🔬
**Description:**  
Topics on scientific research, technological advancements, and communication systems.


**Example:**  
*"Comcast sues small family-owned telephone maker in Florida."*

---

### 🛂 Immigration and Civil Rights 🛂
**Description:**  
Focuses on immigration policies, civil rights, minority issues, and Native American matters.


**Example:**  
*"This is crazy, my son can’t even get a job at Fast Food Express due to the recent influx of Swedish Meatball Farmers from Portugal."*

---

### 🚧 Transportation and Infrastructure 🚧
**Description:**  
Covers transportation systems, public works, and infrastructure development.
 

**Example:**  
*"I swear to god if they don’t fix these potholes I’m going to write another strongly written letter."*

---

### 🎭 Culture and Recreation 🎭
**Description:**  
Includes arts, culture, religion, sports, recreational activities, and animal-related topics.

**Example:**  
*"I love these moose. I’m so glad we can own 5 now legally."*

---

### ❓ Other / Uncategorized ❓
**Description:**  
Use this label if the content does not fit into any specific category or is uncategorized.

**Example:**  
*"The post discusses personal opinions on various unrelated topics without a clear topic focus."*
