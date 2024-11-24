# Social_Media_Final


## Reddit Data Collection

The `Reddit_Data_Scrapers` folder contains scripts designed for efficient and large-scale collection of Reddit posts and comments from state-specific subreddits. These scripts utilize multiple Reddit API keys to manage rate limits and optimize asynchronous data fetching.

---

### Files in the Folder
1. **`redditPostPull.py`**  
   - **Purpose**: Retrieves the top posts from specified state subreddits over the past year.  
   - **Output**: Saves collected posts to a CSV file (`reddit_posts.csv`).

2. **`redditCommentPull.py`**  
   - **Purpose**: Fetches all comments for posts collected by `redditPostPull.py`.  
   - **Output**: Saves comments to a CSV file (`reddit_comments.csv`), grouped by state.

---

### Requirements
Before running the scripts, ensure the following are set up:

1. **Environment**:
   - Python 3.8+.
   - Install dependencies via:
     ```bash
     pip install -r requirements.txt
     ```

2. **Configuration Files**:
   - **`.env` File**:
     Create a file named `reddit_env.env` with the following content:
     ```plaintext
     PolicyPosseReddit_UserAgent=YOUR_USER_AGENT
     ```
   - **`reddit_api_keys.json`**:
     Structure your API keys as follows:
     ```json
     {
       "group1": [
         {"client_id": "YOUR_CLIENT_ID", "api_key": "YOUR_API_KEY"},
         {"client_id": "YOUR_CLIENT_ID", "api_key": "YOUR_API_KEY"}
       ],
       "group2": [
         {"client_id": "YOUR_CLIENT_ID", "api_key": "YOUR_API_KEY"}
       ]
     }
     ```

---

### Usage Instructions

#### Step 1: Pulling Reddit Posts
- **Script**: `redditPostPull.py`  
- **Description**:
  - Collects the top posts from state subreddits over the last year.
  - Rotates between multiple API keys for rate-limited, asynchronous scraping.
- **Output**:
  - Saves posts to `reddit_posts.csv` with details like `post_id`, `title`, `selftext`, `created_utc`, `score`, and more.
- **Execution**:
  Run the script:
  ```bash
  python redditPostPull.py
