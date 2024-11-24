# Social_Media_Final


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

## `reddit_posts.csv`

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

## Step 2: Run 
- **Script**: `redditCommentPull.py`  
- **Description**:
  - Collects all the comments from the top posts produced by redditPostPull.py
  - Rotates between multiple API key groups for rate-limited, asynchronous scraping.
- **Output**:
  - Saves comments to `reddit_comments.csv`
    
## `reddit_comments.csv`  

| **Column**       | **Description**                                             |
|-------------------|-------------------------------------------------------------|
| `post_id`        | Identifier of the post to which the comment belongs         |
| `state`          | Name of the subreddit (state)                               |
| `comment_id`     | Unique identifier of the comment                            |
| `body`           | Text content of the comment                                 |
| `created_utc`    | UTC timestamp of when the comment was created               |
| `score`          | Score of the comment                                        |
| `author`         | Username of the comment's author                            |



# Policy Area Categories

---

### 🔴 Health and Healthcare
**Description:**  
Topics related to health, healthcare services, public health initiatives, and medical research.

**Includes:**  
- Public health policies  
- Healthcare funding and access  
- Medical research and innovation  
- Health insurance regulations  

**Example:**  
*"Another public hospital closes in Montana, the third this year."*

---

### 🟠 Defense and National Security
**Description:**  
Covers armed forces, national defense, homeland security, and military policies.

**Includes:**  
- Military funding and procurement  
- Defense strategies and policies  
- Homeland security measures  
- Veteran affairs  

**Example:**  
*"I’m worried that China may come and steal my goats in the night, is that possible? Do they like goats?"*

---

### 🔵 Crime and Law Enforcement
**Description:**  
Includes crime prevention, law enforcement, policing, and emergency management.

**Includes:**  
- Police funding and reform  
- Criminal justice policies  
- Emergency response protocols  
- Crime statistics and prevention programs  

**Example:**  
*"Third officer arrested in New York this week on corruption charges."*

---

### 🌍 International Affairs and Trade
**Description:**  
Focuses on international relations, foreign trade, diplomacy, and international finance.

**Includes:**  
- Trade agreements and tariffs  
- Diplomatic relations  
- International aid and development  
- Global economic policies  

**Example:**  
*"Vermont tightens border regulations with Canada, will maple syrup prices go up?"*

---

### 🟢 Government Operations and Politics
**Description:**  
Topics on government operations, legislation, law, political processes, and congressional matters.

**Includes:**  
- Legislative procedures  
- Government budgeting and spending  
- Political reforms  
- Electoral processes  

**Example:**  
*"State congress motions for unlimited snack budget."*

---

### 🟠 Economy and Finance
**Description:**  
Encompasses topics related to financial stability, economic growth, labor policies, and trade practices that impact citizens’ day-to-day lives and the overall economy.

**Includes:**  
- Taxation and fiscal policy  
- Economic growth and development initiatives  
- Commerce and trade regulations  
- Employment and labor policies  
- Financial markets and regulations  
- Inflation and interest rate policies  

**Example:**  
*"If our property taxes go up again this year, I’m moving to the moon. I mean it this time, Elon is really making progress on the moon."*

---

### 🌱 Environment and Natural Resources
**Description:**  
Covers environmental protection, natural resources, energy, and water resource management.

**Includes:**  
- Renewable energy initiatives  
- Conservation efforts  
- Water resource management  
- Climate change policies  

**Example:**  
*"Historic flood washes away brand new solar panel installations."*

---

### 📚 Education and Social Services
**Description:**  
Covers education, social welfare, housing, family support, and social sciences.

**Includes:**  
- Public education funding  
- Social welfare programs  
- Housing policies  
- Family support services  

**Example:**  
*"Affordable housing is impossible to find right now in our state!"*

---

### 🌾 Agriculture and Food
**Description:**  
Includes agriculture, farming policies, food production, and food safety.

**Includes:**  
- Agricultural subsidies  
- Food safety regulations  
- Sustainable farming practices  
- Rural development  

**Example:**  
*"Organic farming takes a big hit this year, due to the wow-crop-delicious insect boom."*

---

### 🔬 Science, Technology, and Communications
**Description:**  
Topics on scientific research, technological advancements, and communication systems.

**Includes:**  
- Research and development funding  
- Technology infrastructure  
- Telecommunications regulations  
- Innovation policies  

**Example:**  
*"Comcast sues small family-owned telephone maker in Florida."*

---

### 🛂 Immigration and Civil Rights
**Description:**  
Focuses on immigration policies, civil rights, minority issues, and Native American matters.

**Includes:**  
- Immigration reform  
- Civil liberties protections  
- Minority rights  
- Native American affairs  

**Example:**  
*"This is crazy, my son can’t even get a job at Fast Food Express due to the recent influx of Swedish Meatball Farmers from Portugal."*

---

### 🚧 Transportation and Infrastructure
**Description:**  
Covers transportation systems, public works, and infrastructure development.

**Includes:**  
- Public transportation funding  
- Infrastructure projects  
- Transportation safety regulations  
- Urban planning  

**Example:**  
*"I swear to god if they don’t fix these potholes I’m going to write another strongly written letter."*

---

### 🎭 Culture and Recreation
**Description:**  
Includes arts, culture, religion, sports, recreational activities, and animal-related topics.

**Includes:**  
- Arts funding and grants  
- Cultural heritage preservation  
- Recreational facilities  
- Animal welfare policies  

**Example:**  
*"I love these moose. I’m so glad we can own 5 now legally."*

---

### ❓ Other / Uncategorized
**Description:**  
Use this label if the content does not fit into any specific category or is uncategorized.

**Includes:**  
- Miscellaneous topics not covered by other categories  
- Ambiguous or unclear content  

**Example:**  
*"The post discusses personal opinions on various unrelated topics without a clear topic focus."*
