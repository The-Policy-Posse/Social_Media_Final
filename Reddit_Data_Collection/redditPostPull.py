# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 00:27:13 2024

@author: dforc

Description:
    This script asynchronously retrieves top posts from state-specific subreddits 
    over the past year using multiple Reddit API keys for rate limiting. It processes 
    data in batches, rotating through API keys to adhere to request limits, and saves 
    the final output as a CSV file. Key features include initializing multiple Reddit 
    clients, fetching posts asynchronously, and managing API rate limits.
"""

import asyncio
import asyncpraw
import pandas as pd
import json
import aiohttp
from tqdm.asyncio import tqdm
import os
from dotenv import load_dotenv
import nest_asyncio

## Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

## Load user agent from .env file
load_dotenv('reddit_env.env')
user_agent = os.getenv('PolicyPosseReddit_UserAgent')

## Load Reddit API keys from JSON file
with open('reddit_api_keys.json') as f:
    reddit_api_keys = json.load(f)

## Constants for rate limiting
MAX_REQUESTS_PER_MINUTE = 70  # Max requests per API key per minute
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # Time between requests per key


########################################################################
## Helper function to initialize Reddit clients
########################################################################

async def initialize_reddit_clients():
    """
    Initialize Reddit clients using API keys from a JSON file, 
    with each client tied to a unique aiohttp session.
    """
    clients = []
    for group in reddit_api_keys.values():
        for key_info in group:
            session = aiohttp.ClientSession()
            reddit = asyncpraw.Reddit(
                client_id=key_info["client_id"],
                client_secret=key_info["api_key"],
                user_agent=user_agent,
                requestor_kwargs={"session": session}
            )
            clients.append((reddit, session))  # Add both Reddit client and session for later closing
    return clients


########################################################################
## Function to retrieve posts asynchronously
########################################################################

async def fetch_posts_for_subreddit(reddit_client, subreddit_name, max_posts=600):
    """
    Fetch top posts for the past year from a specific subreddit asynchronously.
    """
    try:
        subreddit = await reddit_client.subreddit(subreddit_name)
        posts = []
        async for submission in subreddit.top(limit=max_posts, time_filter="year"):
            posts.append({
                'post_id': submission.id,
                'state': subreddit_name,
                'title': submission.title,
                'selftext': submission.selftext,
                'created_utc': submission.created_utc,
                'score': submission.score,
                'url': submission.url,
                'num_comments': submission.num_comments,
                'author': str(submission.author) if submission.author else 'deleted'
            })
        return posts
    except Exception as e:
        print(f"Error fetching posts for {subreddit_name}: {e}")
        return []


########################################################################
## Main function to manage scraping with multiple API keys and rate limiting
########################################################################

async def main(subreddits):
    """
    Main function to manage asynchronous scraping with rate-limiting 
    and key rotation across multiple Reddit clients.
    """
    all_posts = []
    reddit_clients = await initialize_reddit_clients()  # List of (Reddit client, session) tuples
    num_clients = len(reddit_clients)
    client_index = 0  # Index to track the current Reddit client

    with tqdm(total=len(subreddits), desc="Scraping State Subreddits") as pbar:
        for subreddit_name in subreddits:
            ## Update tqdm with the current state
            pbar.set_description(f"Collecting data for {subreddit_name}")
            
            ## Rotate to the next client in the list
            reddit_client, session = reddit_clients[client_index]
            client_index = (client_index + 1) % num_clients  # Move to the next client

            ## Fetch posts for the subreddit
            posts = await fetch_posts_for_subreddit(reddit_client, subreddit_name)
            all_posts.extend(posts)

            ## Update the progress bar
            pbar.update(1)

            ## Rate limiting: Wait before the next request to stay within API limits
            await asyncio.sleep(REQUEST_INTERVAL)

    ## Save all posts to a CSV file
    df = pd.DataFrame(all_posts)
    df.to_csv('reddit_posts.csv', index=False)
    print("Scraping completed and data saved to reddit_posts.csv.")

    ## Close all Reddit client sessions
    for _, session in reddit_clients:
        await session.close()


########################################################################
## List of state subreddits to scrape
########################################################################

state_subreddits = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut",
    "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa",
    "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan",
    "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "NewHampshire",
    "NewJersey", "NewMexico", "NewYork", "NorthCarolina", "NorthDakota", "Ohio", "Oklahoma",
    "Oregon", "Pennsylvania", "RhodeIsland", "SouthCarolina", "SouthDakota", "Tennessee",
    "Texas", "Utah", "Vermont", "Virginia", "Washington", "WestVirginia", "Wisconsin", "Wyoming"
]


########################################################################
## Wrapper function to start the main function in an asyncio event loop
########################################################################

def run_main():
    """
    Wrapper function to execute the main scraping function within an asyncio event loop.
    """
    asyncio.run(main(state_subreddits))


########################################################################
## Execute the wrapper function
########################################################################

if __name__ == "__main__":
    run_main()