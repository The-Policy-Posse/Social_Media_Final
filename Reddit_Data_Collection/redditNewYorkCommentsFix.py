# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:11:31 2024

@author: dforc
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
from datetime import datetime, timezone

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load user agent from .env file
load_dotenv('reddit_env.env')
user_agent = os.getenv('PolicyPosseReddit_UserAgent')

# Load Reddit API keys from JSON file
with open('reddit_api_keys.json') as f:
    reddit_api_keys = json.load(f)

# Constants for rate limiting
MAX_REQUESTS_PER_MINUTE = 70  # Max requests per API key per minute
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE  # Time between requests per key

# Helper function to initialize Reddit clients
async def initialize_reddit_clients():
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

# Function to fetch all comments for a specific post
async def fetch_comments_for_post(reddit_client, submission_id, state, rate_limiter):
    async with rate_limiter:  # Use semaphore to control rate limit
        try:
            submission = await reddit_client.submission(id=submission_id)
            await submission.comments.replace_more(limit=None)  # Load all "More Comments"
            comments_data = []
            for comment in submission.comments.list():
                comments_data.append({
                    'post_id': submission_id,
                    'state': state,
                    'comment_id': comment.id,
                    'body': comment.body,
                    'created_utc': datetime.fromtimestamp(comment.created_utc, tz=timezone.utc).isoformat(),
                    'score': comment.score,
                    'author': str(comment.author) if comment.author else 'deleted'
                })
            return comments_data
        except Exception as e:
            print(f"Error fetching comments for post {submission_id}: {e}")
            return []

# Function to write comments to CSV in append mode
def write_comments_to_csv(comments, state, file_name='reddit_comments.csv'):
    df = pd.DataFrame(comments)
    df.to_csv(file_name, mode='a', index=False, header=not os.path.exists(file_name))
    print(f"State {state} complete, writing to CSV")  # Print statement for state completion

# Main function to process comments for New York posts only
async def main(posts):
    reddit_clients = await initialize_reddit_clients()  # List of (Reddit client, session) tuples
    num_clients = len(reddit_clients)

    # Semaphore to manage rate limit (12 clients with 70 requests per minute each)
    rate_limiter = asyncio.Semaphore(num_clients * MAX_REQUESTS_PER_MINUTE / 60)

    state_comments = []  # Collect all comments for batch writing

    async def process_post(row, reddit_client):
        submission_id = row['post_id']
        state = row['state']

        comments = await fetch_comments_for_post(reddit_client, submission_id, state, rate_limiter)
        state_comments.extend(comments)

    tasks = []
    with tqdm(total=len(posts), desc="Fetching Comments for New York") as pbar:
        for index, row in posts.iterrows():
            reddit_client, _ = reddit_clients[index % num_clients]
            task = process_post(row, reddit_client)
            tasks.append(task)
            pbar.update(1)

            if len(tasks) >= num_clients * 2:  # Double the concurrency for New York
                await asyncio.gather(*tasks)
                tasks = []  # Clear tasks list once batch is processed

        if tasks:
            await asyncio.gather(*tasks)

        if state_comments:
            write_comments_to_csv(state_comments, "newyork")

    for _, session in reddit_clients:
        await session.close()

# Load posts filtered for New York
def load_posts():
    posts = pd.read_csv('reddit_posts.csv')
    return posts[posts['state'] == 'NewYork']

# Wrapper function to start the main function in an asyncio event loop
def run_main():
    posts = load_posts()
    asyncio.run(main(posts))

# Execute the wrapper function
if __name__ == "__main__":
    run_main()