# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 02:35:09 2024

@author: dforc

Description:
    This script asynchronously downloads images from URLs extracted from a 
    Reddit posts dataset. It uses asyncio for concurrency, aiohttp for HTTP 
    requests, and retry logic with exponential backoff to handle rate limits 
    and transient errors. Downloaded images are saved locally in a specified 
    directory, with filenames corresponding to the post IDs.
"""

import os
import re
import aiohttp
import asyncio
from aiohttp import ClientSession
from tqdm import tqdm
import pandas as pd
import nest_asyncio
import time

# ##########
# ## Initialize asyncio for nested event loops
# ##########
nest_asyncio.apply()

# ##########
# ## Load and Preprocess the Data
# ##########

# Load the dataset containing Reddit posts
posts_df = pd.read_csv('reddit_posts.csv')  # File should contain a 'url' column for post URLs

# Define valid image extensions
image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

# Clean and normalize URLs in the dataset
posts_df['url'] = posts_df['url'].apply(lambda x: re.sub(r"\\+", "/", str(x)).strip())

# Filter URLs that point to image files
posts_df['image_url'] = posts_df['url'].apply(lambda x: x if x.lower().endswith(image_extensions) else None)

# Extract (post_id, image_url) pairs for valid image URLs
image_urls = posts_df[['post_id', 'image_url']].dropna().values

# ##########
# ## Create Directory for Images
# ##########

# Ensure the target directory exists
os.makedirs("post_images", exist_ok=True)

# ##########
# ## Asynchronous Image Download Function
# ##########

async def download_image(post_id, url, session, pbar, retries=8, initial_delay=1):
    """
    Downloads an image asynchronously with retry logic and exponential backoff.

    Parameters:
        post_id (str): The unique ID of the post (used as filename).
        url (str): The URL of the image to download.
        session (aiohttp.ClientSession): The HTTP session for the request.
        pbar (tqdm): Progress bar for tracking download progress.
        retries (int): Number of retry attempts for failed downloads.
        initial_delay (int): Initial delay in seconds before retrying.

    Returns:
        None
    """
    image_name = os.path.join("post_images", f"{post_id}.jpg")  # Save image as post_id.jpg
    delay = initial_delay

    for attempt in range(retries):
        try:
            async with session.get(url) as response:
                if response.status == 200:  # Success
                    with open(image_name, "wb") as f:
                        f.write(await response.read())
                    pbar.update(1)  # Update progress bar
                    return
                elif response.status == 429:  # Rate limit hit
                    print(f"Rate limit hit for {url}, retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    print(f"Failed to download {url} with status {response.status}, skipping.")
                    break
        except Exception as e:
            print(f"Error downloading {url}: {e}, retrying in {delay} seconds...")
            await asyncio.sleep(delay)
            delay *= 2  # Exponential backoff
    pbar.update(1)  # Update progress bar even if all attempts fail

# ##########
# ## Asynchronous Task Manager
# ##########

async def download_all_images(post_id_url_pairs):
    """
    Manages the download of all images using asyncio and aiohttp.

    Parameters:
        post_id_url_pairs (list): List of (post_id, image_url) tuples.

    Returns:
        None
    """
    async with ClientSession() as session:  # HTTP session for all requests
        with tqdm(total=len(post_id_url_pairs), desc="Downloading Images") as pbar:
            # Create a task for each image download
            tasks = [download_image(post_id, url, session, pbar) for post_id, url in post_id_url_pairs]
            await asyncio.gather(*tasks)  # Run all tasks concurrently

# ##########
# ## Execute the Download Process
# ##########

# Run the asynchronous download process
asyncio.run(download_all_images(image_urls))
