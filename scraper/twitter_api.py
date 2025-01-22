import csv
import logging
import os
from dotenv import load_dotenv, dotenv_values
import time
from typing import List
from logging.handlers import RotatingFileHandler

import tweepy
from concurrent.futures import ThreadPoolExecutor, as_completed


load_dotenv() 
# Configure logging
logger = logging.getLogger('TwitterScraper')
logger.setLevel(logging.DEBUG)
os.makedirs('../logs', exist_ok=True)
handler = RotatingFileHandler('../logs/scraper.log', maxBytes=5*1024*1024, backupCount=2)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Add console handler for better visibility
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def load_accounts(csv_path: str) -> List[str]:
    """
    Load Twitter accounts from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        list: List of Twitter account handles.
    """
    accounts = []
    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                twitter_handle = row.get('Twitter Account', '').strip()
                if twitter_handle:
                    accounts.append(twitter_handle)
        logger.info(f"Loaded {len(accounts)} accounts from {csv_path}")
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading accounts: {e}")
    return accounts

# Load Bearer Token from environment variable for security
BEARER_TOKEN = os.getenv('BEARER_TOKEN')

if not BEARER_TOKEN:
    logger.error("Bearer Token not found. Please set the BEARER_TOKEN environment variable.")
    exit(1)

# Initialize Tweepy client with rate limit handling
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def get_last_tweet_id(username: str) -> str:
    """
    Retrieve the ID of the last fetched tweet.
    
    Args:
        username (str): Twitter handle (without @).
        
    Returns:
        str: ID of the last tweet.
    """
    file_path = f"../data/tweets/{username}_tweets.csv"
    try:
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            if rows:
                return rows[-1]['id']
    except FileNotFoundError:
        # File does not exist, implying no tweets have been fetched yet
        pass
    except Exception as e:
        logger.error(f"Error reading last tweet ID for @{username}: {e}")
    return None

def fetch_tweets(username: str, since_id: str = None) -> List[dict]:
    """
    Fetch tweets for a given Twitter username using Tweepy.
    
    Args:
        username (str): Twitter handle (without @).
        since_id (str): Returns results with an ID greater than (that is, more recent than) the specified ID.
        
    Returns:
        list: List of tweet objects.
    """
    try:
        query = f"from:{username} -is:retweet"
        tweets = []
        
        try:
            paginator = tweepy.Paginator(
                client.search_recent_tweets,
                query=query,
                tweet_fields=['id', 'text', 'created_at'],
                since_id=since_id,
                max_results=100
            )
            
            for response in paginator:
                if response and response.data:
                    for tweet in response.data:
                        tweets.append({
                            'id': tweet.id,
                            'text': tweet.text.replace('\n', ' ').replace('\r', ' '),
                            'created_at': tweet.created_at.isoformat()
                        })
                    logger.info(f"Fetched batch of {len(response.data)} tweets for @{username}")
                    
                    # Save tweets in batches to prevent data loss during rate limiting
                    if len(tweets) >= 100:
                        save_tweets(username, tweets)
                        tweets = []
                        
        except tweepy.TooManyRequests as e:
            logger.warning(f"Rate limit hit while fetching tweets for @{username}. Saving current batch.")
            if tweets:  # Save any tweets we got before the rate limit
                save_tweets(username, tweets)
            raise e
            
        # Save any remaining tweets
        if tweets:
            save_tweets(username, tweets)
            
        return tweets
        
    except tweepy.TweepyException as e:
        logger.error(f"Tweepy error occurred for @{username}: {e}")
    except Exception as e:
        logger.error(f"Error occurred for @{username}: {e}")
    return []

def save_tweets(username: str, tweets: list):
    """
    Save tweets to the account's CSV file.
    
    Args:
        username (str): Twitter handle (without @).
        tweets (list): List of tweet objects.
    """
    # Create data directory if it doesn't exist
    os.makedirs('../data/tweets', exist_ok=True)
    
    file_path = f"../data/tweets/{username}_tweets.csv"
    file_exists = os.path.isfile(file_path)
    
    try:
        with open(file_path, mode='a', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=['id', 'text', 'created_at'])
            if not file_exists:
                writer.writeheader()
            for tweet in tweets:
                writer.writerow(tweet)
        logger.info(f"Saved {len(tweets)} tweets for @{username} to {file_path}")
    except Exception as e:
        logger.error(f"Error saving tweets for @{username}: {e}")

def scrape_account(username: str):
    """
    Scrape tweets for a single Twitter account.
    
    Args:
        username (str): Twitter handle (without @).
    """
    logger.info(f"Starting to scrape tweets for @{username}")
    
    since_id = get_last_tweet_id(username)
    if since_id:
        logger.info(f"Fetching tweets for @{username} since ID {since_id}")
    else:
        logger.info(f"Fetching all available tweets for @{username}")
    
    tweets = fetch_tweets(username, since_id)
    if tweets:
        logger.info(f"Found {len(tweets)} new tweets for @{username}")
        save_tweets(username, tweets)
    else:
        logger.info(f"No new tweets found for @{username}")
    
    logger.info(f"Completed scraping for @{username}")

def main():
    logger.info("Starting Twitter scraping process...")
    
    accounts = load_accounts('account.csv')
    if not accounts:
        logger.error("No accounts to process. Exiting.")
        return
    
    logger.info(f"Found {len(accounts)} accounts to process: {', '.join(accounts)}")
    
    # Reduce number of workers to avoid hitting rate limits too quickly
    max_workers = 3
    logger.info(f"Initializing thread pool with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_username = {executor.submit(scrape_account, username.lstrip('@')): username for username in accounts}
        
        completed = 0
        total = len(accounts)
        for future in as_completed(future_to_username):
            username = future_to_username[future]
            try:
                future.result()
                completed += 1
                logger.info(f"Progress: {completed}/{total} accounts processed")
            except tweepy.TooManyRequests:
                logger.warning("Rate limit reached. Waiting before continuing...")
                time.sleep(900)  # Wait 15 minutes for rate limit reset
            except Exception as exc:
                logger.error(f"Exception for @{username}: {exc}")
            
    logger.info("Scraping process completed.")

if __name__ == "__main__":
    try:
        logger.info("Checking environment setup...")
        if not BEARER_TOKEN:
            logger.error("BEARER_TOKEN not found in environment variables")
            exit(1)
        
        if not os.path.exists('account.csv'):
            logger.error("account.csv file not found in scraper directory")
            exit(1)
            
        logger.info("Environment check completed, starting main process...")
        main()
    except Exception as e:
        logger.error(f"Fatal error occurred: {str(e)}", exc_info=True)
        exit(1)