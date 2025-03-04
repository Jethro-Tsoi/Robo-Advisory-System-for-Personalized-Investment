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

# Rate limiting configurations
MAX_TWEETS_PER_ACCOUNT = 1000  # Daily limit per account
MAX_TOTAL_TWEETS = 5000  # Total daily limit

# Error handling constants
INITIAL_BACKOFF = 60  # Initial backoff time in seconds
MAX_BACKOFF = 3600  # Maximum backoff time in seconds (1 hour)
BACKOFF_FACTOR = 2  # Exponential backoff multiplier

# Rate limit tracking
endpoint_limits = {
    'user_timeline': {
        'remaining': 900,  # Requests per 15-minute window
        'reset_time': None,
        'window_size': 15 * 60  # 15 minutes in seconds
    },
    'user_lookup': {
        'remaining': 300,  # Requests per 15-minute window
        'reset_time': None,
        'window_size': 15 * 60
    }
}

# Session management
session_metrics = {
    'total_tweets': 0,
    'account_tweets': {},
    'last_request_time': None
}

def update_rate_limits(response, endpoint='user_timeline'):
    """Update rate limit information from response headers."""
    if hasattr(response, 'meta'):
        meta = response.meta
        if 'x-rate-limit-remaining' in meta:
            endpoint_limits[endpoint]['remaining'] = int(meta['x-rate-limit-remaining'])
        if 'x-rate-limit-reset' in meta:
            endpoint_limits[endpoint]['reset_time'] = int(meta['x-rate-limit-reset'])

def check_rate_limit(endpoint):
    """
    Check if we can make a request to the specified endpoint.
    Returns tuple: (can_request, wait_time)
    """
    limit_info = endpoint_limits[endpoint]
    current_time = time.time()
    
    # If we have requests remaining, allow it
    if limit_info['remaining'] > 0:
        return True, 0
        
    # If reset time is set, calculate wait time
    if limit_info['reset_time']:
        wait_time = limit_info['reset_time'] - current_time
        if wait_time <= 0:
            # Reset window has passed
            limit_info['remaining'] = limit_info['window_size']
            limit_info['reset_time'] = current_time + limit_info['window_size']
            return True, 0
        return False, wait_time
        
    # If no reset time set, use conservative window
    limit_info['reset_time'] = current_time + limit_info['window_size']
    return False, limit_info['window_size']

def wait_for_rate_limit(endpoint):
    """Wait if necessary for rate limit reset."""
    can_request, wait_time = check_rate_limit(endpoint)
    if not can_request:
        logger.info(f"Rate limit reached for {endpoint}. Waiting {wait_time:.0f} seconds...")
        time.sleep(wait_time)
        endpoint_limits[endpoint]['remaining'] = endpoint_limits[endpoint]['window_size']
        endpoint_limits[endpoint]['reset_time'] = time.time() + endpoint_limits[endpoint]['window_size']

# Initialize Tweepy client with rate limit handling
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def check_session_limits():
    """Check if session limits have been reached."""
    # Check total tweets limit
    if session_metrics['total_tweets'] >= MAX_TOTAL_TWEETS:
        logger.info("Maximum total tweets limit reached for today")
        return False
    return True

def check_account_limits(username: str, tweet_count: int) -> bool:
    """Check if account-specific limits have been reached."""
    current_count = session_metrics['account_tweets'].get(username, 0)
    new_count = current_count + tweet_count
    
    if new_count > MAX_TWEETS_PER_ACCOUNT:
        logger.info(f"Account @{username} has reached its daily tweet limit")
        return False
        
    session_metrics['account_tweets'][username] = new_count
    session_metrics['total_tweets'] += tweet_count
    return True

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
        tweets = []
        current_backoff = INITIAL_BACKOFF
        
        try:
            # First get user ID from username
            wait_for_rate_limit('user_lookup')
            user = client.get_user(username=username)
            if not user.data:
                logger.error(f"Could not find user @{username}")
                return []
            
            update_rate_limits(user, 'user_lookup')
            user_id = user.data.id
            logger.info(f"Found user ID for @{username}: {user_id}")
            
            # Parameters for pagination
            pagination_token = None
            page_count = 0
            max_pages = 50  # Reduced from 1000 to be more conservative
            
            while True:
                try:
                    # Wait for rate limit and get tweets
                    wait_for_rate_limit('user_timeline')
                    response = client.get_users_tweets(
                        user_id,
                        tweet_fields=['id', 'text', 'created_at', 'public_metrics', 'referenced_tweets'],
                        pagination_token=pagination_token,
                        since_id=since_id,
                        max_results=100,  # Maximum allowed by Twitter API
                        exclude=['retweets']
                    )
                    update_rate_limits(response, 'user_timeline')
                    
                    if not response.data:
                        break
                    
                    page_count += 1
                    logger.info(f"Processing page {page_count} for @{username}")
                    
                    batch_tweets = []
                    for tweet in response.data:
                        # Check if it's a retweet or quote tweet
                        is_retweet = False
                        is_quote = False
                        if tweet.referenced_tweets:
                            for ref in tweet.referenced_tweets:
                                if ref.type == 'retweeted':
                                    is_retweet = True
                                elif ref.type == 'quoted':
                                    is_quote = True
                        
                        tweet_type = 'retweet' if is_retweet else 'quote' if is_quote else 'original'
                        
                        batch_tweets.append({
                            'id': tweet.id,
                            'text': tweet.text.replace('\n', ' ').replace('\r', ' '),
                            'created_at': tweet.created_at.isoformat(),
                            'type': tweet_type,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'quote_count': tweet.public_metrics['quote_count']
                        })
                    
                    # Check account limits before adding tweets
                    if not check_account_limits(username, len(batch_tweets)):
                        break
                        
                    tweets.extend(batch_tweets)
                    logger.info(f"Fetched {len(batch_tweets)} tweets from page {page_count} for @{username}")
                    
                    # Save tweets in smaller batches
                    if len(tweets) >= 200:  # Reduced from 500
                        logger.info(f"Saving batch of {len(tweets)} tweets for @{username}")
                        save_tweets(username, tweets)
                        tweets = []
                    
                    # Check if we have more pages
                    if not response.meta or 'next_token' not in response.meta:
                        logger.info(f"No more pages available for @{username}")
                        break
                    
                    # Update pagination token for next page
                    pagination_token = response.meta['next_token']
                    
                    # Check if we've reached max pages
                    if page_count >= max_pages:
                        logger.info(f"Reached maximum page limit ({max_pages}) for @{username}")
                        break
                    
                    # Update request time for rate tracking
                    session_metrics['last_request_time'] = time.time()
                    
                except tweepy.TooManyRequests as e:
                    logger.warning(f"Rate limit hit on page {page_count} for @{username}")
                    if tweets:
                        save_tweets(username, tweets)
                        tweets = []
                    
                    # Use our rate limit tracking
                    endpoint_limits['user_timeline']['remaining'] = 0
                    if hasattr(e.response, 'headers') and 'x-rate-limit-reset' in e.response.headers:
                        reset_time = int(e.response.headers['x-rate-limit-reset'])
                        endpoint_limits['user_timeline']['reset_time'] = reset_time
                    wait_for_rate_limit('user_timeline')
                    continue
                
                except Exception as e:
                    logger.error(f"Error on page {page_count} for @{username}: {str(e)}")
                    if tweets:
                        save_tweets(username, tweets)
                        tweets = []
                    time.sleep(current_backoff)
                    current_backoff = min(current_backoff * BACKOFF_FACTOR, MAX_BACKOFF)
                    continue
            
            # Save any remaining tweets
            if tweets:
                logger.info(f"Saving final batch of {len(tweets)} tweets for @{username}")
                save_tweets(username, tweets)
            
            logger.info(f"Completed fetching tweets for @{username}. Total pages processed: {page_count}")
            return tweets
            
        except tweepy.TooManyRequests as e:
            logger.warning(f"Rate limit hit while fetching tweets for @{username}")
            if tweets:
                save_tweets(username, tweets)
            raise e
            
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
            writer = csv.DictWriter(file, fieldnames=['id', 'text', 'created_at', 'type', 
                                                    'retweet_count', 'reply_count', 
                                                    'like_count', 'quote_count'])
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

def process_account_batch(accounts: List[str], batch_size: int = 3) -> None:
    """
    Process a batch of accounts while respecting rate limits.
    
    Args:
        accounts: List of account handles to process
        batch_size: Number of accounts to process in parallel
    """
    total = len(accounts)
    completed = 0
    
    # Process accounts in batches
    for i in range(0, total, batch_size):
        batch = accounts[i:i + batch_size]
        logger.info(f"Processing batch of {len(batch)} accounts: {', '.join(batch)}")
        
        # Use ThreadPoolExecutor for parallel processing within batch
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            futures = {executor.submit(scrape_account, username.lstrip('@')): username for username in batch}
            
            for future in as_completed(futures):
                username = futures[future]
                try:
                    future.result()
                    completed += 1
                    logger.info(f"Progress: {completed}/{total} accounts processed")
                except tweepy.TooManyRequests:
                    # Rate limits are now handled by wait_for_rate_limit
                    logger.warning(f"Rate limit reached during processing of @{username}")
                except Exception as exc:
                    logger.error(f"Exception for @{username}: {exc}")
                    
        # Check session limits after each batch
        if not check_session_limits():
            logger.info("Session limits reached. Stopping batch processing.")
            break

def main():
    logger.info("Starting Twitter scraping process...")
    
    accounts = load_accounts('account.csv')
    if not accounts:
        logger.error("No accounts to process. Exiting.")
        return
    
    logger.info(f"Found {len(accounts)} accounts to process: {', '.join(accounts)}")
    
    if not check_session_limits():
        logger.info("Session limits reached. Exiting.")
        return
    
    # Process accounts in smaller batches
    process_account_batch(accounts, batch_size=3)
    
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
