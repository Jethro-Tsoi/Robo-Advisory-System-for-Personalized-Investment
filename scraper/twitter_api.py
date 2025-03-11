import csv
import pandas as pd
import logging
import os
from dotenv import load_dotenv
import time
from typing import List, Dict, Optional, Any
from logging.handlers import RotatingFileHandler
import datetime
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
        # Use pandas with skipinitialspace=True to handle spaces after commas
        df = pd.read_csv(csv_path, skipinitialspace=True)
        accounts = df["X Handle"].tolist()
        # Clean up handles by removing @ if present
        accounts = [handle.lstrip('@') for handle in accounts if isinstance(handle, str)]
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

# X API v2 hard limits
X_API_TWEET_LIMIT = 3200  # X API v2 has a hard limit of 3200 most recent tweets per user

# Free tier limits
FREE_TIER_MONTHLY_READS = 100  # Free tier has 100 reads per month
FREE_TIER_MONTHLY_POSTS = 500  # Free tier has 500 posts per month

# Rate limiting configurations for free tier
FREE_TIER_RATE_LIMITS = {
    'user_timeline': {
        'requests_per_window': 1,  # 1 request per 15 minutes for free tier
        'window_size': 15 * 60,    # 15 minutes in seconds
    },
    'user_lookup': {
        'requests_per_window': 1,  # 1 request per 15 minutes for free tier
        'window_size': 15 * 60,    # 15 minutes in seconds
    }
}

# Error handling constants
INITIAL_BACKOFF = 60  # Initial backoff time in seconds
MAX_BACKOFF = 3600  # Maximum backoff time in seconds (1 hour)
BACKOFF_FACTOR = 2  # Exponential backoff multiplier

# Rate limit tracking
endpoint_limits = {
    'user_timeline': {
        'remaining': 1,  # Free tier: 1 request per 15-minute window
        'reset_time': time.time() + (15 * 60),
        'window_size': 15 * 60,  # 15 minutes in seconds
        'max_requests': 1
    },
    'user_lookup': {
        'remaining': 1,  # Free tier: 1 request per 15-minute window
        'reset_time': time.time() + (15 * 60),
        'window_size': 15 * 60,
        'max_requests': 1
    }
}

# Monthly quota tracking
monthly_quota = {
    'reads_used': 0,
    'reads_limit': FREE_TIER_MONTHLY_READS,
    'posts_used': 0,
    'posts_limit': FREE_TIER_MONTHLY_POSTS,
    'reset_date': None  # Will be set on first run
}

def initialize_rate_limits():
    """Initialize rate limits for all endpoints with current window."""
    current_time = time.time()
    for endpoint in endpoint_limits:
        endpoint_limits[endpoint]['reset_time'] = current_time + endpoint_limits[endpoint]['window_size']
        endpoint_limits[endpoint]['remaining'] = endpoint_limits[endpoint]['max_requests']
    
    # Initialize monthly quota if not set
    if not monthly_quota['reset_date']:
        # Set reset date to the 1st of next month
        today = datetime.datetime.now()
        if today.month == 12:
            next_month = datetime.datetime(today.year + 1, 1, 1)
        else:
            next_month = datetime.datetime(today.year, today.month + 1, 1)
        monthly_quota['reset_date'] = next_month.timestamp()
        logger.info(f"Monthly quota reset date set to {next_month.strftime('%Y-%m-%d')}")

# Session management
session_metrics = {
    'total_tweets': 0,
    'account_tweets': {},
    'last_request_time': None
}

def update_rate_limits(response, endpoint='user_timeline'):
    """Update rate limit information from response headers."""
    if not response:
        return
        
    # Check response headers first
    if hasattr(response, 'response') and hasattr(response.response, 'headers'):
        headers = response.response.headers
        if 'x-rate-limit-remaining' in headers:
            endpoint_limits[endpoint]['remaining'] = int(headers['x-rate-limit-remaining'])
        if 'x-rate-limit-reset' in headers:
            endpoint_limits[endpoint]['reset_time'] = int(headers['x-rate-limit-reset'])
    # Then check response meta
    elif hasattr(response, 'meta'):
        meta = response.meta
        if 'x-rate-limit-remaining' in meta:
            endpoint_limits[endpoint]['remaining'] = int(meta['x-rate-limit-remaining'])
        if 'x-rate-limit-reset' in meta:
            endpoint_limits[endpoint]['reset_time'] = int(meta['x-rate-limit-reset'])

def wait_for_rate_limit(endpoint):
    """Wait if necessary for rate limit reset."""
    limit_info = endpoint_limits[endpoint]
    current_time = time.time()
    
    # If we have remaining requests and know our limits, proceed
    if limit_info['remaining'] > 0 and limit_info['reset_time']:
        logger.debug(f"{endpoint}: {limit_info['remaining']} requests remaining until {time.ctime(limit_info['reset_time'])}")
        return
        
    # Need to wait for reset
    wait_time = max(0, limit_info['reset_time'] - current_time)
    if wait_time > 0:
        logger.info(f"Rate limit reached for {endpoint} ({limit_info['max_requests']} requests / {limit_info['window_size']/60} minutes).")
        logger.info(f"Waiting {wait_time:.0f} seconds until {time.ctime(limit_info['reset_time'])}...")
        time.sleep(wait_time + 1)  # Add 1 second buffer
    
    # Reset the limits after waiting
    limit_info['remaining'] = limit_info['max_requests']
    limit_info['reset_time'] = time.time() + limit_info['window_size']
    logger.info(f"Rate limits reset for {endpoint}. {limit_info['remaining']} requests available.")

def check_monthly_quota(read_count=0):
    """
    Check if monthly quota has been reached and update usage.
    
    Args:
        read_count (int): Number of reads to add to the quota
        
    Returns:
        bool: True if quota is available, False if exceeded
    """
    # Check if we need to reset the monthly quota
    current_time = time.time()
    if current_time > monthly_quota['reset_date']:
        # Reset monthly quota
        today = datetime.datetime.now()
        if today.month == 12:
            next_month = datetime.datetime(today.year + 1, 1, 1)
        else:
            next_month = datetime.datetime(today.year, today.month + 1, 1)
        monthly_quota['reset_date'] = next_month.timestamp()
        monthly_quota['reads_used'] = 0
        monthly_quota['posts_used'] = 0
        logger.info(f"Monthly quota reset. Next reset on {next_month.strftime('%Y-%m-%d')}")
    
    # Update and check quota
    if read_count > 0:
        new_total = monthly_quota['reads_used'] + read_count
        if new_total > monthly_quota['reads_limit']:
            logger.warning(f"Monthly read quota would be exceeded: {new_total}/{monthly_quota['reads_limit']}")
            return False
        monthly_quota['reads_used'] = new_total
        logger.info(f"Monthly read quota updated: {monthly_quota['reads_used']}/{monthly_quota['reads_limit']}")
    
    return True

# Initialize Tweepy client with rate limit handling
client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

def get_last_tweet_id(username: str) -> Optional[str]:
    """
    Retrieve the ID of the last fetched tweet.
    
    Args:
        username (str): Twitter handle (without @).
        
    Returns:
        str: ID of the last tweet or None if no tweets found.
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

def fetch_tweets(username: str, since_id: Optional[str] = None, max_tweets: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch tweets for a given Twitter username using X API v2 (formerly Twitter API).
    
    Args:
        username (str): Twitter handle (without @).
        since_id (str): Returns results with an ID greater than (that is, more recent than) the specified ID.
        max_tweets (int): Maximum number of tweets to fetch (default: 100, respects free tier limits)
        
    Returns:
        list: List of tweet objects.
    """
    try:
        tweets = []
        current_backoff = INITIAL_BACKOFF
        
        # Check monthly quota before making any requests
        # Each API request counts as 1 read, not each tweet
        # Calculate how many requests we'll need
        requests_needed = (max_tweets + 99) // 100  # Ceiling division to get number of requests
        
        if not check_monthly_quota(requests_needed):
            logger.warning(f"Monthly read quota would be exceeded. Limiting to available quota.")
            available_requests = max(0, monthly_quota['reads_limit'] - monthly_quota['reads_used'])
            max_tweets = available_requests * 100  # Each request can fetch up to 100 tweets
            if max_tweets <= 0:
                logger.error("Monthly read quota already exhausted. Cannot fetch tweets.")
                return []
        
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
            tweet_count = 0
            max_results_per_request = min(100, max_tweets)  # Maximum 100 per request
            
            while tweet_count < max_tweets:
                try:
                    # Wait for rate limit and get tweets
                    wait_for_rate_limit('user_timeline')
                    
                    # Calculate how many tweets to request in this batch
                    remaining_tweets = max_tweets - tweet_count
                    current_max_results = min(max_results_per_request, remaining_tweets)
                    
                    logger.info(f"Requesting {current_max_results} tweets for @{username} (page {page_count+1})")
                    
                    response = client.get_users_tweets(
                        user_id,
                        tweet_fields=['id', 'text', 'created_at', 'public_metrics', 'referenced_tweets'],
                        pagination_token=pagination_token,
                        since_id=since_id,
                        max_results=current_max_results,
                        exclude=['retweets']
                    )
                    update_rate_limits(response, 'user_timeline')
                    
                    if not response.data:
                        logger.info(f"No more tweets available for @{username}")
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
                    
                    tweets.extend(batch_tweets)
                    tweet_count += len(batch_tweets)
                    logger.info(f"Fetched {len(batch_tweets)} tweets from page {page_count} for @{username} (Total: {tweet_count}/{max_tweets})")
                    
                    # Save tweets in smaller batches to avoid data loss
                    if len(tweets) >= 50:  # Save more frequently with smaller batches
                        logger.info(f"Saving batch of {len(tweets)} tweets for @{username}")
                        save_tweets(username, tweets)
                        tweets = []
                    
                    # Check if we have more pages
                    if not response.meta or 'next_token' not in response.meta:
                        logger.info(f"No more pages available for @{username}")
                        break
                    
                    # Update pagination token for next page
                    pagination_token = response.meta['next_token']
                    
                    # Check if we've reached the tweet limit
                    if tweet_count >= max_tweets:
                        logger.info(f"Reached maximum tweet limit ({max_tweets}) for @{username}")
                        break
                    
                    # Update request time for rate tracking
                    session_metrics['last_request_time'] = time.time()
                    
                    # Add a small delay between requests to be extra cautious with rate limits
                    time.sleep(2)
                    
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
    
    This function saves the collected tweets to a CSV file with a filename based on the 
    Twitter username. If the file does not exist, it will be created with appropriate headers.
    If it exists, new tweets will be appended to it.
    
    Args:
        username (str): Twitter handle (without @).
        tweets (list): List of tweet objects.
        
    Note:
        Due to the X API v2 limit of 3,200 tweets per user, older historical tweets
        beyond this limit will not be included in the dataset unless they were previously
        collected and saved.
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

def scrape_account(username: str, max_tweets: int = 100):
    """
    Scrape tweets for a single Twitter account.
    
    Args:
        username (str): Twitter handle (without @).
        max_tweets (int): Maximum number of tweets to fetch (default: 100 for free tier)
    """
    logger.info(f"Starting to scrape tweets for @{username}")
    
    since_id = get_last_tweet_id(username)
    if since_id:
        logger.info(f"Fetching tweets for @{username} since ID {since_id}")
    else:
        logger.info(f"Fetching all available tweets for @{username} (limited to {max_tweets})")
    
    tweets = fetch_tweets(username, since_id, max_tweets)
    if tweets:
        logger.info(f"Found {len(tweets)} new tweets for @{username}")
    else:
        logger.info(f"No new tweets found for @{username}")
    
    logger.info(f"Completed scraping for @{username}")

def process_account_batch(accounts: List[str], max_tweets_per_account: int = 100) -> None:
    """
    Process a batch of accounts while respecting rate limits.
    
    Args:
        accounts: List of account handles to process
        max_tweets_per_account: Maximum tweets to fetch per account (default: 100 for free tier)
    """
    total = len(accounts)
    completed = 0
    
    # Calculate available quota per account
    available_quota = monthly_quota['reads_limit'] - monthly_quota['reads_used']
    if available_quota <= 0:
        logger.error("Monthly read quota exhausted. Cannot process accounts.")
        return
    
    # Each account will need at least 1 request (user lookup) + 1 request per 100 tweets
    requests_per_account = 1 + (max_tweets_per_account + 99) // 100
    accounts_we_can_process = available_quota // requests_per_account
    
    if accounts_we_can_process < len(accounts):
        logger.warning(f"Can only process {accounts_we_can_process} accounts with available quota")
        accounts = accounts[:accounts_we_can_process]
    
    logger.info(f"Processing {len(accounts)} accounts with {max_tweets_per_account} tweets per account")
    
    # Process accounts sequentially to avoid rate limit issues
    for username in accounts:
        try:
            # Check if we still have quota
            if not check_monthly_quota():
                logger.warning("Monthly quota exhausted during batch processing. Stopping.")
                break
            
            scrape_account(username.lstrip('@'), max_tweets_per_account)
            completed += 1
            logger.info(f"Progress: {completed}/{total} accounts processed")
                
        except tweepy.TooManyRequests:
            logger.warning(f"Rate limit reached during processing of @{username}")
            # Let the rate limit handler deal with it
            continue
        except Exception as exc:
            logger.error(f"Exception for @{username}: {exc}")

def main():
    logger.info("Starting X (Twitter) scraping process...")
    logger.info(f"Note: X API v2 limits access to a maximum of {X_API_TWEET_LIMIT} most recent tweets per user")
    logger.info(f"Free tier limits: {FREE_TIER_MONTHLY_READS} reads per month, {FREE_TIER_MONTHLY_POSTS} posts per month")
    
    # Initialize rate limits
    initialize_rate_limits()
    logger.info("Rate limits initialized")
    
    accounts = load_accounts('account.csv')
    if not accounts:
        logger.error("No accounts to process. Exiting.")
        return
    
    logger.info(f"Found {len(accounts)} accounts to process: {', '.join(accounts)}")
    
    # Check monthly quota
    if not check_monthly_quota():
        logger.error("Monthly quota already exhausted. Exiting.")
        return
    
    # Calculate tweets per account based on available quota
    available_quota = monthly_quota['reads_limit'] - monthly_quota['reads_used']
    tweets_per_account = min(100, available_quota // max(1, len(accounts)))
    
    # Process accounts one at a time to avoid rate limit issues
    process_account_batch(accounts, max_tweets_per_account=tweets_per_account)
    
    logger.info("Scraping process completed.")
    logger.info(f"Monthly quota usage: {monthly_quota['reads_used']}/{monthly_quota['reads_limit']} reads")

def get_api_limits_info():
    """
    Returns information about the X API v2 limits.
    
    This function provides detailed information about the limits 
    of the X API v2 for educational purposes.
    
    Returns:
        dict: Dictionary containing information about API limits
    """
    return {
        "user_timeline_limit": X_API_TWEET_LIMIT,
        "per_request_limit": 100,  # Maximum tweets per request
        "pages_needed_for_max": 32,  # To reach 3,200 tweets
        "free_tier_limits": {
            "monthly_reads": FREE_TIER_MONTHLY_READS,
            "monthly_posts": FREE_TIER_MONTHLY_POSTS,
            "rate_limits": {
                "user_timeline": "1 request per 15-minute window",
                "user_lookup": "1 request per 15-minute window"
            }
        },
        "notes": [
            "X API v2 has a hard limit of 3,200 most recent tweets per user",
            "Free tier is limited to 100 reads per month",
            "Historical tweets beyond 3,200 are not accessible through standard API",
            "Premium API access is required for full tweet archives",
            "This limit is enforced by the Twitter/X platform, not by this code"
        ]
    }

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
