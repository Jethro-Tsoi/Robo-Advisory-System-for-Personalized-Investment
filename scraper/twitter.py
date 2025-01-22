import asyncio
import csv
import logging
from datetime import datetime
from pathlib import Path
from twscrape import API, gather
from twscrape.logger import set_log_level

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('twitter_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TwitterScraper:
    def __init__(self):
        self.api = API()
        self.output_file = "tweets_output.csv"
        self.accounts = []
        self.progress_file = "scraping_progress.txt"
        self.processed_tweets_file = "processed_tweets.txt"
        self.credentials_file = "scraper/credentials.csv"
        
    async def setup(self):
        """Initialize the Twitter API with accounts loaded from CSV"""
        try:
            credentials = self.load_credentials(self.credentials_file)
            for cred in credentials:
                await self.api.pool.add_account(
                    username=cred['username'],
                    password=cred['password'],
                    email=cred['email'],
                    email_password=cred['email_password'],
                    auth_type=cred.get('auth_type', 'standard')  # default to standard auth
                )
            await self.api.pool.login_all()
            logger.info("Successfully logged in to Twitter API with provided credentials")
        except Exception as e:
            logger.error(f"Error setting up Twitter API: {e}")
            raise

    def load_credentials(self, csv_file):
        """Load Twitter account credentials from CSV file"""
        credentials = []
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if all(key in row for key in ['username', 'password', 'email', 'email_password']):
                        credentials.append({
                            'username': row['username'],
                            'password': row['password'],
                            'email': row['email'],
                            'email_password': row['email_password'],
                            'auth_type': row.get('auth_type', 'standard')  # Optional: specify auth type
                        })
            logger.info(f"Loaded {len(credentials)} Twitter account credentials from CSV")
            return credentials
        except Exception as e:
            logger.error(f"Error loading credentials from CSV: {e}")
            raise

    def load_accounts(self, csv_file):
        """Load Twitter accounts to scrape from CSV file"""
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('Twitter Account'):
                        # Remove @ symbol if present
                        username = row['Twitter Account'].replace('@', '').strip()
                        if username:
                            self.accounts.append(username)
            logger.info(f"Loaded {len(self.accounts)} Twitter accounts to scrape from CSV")
        except Exception as e:
            logger.error(f"Error loading accounts from CSV: {e}")
            raise

    def get_processed_tweets(self):
        """Get set of already processed tweet IDs"""
        processed_tweets = set()
        try:
            if Path(self.processed_tweets_file).exists():
                with open(self.processed_tweets_file, 'r') as f:
                    processed_tweets = set(line.strip() for line in f)
            return processed_tweets
        except Exception as e:
            logger.error(f"Error reading processed tweets file: {e}")
            return set()

    def save_processed_tweet_ids(self, tweet_ids):
        """Save processed tweet IDs to file"""
        try:
            with open(self.processed_tweets_file, 'a') as f:
                for tweet_id in tweet_ids:
                    f.write(f"{tweet_id}\n")
        except Exception as e:
            logger.error(f"Error saving processed tweet IDs: {e}")

    async def scrape_user_tweets(self, username):
        """Scrape tweets for a single user"""
        try:
            processed_tweets = self.get_processed_tweets()
            tweets = []
            new_tweet_ids = set()
            
            async for tweet in self.api.user_tweets(username, limit=100):  # Adjust limit as needed
                if str(tweet.id) not in processed_tweets:
                    tweets.append({
                        'username': username,
                        'tweet_id': tweet.id,
                        'created_at': tweet.date,
                        'text': tweet.rawContent,
                        'likes': tweet.likeCount,
                        'retweets': tweet.retweetCount,
                        'replies': tweet.replyCount,
                        'views': tweet.viewCount,
                        'language': tweet.language
                    })
                    new_tweet_ids.add(str(tweet.id))
            
            if new_tweet_ids:
                self.save_processed_tweet_ids(new_tweet_ids)
                logger.info(f"Found {len(tweets)} new tweets from @{username}")
            else:
                logger.info(f"No new tweets found for @{username}")
                
            return tweets
        except Exception as e:
            logger.error(f"Error scraping tweets for @{username}: {e}")
            return []

    def save_tweets(self, tweets):
        """Save tweets to CSV file"""
        try:
            file_exists = Path(self.output_file).exists()
            
            with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'username', 'tweet_id', 'created_at', 'text', 
                    'likes', 'retweets', 'replies', 'views', 'language'
                ])
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerows(tweets)
            
            logger.info(f"Successfully saved {len(tweets)} tweets to {self.output_file}")
        except Exception as e:
            logger.error(f"Error saving tweets to CSV: {e}")
            raise

    def get_last_processed_account(self):
        """Get the last processed account from progress file"""
        try:
            if Path(self.progress_file).exists():
                with open(self.progress_file, 'r') as f:
                    return f.read().strip()
            return None
        except Exception as e:
            logger.error(f"Error reading progress file: {e}")
            return None

    def save_progress(self, username):
        """Save the last processed account to progress file"""
        try:
            with open(self.progress_file, 'w') as f:
                f.write(username)
        except Exception as e:
            logger.error(f"Error saving progress: {e}")

    async def run(self):
        """Main execution function with resume capability"""
        try:
            # Setup API with loaded credentials
            await self.setup()
            
            # Load accounts to scrape from CSV
            self.load_accounts('scraper/account.csv')
            
            # Get last processed account
            last_processed = self.get_last_processed_account()
            resume_index = 0
            
            if last_processed:
                try:
                    resume_index = self.accounts.index(last_processed) + 1
                    logger.info(f"Resuming from account after @{last_processed}")
                except ValueError:
                    resume_index = 0
                    logger.info("Starting from beginning as last processed account not found")
            
            # Process each account from the resume point
            for username in self.accounts[resume_index:]:
                try:
                    logger.info(f"Starting to scrape tweets for @{username}")
                    tweets = await self.scrape_user_tweets(username)
                    if tweets:
                        self.save_tweets(tweets)
                    self.save_progress(username)  # Save progress after successful processing
                    await asyncio.sleep(2)  # Rate limiting pause
                except Exception as e:
                    logger.error(f"Error processing @{username}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Fatal error in main execution: {e}")
            raise

async def main():
    scraper = TwitterScraper()
    await scraper.run()

if __name__ == "__main__":
    # Set twscrape log level
    set_log_level("INFO")
    
    # Run the scraper
    asyncio.run(main())