#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install polars requests tqdm')


# In[2]:


import os
import polars as pl
import requests
import json
from tqdm import tqdm
import time
import re
# Removed: # Removed: import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor


# In[3]:


class KeyManager:
    """Manages multiple API keys and rotates them when rate limits are reached"""

    def __init__(self, env_prefix='MISTRAL_API_KEY'):
        """
        Initialize the key manager

        Parameters:
        -----------
        env_prefix : str
            Prefix for environment variables storing API keys.
            Keys should be named like MISTRAL_API_KEY, MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, etc.
        """
        self.env_prefix = env_prefix
        self.api_keys = self._load_api_keys()
        self.current_index = 0
        self.rate_limited_keys = {}  # Track which keys hit rate limits and when they can be used again
        self.base_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json"
        }

        if not self.api_keys:
            print("⚠️ No API keys found. Please set at least one API key.")
            key = input(f"Enter your Mistral AI API key: ")
            if key:
                self.api_keys.append(key)
            else:
                raise ValueError("No API key provided. Cannot continue.")

        print(f"Loaded {len(self.api_keys)} Mistral API keys.")

    def _load_api_keys(self):
        """Load API keys from environment variables"""
        api_keys = []

        # Try the base key first
        base_key = os.getenv(self.env_prefix)
        if base_key:
            api_keys.append(base_key)

        # Try numbered keys (MISTRAL_API_KEY_1, MISTRAL_API_KEY_2, etc.)
        for i in range(1, 10):  # Check for up to 10 keys
            key = os.getenv(f"{self.env_prefix}_{i}")
            if key:
                api_keys.append(key)

        return api_keys

    def get_current_key(self):
        """Get the current active API key"""
        return self.api_keys[self.current_index]

    def get_current_headers(self):
        """Get headers with the current API key"""
        headers = self.headers.copy()
        headers["Authorization"] = f"Bearer {self.get_current_key()}"
        return headers

    def rotate_key(self, rate_limited=False, retry_after=60):
        """
        Rotate to the next available API key

        Parameters:
        -----------
        rate_limited : bool
            Whether the current key hit a rate limit
        retry_after : int
            Seconds until the rate-limited key can be used again

        Returns:
        --------
        str : The new API key
        """
        # Mark the current key as rate limited if needed
        if rate_limited:
            self.rate_limited_keys[self.current_index] = time.time() + retry_after
            print(f"API key {self.current_index + 1} rate limited. Will retry after {retry_after} seconds.")

        # Try to find a key that's not rate limited
        original_index = self.current_index
        while True:
            self.current_index = (self.current_index + 1) % len(self.api_keys)

            # Check if this key is rate limited
            if self.current_index in self.rate_limited_keys:
                # Check if enough time has passed
                if time.time() > self.rate_limited_keys[self.current_index]:
                    # Key is no longer rate limited
                    del self.rate_limited_keys[self.current_index]
                    break
            else:
                # Key is not rate limited
                break

            # If we've checked all keys and they're all rate limited, use the least recently rate limited one
            if self.current_index == original_index:
                # Find the key that will be available soonest
                soonest_available = min(self.rate_limited_keys.items(), key=lambda x: x[1])
                self.current_index = soonest_available[0]
                wait_time = max(0, self.rate_limited_keys[self.current_index] - time.time())

                if wait_time > 0:
                    print(f"All API keys are rate limited. Waiting {wait_time:.1f} seconds for the next available key.")
                    time.sleep(wait_time)
                    del self.rate_limited_keys[self.current_index]
                break

        # Return the new key
        key = self.api_keys[self.current_index]
        print(f"Switched to Mistral API key {self.current_index + 1}")
        return key


# In[4]:


# Initialize the Key Manager
key_manager = KeyManager()

# Set the model name
MODEL = "mistral-small-latest"  # Using Mistral's large model instead of Gemini


# 
# > **⚠️ API Key Setup**
# >
# > This notebook uses the Mistral AI API with support for multiple API keys:
# >
# > 1. Set your primary API key as `MISTRAL_API_KEY` environment variable
# > 2. For additional keys, use `MISTRAL_API_KEY_1`, `MISTRAL_API_KEY_2`, etc.
# > 3. The system will automatically rotate between keys if rate limits are encountered
# >
# > Keys can be created at [Mistral AI Platform](https://console.mistral.ai/)
# >
# > **Mistral AI Free Tier Limits:**
# > - 1 request per second (60 requests per minute)
# > - 500,000 tokens per minute
# > - 1 billion tokens per month
# 

# In[5]:


def setup_prompt(text):
    """Configure the prompt for Mistral"""
    return [
        {"role": "system", "content": """
            You are a financial sentiment analyzer. Classify the given tweet's sentiment into one of these categories:

            STRONGLY_POSITIVE - Very bullish, highly confident optimistic outlook
            POSITIVE - Generally optimistic, bullish view
            NEUTRAL - Factual, balanced, or no clear sentiment
            NEGATIVE - Generally pessimistic, bearish view
            STRONGLY_NEGATIVE - Very bearish, highly confident pessimistic outlook

            Examples:
            "Breaking: Company XYZ doubles profit forecast!" -> STRONGLY_POSITIVE
            "Expecting modest gains next quarter" -> POSITIVE
            "Market closed at 35,000" -> NEUTRAL
            "Concerned about rising rates" -> NEGATIVE
            "Crash incoming, sell everything!" -> STRONGLY_NEGATIVE

            Format: Return only one word from: STRONGLY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, STRONGLY_NEGATIVE
        """},
        {"role": "user", "content": f"Analyze the sentiment of this tweet: {text}"}
    ]


# In[6]:


def get_sentiment(text, retries=3):
    """Get sentiment from Mistral AI with retry logic"""
    if not text or len(str(text).strip()) < 3:
        return 'NEUTRAL'

    for attempt in range(retries):
        try:
            # Setup the API request for Mistral
            headers = key_manager.get_current_headers()
            payload = {
                "model": MODEL,
                "temperature": 0.0,  # Deterministic output
                "max_tokens": 10,    # We only need one word
                "messages": setup_prompt(text)
            }

            # Make the API request
            response = requests.post(
                key_manager.base_url,
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                # Extract sentiment from Mistral's response
                response_json = response.json()
                sentiment = response_json['choices'][0]['message']['content'].strip().upper()

                # Validate the response
                valid_labels = [
                    'STRONGLY_POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'STRONGLY_NEGATIVE'
                ]

                if sentiment in valid_labels:
                    return sentiment
                else:
                    print(f"Invalid sentiment received: {sentiment}, defaulting to NEUTRAL")
                    return 'NEUTRAL'
            elif response.status_code == 429:  # Rate limit
                # Get retry_after time if provided
                retry_after = int(response.headers.get('Retry-After', 5))
                key_manager.rotate_key(rate_limited=True, retry_after=retry_after)
                if attempt < retries - 1:
                    continue
                else:
                    return 'NEUTRAL'
            else:
                print(f"API error: {response.status_code} - {response.text}")
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                else:
                    return 'NEUTRAL'

        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limiting errors
            if "quota" in error_str or "rate" in error_str or "429" in error_str:
                # Extract retry time if available (default to 5 seconds if not found)
                retry_after = 0
                if "retryafter" in error_str or "retry-after" in error_str or "retry_after" in error_str:
                    try:
                        # Try to extract the retry time
                        matches = re.findall(r'retry.*?(\\d+)', error_str)
                        if matches:
                            retry_after = int(matches[0])
                    except:
                        pass

                # Switch to another API key if there are multiple keys
                if len(key_manager.api_keys) > 1:
                    key_manager.rotate_key(rate_limited=True, retry_after=retry_after)
                    if attempt < retries - 1:
                        continue
                else:
                    # Only one key, just wait
                    wait_time = min(2 ** attempt * 5, retry_after)  # Exponential backoff with max retry_after
                    print(f"Rate limit hit - waiting {wait_time}s before retry ({attempt+1}/{retries})")
                    time.sleep(wait_time)
                    if attempt < retries - 1:
                        continue

            if attempt == retries - 1:
                print(f"Error processing text: {str(text)[:50]}...\nError: {str(e)}")
                return 'NEUTRAL'
            time.sleep(2)  # Wait before retry

    return 'NEUTRAL'


# In[7]:


# Test the sentiment analysis with key rotation
test_tweet = "Breaking: Tesla stock hits all-time high after unexpected profit surge"
sentiment = get_sentiment(test_tweet)
print(f"Test tweet: '{test_tweet}'")
print(f"Sentiment: {sentiment}")
print(f"Using Mistral API key index: {key_manager.current_index}")


# In[8]:


# Load data from Hugging Face
print("Loading stock market tweets dataset from Hugging Face...")

# Check if the huggingface datasets library is installed
try:
    import huggingface_hub
except ImportError:
    print("Installing huggingface_hub...")
    # get_ipython().system('pip install huggingface_hub')
    import huggingface_hub

# Load the dataset using Polars
df = pl.read_csv('hf://datasets/StephanAkkerman/financial-tweets-stocks/stock.csv')


print(f"Loaded {df.shape[0]} tweets")
print("\nSample tweets:")
df.head(5)


# In[9]:


# Prepare the dataset for sentiment analysis
# Let's make sure we have the 'body' column which contains the tweet text
if 'description' in df.columns:
    tweet_column = 'description'
# elif 'full_text' in df.columns:
#     tweet_column = 'full_text'
else:
    raise ValueError("Could not find tweet text column in the dataset")

print(f"Using '{tweet_column}' column for tweet text")

# For demonstration, let's use a small subset of the data
# Use all data instead of a sample - WARNING: This will process 1.7M tweets!
sample_size = df.shape[0]  # Adjust based on your needs
sample_df = df.sample(sample_size, seed=42)

print(f"\nAnalyzing sentiment for {sample_size} tweets")


# 
# ## Rate Limits for Mistral AI API
# 
# Mistral AI's free tier has the following limits:
# - 1 request per second (60 requests per minute)
# - 500,000 tokens per minute
# - 1 billion tokens per month
# 
# This notebook implements:
# 1. Key rotation to handle multiple API keys
# 2. Automatic retry with exponential backoff
# 3. Batch processing to optimize throughput
# 4. Error handling to ensure robust processing
# 
# If you need to process many tweets, consider:
# - Creating multiple API keys
# - Adjusting batch size and workers based on your needs
# - Processing tweets in smaller batches with appropriate delays
# 

# In[10]:


def process_tweets(tweets, batch_size=4, max_workers=2):
    """Process tweets in batches with Mistral AI API

    Mistral AI free tier allows:
    - 1 request per second (60 requests per minute)
    - 500,000 tokens per minute
    - 1 billion tokens per month
    """
    all_sentiments = []

    # For demonstration, we'll process a reasonable number of tweets
    # Adjust max_tweets if needed - 200 is safe for the free tier
    max_tweets = min(200, len(tweets))
    tweets = tweets[:max_tweets]

    print(f"Processing {max_tweets} tweets using Mistral AI API with {len(key_manager.api_keys)} API keys")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(tweets), batch_size), desc="Processing tweet batches"):
            batch = tweets[i:i+batch_size]

            try:
                # Process tweets in parallel
                results = list(executor.map(get_sentiment, batch))
                all_sentiments.extend(results)

                # Add a short delay between batches to avoid rate limiting
                time.sleep(2)

            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add neutral sentiments for this batch in case of failure
                all_sentiments.extend(['NEUTRAL'] * len(batch))
                time.sleep(5)  # Wait a bit longer after an error

    # If we didn't get enough sentiments (due to errors), fill with NEUTRAL
    if len(all_sentiments) < len(tweets):
        all_sentiments.extend(['NEUTRAL'] * (len(tweets) - len(all_sentiments)))

    return all_sentiments


# In[11]:


# Test the sentiment analysis with key rotation
test_tweet = "Breaking: Tesla stock hits all-time high after unexpected profit surge"
sentiment = get_sentiment(test_tweet)
print(f"Test tweet: '{test_tweet}'")
print(f"Sentiment: {sentiment}")
print(f"Using API key index: {key_manager.current_index}")


# In[12]:


def process_tweets(tweets, batch_size=4, max_workers=2):
    """Process tweets in batches with Mistral AI API

    Mistral AI free tier allows:
    - 1 request per second (60 requests per minute)
    - 500,000 tokens per minute
    - 1 billion tokens per month
    """
    all_sentiments = []

    # Process all tweets
    # Removed limitation: max_tweets = min(200, len(tweets))
    # Removed limitation: tweets = tweets[:max_tweets]

    print(f"Processing {len(tweets)} tweets using Mistral AI API with {len(key_manager.api_keys)} API keys")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in tqdm(range(0, len(tweets), batch_size), desc="Processing tweet batches"):
            batch = tweets[i:i+batch_size]

            try:
                # Process tweets in parallel
                results = list(executor.map(get_sentiment, batch))
                all_sentiments.extend(results)

                # Add a delay between batches to respect Mistral's 1 req/sec rate limit
                # With batch_size and max_workers, adjust sleep time accordingly
                time.sleep(batch_size * max_workers)  # Conservative approach

            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add neutral sentiments for this batch in case of failure
                all_sentiments.extend(['NEUTRAL'] * len(batch))
                time.sleep(5)  # Wait a bit longer after an error

    # If we didn't get enough sentiments (due to errors), fill with NEUTRAL
    if len(all_sentiments) < len(tweets):
        all_sentiments.extend(['NEUTRAL'] * (len(tweets) - len(all_sentiments)))

    return all_sentiments


# In[13]:


def process_tweets_concurrent_resumed(tweets, start_batch=0, batch_size=4, max_workers=2, save_every=100, rate_limit_wait=5, force_reprocess=False):
    """Process tweets in batches with Mistral AI API using concurrent API keys, resuming from a specific batch

    Parameters:
    -----------
    tweets : list
        List of tweets to process
    start_batch : int
        Batch index to start processing from (for resuming interrupted processing)
    batch_size : int
        Number of tweets to process in each batch
    max_workers : int
        Number of concurrent workers
    save_every : int
        Save results after processing this many tweets
    rate_limit_wait : int
        Default wait time in seconds when rate limited (default: 5)
    force_reprocess : bool
        If True, will reprocess tweets even if they already have sentiment values

    Returns:
    --------
    list : Sentiment labels for each tweet
    """
    # Initialize the global key manager if it doesn't exist
    global key_manager
    if 'key_manager' not in globals():
        key_manager = KeyManager()

    # Set model name if not defined
    global MODEL
    if 'MODEL' not in globals():
        MODEL = "mistral-small-latest"

    # Load partial results if available
    save_path = "../data/labeled_stock_tweets_partial.csv"
    try:
        partial_df = pl.read_csv(save_path)
        # Extract existing sentiments from the partial results
        all_sentiments = partial_df['sentiment'].to_list()
        print(f"Loaded {len(all_sentiments)} existing sentiment labels")
    except:
        # If no partial results exist, pre-allocate the array with None values
        all_sentiments = [None] * len(tweets)
        print("No existing results found, starting fresh")

    # Sort tweet indices in descending order (process larger row numbers first)
    tweet_indices = list(range(len(tweets)))
    tweet_indices.sort(reverse=True)

    # Create a KeyManager for each worker
    key_managers = []

    # Check how many API keys we have available
    available_keys = key_manager.api_keys.copy()
    num_keys = len(available_keys)

    print(f"Using {num_keys} API keys concurrently for processing {len(tweets)} tweets")
    print(f"Resuming from batch {start_batch}")

    if num_keys == 0:
        raise ValueError("No API keys available")

    # Create a manager for each key
    for i, key in enumerate(available_keys):
        # Create a separate manager for each key
        km = KeyManager()
        # Replace the API keys with just one key
        km.api_keys = [key]
        km.current_index = 0
        key_managers.append(km)

    # If force_reprocess is True, clear the sentiments for batches we want to reprocess
    if force_reprocess and start_batch > 0:
        # Create batches first to identify which indices to clear
        temp_batches = []
        for i in range(0, len(tweet_indices), batch_size):
            batch_indices = tweet_indices[i:i+batch_size]
            temp_batches.append(batch_indices)

        # Clear sentiments for the batches we want to reprocess
        for batch_idx in range(start_batch, len(temp_batches)):
            for idx in temp_batches[batch_idx]:
                if idx < len(all_sentiments):
                    all_sentiments[idx] = None

        print(f"Cleared sentiments for batches starting from {start_batch} to force reprocessing")

    # Function to process a tweet with a specific key manager
    def process_tweet_with_key(args):
        idx, tweet, key_idx = args
        # Skip already processed tweets (with non-None sentiments)
        if not force_reprocess and idx < len(all_sentiments) and all_sentiments[idx] is not None:
            return (idx, all_sentiments[idx])

        # Get the key manager for this worker
        km = key_managers[key_idx % num_keys]

        # Define a local get_sentiment function that uses this specific key manager
        def local_get_sentiment(text, retries=3):
            if not text or len(str(text).strip()) < 3:
                return 'NEUTRAL'

            for attempt in range(retries):
                try:
                    # Setup the API request for Mistral
                    headers = km.get_current_headers()
                    payload = {
                        "model": MODEL,
                        "temperature": 0.0,  # Deterministic output
                        "max_tokens": 10,    # We only need one word
                        "messages": setup_prompt(text)
                    }

                    # Make the API request
                    response = requests.post(
                        km.base_url,
                        headers=headers,
                        json=payload
                    )

                    if response.status_code == 200:
                        # Extract sentiment from Mistral's response
                        response_json = response.json()
                        sentiment = response_json['choices'][0]['message']['content'].strip().upper()

                        # Validate the response
                        valid_labels = [
                            'STRONGLY_POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'STRONGLY_NEGATIVE'
                        ]

                        if sentiment in valid_labels:
                            return sentiment
                        else:
                            print(f"Invalid sentiment received: {sentiment}, defaulting to NEUTRAL")
                            return 'NEUTRAL'
                    elif response.status_code == 429:  # Rate limit
                        # If rate limited, just wait instead of switching keys
                        # Always use rate_limit_wait as fallback
                        retry_after = int(response.headers.get('Retry-After', rate_limit_wait))
                        print(f"API key {key_idx+1} rate limited. Waiting {retry_after} seconds.")
                        time.sleep(retry_after)
                        if attempt < retries - 1:
                            continue
                        else:
                            return 'NEUTRAL'
                    else:
                        print(f"API error: {response.status_code} - {response.text}")
                        if attempt < retries - 1:
                            time.sleep(2)  # Wait before retry
                            continue
                        else:
                            return 'NEUTRAL'

                except Exception as e:
                    error_str = str(e).lower()
                    # Check for rate limiting errors
                    if "quota" in error_str or "rate" in error_str or "429" in error_str:
                        # Extract retry time if available (default to rate_limit_wait if not found)
                        retry_after = rate_limit_wait
                        if "retryafter" in error_str or "retry-after" in error_str or "retry_after" in error_str:
                            try:
                                # Try to extract the retry time
                                matches = re.findall(r'retry.*?(\d+)', error_str)
                                if matches:
                                    retry_after = int(matches[0])
                            except:
                                pass

                        # Just wait instead of switching keys
                        wait_time = min(2 ** attempt * 2, retry_after)  # Use smaller exponential backoff
                        print(f"Rate limit hit for key {key_idx+1} - waiting {wait_time}s before retry ({attempt+1}/{retries})")
                        time.sleep(wait_time)
                        if attempt < retries - 1:
                            continue

                    if attempt == retries - 1:
                        print(f"Error processing text: {str(text)[:50]}...\nError: {str(e)}")
                        return 'NEUTRAL'
                    time.sleep(2)  # Wait before retry

            return 'NEUTRAL'

        # Process the tweet
        try:
            result = local_get_sentiment(tweet)
            return (idx, result)
        except Exception as e:
            print(f"Error processing tweet {idx}: {e}")
            return (idx, 'NEUTRAL')

    # Create batches for processing
    batches = []
    for i in range(0, len(tweet_indices), batch_size):
        batch_indices = tweet_indices[i:i+batch_size]
        # More balanced key assignment - round robin style
        batch = [(idx, tweets[idx], (i + idx) % num_keys) for idx in batch_indices]
        batches.append(batch)

    # Process batches and periodically save results
    # Calculate the processed count based on non-None values in all_sentiments
    processed_count = sum(1 for s in all_sentiments if s is not None)
    print(f"Found {processed_count} previously processed tweets")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Skip to the starting batch
    batches = batches[start_batch:]
    print(f"Skipping {start_batch} batches, {start_batch * batch_size} tweets")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_idx, batch in enumerate(tqdm(batches, desc=f"Processing tweet batches (starting from batch {start_batch})")):
            try:
                # Process batch
                results = list(executor.map(process_tweet_with_key, batch))

                # Update results
                for idx, sentiment in results:
                    # Only update if it wasn't already set (avoiding repeat API calls)
                    if idx < len(all_sentiments) and (all_sentiments[idx] is None or force_reprocess):
                        all_sentiments[idx] = sentiment
                        processed_count += 1

                # Periodically save results
                if (processed_count % save_every < batch_size) or (batch_idx % 50 == 0 and batch_idx > 0):
                    # Create a temporary dataframe with current results
                    try:
                        # Try to load the original dataframe
                        global df
                        if 'df' not in globals():
                            # If df is not defined globally, create a stub
                            # In actual use, you would need to make sure df is defined
                            # before calling this function
                            raise NameError("df not defined")

                        temp_df = df.clone()
                        # Only include processed tweets (non-None sentiments)
                        valid_sentiments = [s if s is not None else 'NEUTRAL' for s in all_sentiments]
                        temp_df = temp_df.with_columns(pl.Series(name='sentiment', values=valid_sentiments))
                        # Save to CSV
                        temp_df.write_csv(save_path)
                        print(f"\nSaved {processed_count} processed tweets to {save_path}")
                    except Exception as e:
                        print(f"Error saving results: {e}")

                    # Save progress information to a JSON file
                    import json
                    progress = {
                        "current_batch": start_batch + batch_idx + 1,
                        "processed_count": processed_count,
                        "total_batches": len(batches) + start_batch,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    with open("../data/labeled_stock_tweets_progress.json", "w") as f:
                        json.dump(progress, f)

                # Add a short delay between batches to avoid API overload
                # Uncomment if needed
                # time.sleep(0.5)

            except Exception as e:
                print(f"Error processing batch: {e}")
                # Continue with the next batch
                time.sleep(5)  # Wait a bit longer after an error

    # Replace any None values with NEUTRAL
    all_sentiments = [s if s is not None else 'NEUTRAL' for s in all_sentiments]

    return all_sentiments


# In[ ]:


import os

# Process tweets using the resumed concurrent approach
tweets = sample_df[tweet_column].to_list()

# Define parameters
BATCH_SIZE = 4  # Number of tweets to process in each batch
MAX_WORKERS = 2  # Number of concurrent workers (match to number of API keys)
SAVE_EVERY = 100  # Save results every N tweets processed
RATE_LIMIT_WAIT = 0  # Default wait time in seconds when rate limited

# Change this to the batch you want to resume from
resume_batch = int(13600 / BATCH_SIZE)

# Process tweets using concurrent API keys with resuming capability
sentiments = process_tweets_concurrent_resumed(
    tweets, 
    start_batch=resume_batch,
    batch_size=BATCH_SIZE, 
    max_workers=MAX_WORKERS,
    save_every=SAVE_EVERY,
    rate_limit_wait=RATE_LIMIT_WAIT,
    force_reprocess=True
)

# Add sentiments to the DataFrame
sample_df = sample_df.with_columns(pl.Series(name='sentiment', values=sentiments))

# Display the results
sample_df.select([tweet_column, 'sentiment']).head(10)


# In[15]:


# Save the labeled data
output_path = "../data/labeled_stock_tweets.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
sample_df.write_csv(output_path)
print(f"\nSaved labeled data to {output_path}")

# Display some examples of each sentiment
print("\nExamples for each sentiment:")
for sentiment in ['STRONGLY_POSITIVE', 'POSITIVE', 'NEUTRAL', 'NEGATIVE', 'STRONGLY_NEGATIVE']:
    examples = sample_df.filter(pl.col("sentiment") == sentiment).sample(1, seed=42)
    if examples.shape[0] > 0:
        print(f"\n{sentiment}:\n{examples[0, tweet_column]}")

