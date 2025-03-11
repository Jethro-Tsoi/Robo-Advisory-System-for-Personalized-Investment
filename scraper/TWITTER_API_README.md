# X API v2 Usage Guide (Free Tier Optimized)

## Overview

This document explains how the `twitter_api.py` script interacts with X API v2 (formerly Twitter API) to collect tweets from specified accounts, with a focus on the free tier limitations and the 3,200 tweet limit.

## X API v2 Limitations

### Free Tier Constraints

The X API v2 free tier has significant limitations that our implementation carefully manages:

```
┌─────────────────┬───────────────────────────┬────────────────────┐
│   Constraint    │         Free Tier         │   Premium Tiers    │
├─────────────────┼───────────────────────────┼────────────────────┤
│ Monthly Reads   │ 100 reads per month       │ 10,000+ per month  │
│ Monthly Posts   │ 500 posts per month       │ 3,000+ per month   │
│ Rate Limits     │ 1 request per 15 minutes  │ 5-900 per 15 min   │
└─────────────────┴───────────────────────────┴────────────────────┘
```

### Tweet History Access Limits

The X API v2 has a hard limit of **3,200 most recent tweets** per user that can be accessed through the standard API. This is a platform-imposed limitation, not a restriction in our code:

```
┌─────────────────┬───────────────────┬────────────────────┐
│   API Access    │  Tweets Available │    Time Period     │
├─────────────────┼───────────────────┼────────────────────┤
│ Standard Access │ 3,200 most recent │    Most recent     │
│ Premium Access  │   Full archive    │ Complete history   │
└─────────────────┴───────────────────┴────────────────────┘
```

### API Rate Limits (Free Tier)

The X API v2 free tier imposes strict rate limits on how many requests can be made within a time window:

```
┌─────────────────┬───────────────────────────┐
│    Endpoint     │     Free Tier Limit       │
├─────────────────┼───────────────────────────┤
│ User Timeline   │ 1 request per 15 min      │
│ User Lookup     │ 1 request per 15 min      │
└─────────────────┴───────────────────────────┘
```

## Implementation Details

Our code implements the following features to work optimally within these constraints:

### 1. Monthly Quota Management

- Tracks the 100 reads per month quota
- Distributes available quota across accounts
- Automatically resets quota tracking on the 1st of each month
- Prevents exceeding the monthly quota

### 2. Smart Pagination

- Each request fetches up to 100 tweets (maximum allowed by API)
- Optimizes the number of tweets requested based on available quota
- Uses pagination tokens to efficiently retrieve tweets

### 3. Rate Limit Management

- Strict adherence to the 1 request per 15 minutes limit
- Automatic waiting when rate limits are reached
- Efficient use of available quota

### 4. Progress Tracking

- Reports current progress toward monthly quota
- Clear indication when API limits are reached
- Efficient batch saving to prevent data loss

## Optimized for Free Tier

Our implementation is specifically optimized for the free tier constraints:

1. **Quota Distribution**: Automatically calculates how many tweets to fetch per account based on the number of accounts and remaining monthly quota
2. **Conservative Pacing**: Adds small delays between requests to avoid rate limit issues
3. **Incremental Collection**: Stores the last tweet ID to efficiently fetch only new tweets in subsequent runs
4. **Batch Saving**: Saves tweets in small batches (50 tweets) to prevent data loss if errors occur

## Usage Example

```python
# Inside twitter_api.py
# The fetch_tweets function automatically handles pagination and respects free tier limits
tweets = fetch_tweets('username', max_tweets=100)  # Limit to 100 tweets for free tier

# You can also check API limits programmatically
limits_info = get_api_limits_info()
print(limits_info)
```

## Best Practices for Free Tier

1. **Run the collector regularly** to capture tweets before they fall outside the 3,200 tweet window
2. **Limit the number of accounts** to ensure each account gets a reasonable allocation of the monthly quota
3. **Focus on high-value accounts** that provide the most relevant data for your analysis
4. **Schedule collection strategically** to maximize the value from the limited quota
5. **Consider upgrading to Basic tier** ($200/month) if you need more comprehensive data collection

## Additional Resources

- [X Developer Portal](https://developer.twitter.com/en/docs/twitter-api)
- [X API v2 Documentation](https://developer.twitter.com/en/docs/twitter-api/tweets/timelines/introduction)
- [Tweepy Documentation](https://docs.tweepy.org/en/stable/)
