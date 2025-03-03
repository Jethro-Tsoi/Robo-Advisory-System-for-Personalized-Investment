import os
import logging
import asyncio
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from financial_crawler import FinancialCrawler

def setup_logging():
    """Configure test logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def verify_env_variables():
    """Check if required environment variables are set."""
    load_dotenv()
    required_vars = ['TWITTER_USER', 'TWITTER_PASS', 'SA_USER', 'SA_PASS']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

async def test_single_source():
    """Test crawling a single source to verify setup."""
    config_path = "crawler_config/config.yaml"
    crawler = FinancialCrawler(config_path)
    
    # Test with SeekingAlpha first as it's typically more stable
    source_name = "seekingalpha"
    source_config = crawler.config['sources'][source_name]
    
    try:
        # Setup browser and crawler configurations
        browser_config = BrowserConfig(
            browser_type="chromium",
            headless=True
        )
        
        auth_config = source_config.get('auth', {})
        if auth_config:
            username = os.getenv(auth_config['username'].strip('${}'))
            password = os.getenv(auth_config['password'].strip('${}'))
            
        # Create CSS selector string from source config
        css_selectors = []
        for name, selector in source_config['selectors'].items():
            css_selectors.append(selector)
            
        # Configure crawler
        crawler_config = CrawlerRunConfig(
            css_selector=", ".join(css_selectors),
            wait_until='networkidle',
            page_timeout=60000,
            verbose=True
        )
        
        # Initialize crawler with auth hooks if needed
        crawler_instance = AsyncWebCrawler(config=browser_config)
        logging.info("Crawler instance created successfully")
        
        # Test URL setup
        test_url = source_config['url_patterns'][0]
        
        logging.info(f"Testing crawl of {test_url}")
        async with crawler_instance as crawler_session:
            # Set up auth hooks if credentials are present
            if auth_config:
                async def auth_hook(page):
                    await page.type('input[name="username"]', username)
                    await page.type('input[name="password"]', password)
                    await page.click('button[type="submit"]')
                crawler_session.before_return_html = auth_hook
            
            results = await crawler_session.arun(url=test_url, config=crawler_config)
        
        if results:
            logging.info("Successfully retrieved data from source")
            
            # Get output directory from the FinancialCrawler config
            output_dir = Path(crawler.config['output']['directory'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save test results
            df = pd.DataFrame(results)
            test_output = output_dir / f"test_{source_name}.csv"
            df.to_csv(test_output, index=False)
            logging.info(f"Saved test results to {test_output}")
            
            # Print sample data
            logging.info("\nSample data:")
            for col in df.columns:
                logging.info(f"{col}: {df[col].iloc[0] if not df.empty else 'No data'}")
        else:
            logging.warning("No data retrieved from source")
            
    except Exception as e:
        logging.error(f"Error during test crawl: {str(e)}")
        raise

def verify_data_quality():
    """Verify the quality of collected data."""
    config_path = "crawler_config/config.yaml"
    crawler = FinancialCrawler(config_path)
    
    output_dir = Path(crawler.config['output']['directory'])
    data_files = list(output_dir.rglob("*.csv"))
    
    if not data_files:
        logging.warning("No data files found for quality verification")
        return
        
    for data_file in data_files:
        try:
            # Load the data file
            df = pd.read_csv(data_file)
            
            # Calculate quality metrics
            completeness = crawler.metrics_calculator.calculate_completeness(df)
            uniqueness = crawler.metrics_calculator.calculate_uniqueness(
                df, crawler.config['quality_metrics'][1]['fields']
            )
            
            logging.info(f"\nQuality metrics for {data_file.name}:")
            logging.info(f"Records count: {len(df)}")
            logging.info(f"Completeness: {completeness:.2f}")
            logging.info(f"Uniqueness: {uniqueness}")
            
            # Verify required fields
            required_fields = crawler.config['validation']['required_fields']
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                logging.warning(f"Missing required fields: {missing_fields}")
            
        except Exception as e:
            logging.error(f"Error analyzing {data_file}: {str(e)}")

async def main():
    """Run test suite."""
    setup_logging()
    logging.info("Starting crawler test suite")
    
    try:
        # Verify environment setup
        logging.info("\nVerifying environment variables...")
        verify_env_variables()
        
        # Test single source crawling
        logging.info("\nTesting single source crawl...")
        await test_single_source()
        
        # Verify data quality
        logging.info("\nVerifying data quality...")
        verify_data_quality()
        
        logging.info("\nTest suite completed successfully")
        
    except Exception as e:
        logging.error(f"Test suite failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)})")
        raise
