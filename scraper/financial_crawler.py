import os
import yaml
import logging
import asyncio
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from crawl4ai.async_webcrawler import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from validators.data_validation import (
    create_validator,
    validate_data_point,
    QualityMetricsCalculator
)

class FinancialCrawler:
    """Financial data crawler for collecting KOL content from multiple sources."""
    
    def __init__(self, config_path: str):
        """
        Initialize the crawler with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_logging()
        self.validators = self._initialize_validators()
        self.metrics_calculator = QualityMetricsCalculator()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load crawler configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def setup_logging(self):
        """Configure logging based on settings."""
        log_config = self.config['logging']
        log_dir = Path(log_config['directory'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"crawler_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_config['level']),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _initialize_validators(self) -> List[Any]:
        """Initialize data validators from configuration."""
        validation_cfg = self.config['validation']
        validators = [
            create_validator('content_length', validation_cfg['content']),
            create_validator('required_fields', {'fields': validation_cfg['required_fields']}),
            create_validator('author_verification', validation_cfg['author_metrics'])
        ]
        return validators
        
    def _prepare_storage(self, source: str) -> Path:
        """Prepare storage directory for crawled data."""
        output_config = self.config['output']
        base_dir = Path(output_config['directory'])
        
        if output_config['options']['partition_by'] == 'source':
            data_dir = base_dir / source
        else:
            data_dir = base_dir
            
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
        
    def _save_data(self, data: List[Dict[str, Any]], source: str):
        """Save crawled data to appropriate storage location."""
        if not data:
            self.logger.warning(f"No data to save for source: {source}")
            return
            
        data_dir = self._prepare_storage(source)
        date_str = datetime.now().strftime(self.config['output']['options']['date_format'])
        output_file = data_dir / f"{source}_{date_str}.csv"
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        self.logger.info(f"Saved {len(data)} records to {output_file}")
        
        # Calculate quality metrics
        completeness = self.metrics_calculator.calculate_completeness(df)
        uniqueness = self.metrics_calculator.calculate_uniqueness(
            df, self.config['quality_metrics'][1]['fields']
        )
        
        self.logger.info(f"Data quality metrics for {source}:")
        self.logger.info(f"Completeness: {completeness:.2f}")
        self.logger.info(f"Uniqueness: {uniqueness}")
        
    async def crawl_source(self, source_name: str, source_config: Dict[str, Any]):
        """Crawl a specific source using crawl4ai."""
        self.logger.info(f"Starting crawl for source: {source_name}")
        
        # Setup configurations
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
            verbose=True,
            delay_before_return_html=source_config['rate_limits']['pause_between_requests'],
            mean_delay=60.0 / source_config['rate_limits']['requests_per_minute']
        )
        
        # Create and run crawler
        collected_data = []
        try:
            crawler = AsyncWebCrawler(config=browser_config)
            async with crawler as crawler_session:
                # Set up auth hooks if credentials are present
                if auth_config:
                    async def auth_hook(page):
                        await page.type('input[name="username"]', username)
                        await page.type('input[name="password"]', password)
                        await page.click('button[type="submit"]')
                    crawler_session.before_return_html = auth_hook
                # Process each URL
                for url in source_config['url_patterns']:
                    try:
                        result = await crawler_session.arun(url=url, config=crawler_config)
                        if result and hasattr(result, 'data'):
                            # Process and validate data
                            for item in result.data:
                                if validate_data_point(item, self.validators):
                                    collected_data.append(item)
                                    
                                    # Save in batches of 1000
                                    if len(collected_data) >= 1000:
                                        self._save_data(collected_data, source_name)
                                        collected_data = []
                    except Exception as e:
                        self.logger.error(f"Error crawling {url}: {str(e)}")
                        continue
                        
                # Save any remaining data
                if collected_data:
                    self._save_data(collected_data, source_name)
                    
        except Exception as e:
            self.logger.error(f"Error during crawling: {str(e)}")
            if collected_data:
                self.logger.info("Saving partial data collected before error")
                self._save_data(collected_data, source_name)
            raise
            
    async def run(self):
        """Run the crawler for all configured sources."""
        self.logger.info("Starting financial data crawler")
        
        for source_name, source_config in self.config['sources'].items():
            try:
                await self.crawl_source(source_name, source_config)
            except Exception as e:
                self.logger.error(f"Error crawling {source_name}: {str(e)}")
                
        self.logger.info("Crawling completed")

if __name__ == "__main__":
    config_path = "crawler_config/config.yaml"
    crawler = FinancialCrawler(config_path)
    
    try:
        asyncio.run(crawler.run())
    except KeyboardInterrupt:
        print("\nCrawling interrupted by user")
    except Exception as e:
        print(f"Error: {str(e)}")
