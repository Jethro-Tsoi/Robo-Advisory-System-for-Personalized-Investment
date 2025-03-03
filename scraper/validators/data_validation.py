import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class DataValidator:
    """Base validator class for financial data quality checks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def validate(self, data: Dict[str, Any]) -> bool:
        """
        Base validation method to be implemented by specific validators.
        
        Args:
            data: Dictionary containing the data to validate
            
        Returns:
            bool: True if validation passes, False otherwise
        """
        raise NotImplementedError

class ContentLengthValidator(DataValidator):
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate content length is within acceptable range."""
        content = data.get('content', '')
        if not content:
            logger.warning("Empty content found")
            return False
            
        content_length = len(content)
        min_length = self.config.get('min_length', 50)
        max_length = self.config.get('max_length', 50000)
        
        if min_length <= content_length <= max_length:
            return True
        
        logger.warning(
            f"Content length {content_length} outside acceptable range "
            f"({min_length}-{max_length})"
        )
        return False

class RequiredFieldsValidator(DataValidator):
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate all required fields are present and non-empty."""
        required_fields = self.config.get('fields', [])
        
        for field in required_fields:
            if field not in data or not data[field]:
                logger.warning(f"Missing required field: {field}")
                return False
        return True

class AuthorVerificationValidator(DataValidator):
    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate author meets minimum criteria."""
        author_metrics = data.get('author_metrics', {})
        min_followers = self.config.get('min_followers', 1000)
        min_posts = self.config.get('min_posts', 50)
        
        if not author_metrics:
            logger.warning("Missing author metrics")
            return False
            
        followers = author_metrics.get('followers', 0)
        posts = author_metrics.get('posts', 0)
        
        if followers < min_followers:
            logger.warning(
                f"Author followers ({followers}) below minimum ({min_followers})"
            )
            return False
            
        if posts < min_posts:
            logger.warning(
                f"Author posts ({posts}) below minimum ({min_posts})"
            )
            return False
            
        return True

class QualityMetricsCalculator:
    """Calculate various quality metrics for the dataset."""
    
    @staticmethod
    def calculate_completeness(df: pd.DataFrame) -> float:
        """
        Calculate the completeness ratio of the dataset.
        
        Args:
            df: DataFrame containing the data
            
        Returns:
            float: Completeness score between 0 and 1
        """
        total_cells = df.size
        filled_cells = total_cells - df.isna().sum().sum()
        return filled_cells / total_cells

    @staticmethod
    def calculate_uniqueness(df: pd.DataFrame, columns: List[str]) -> Dict[str, float]:
        """
        Calculate uniqueness ratio for specified columns.
        
        Args:
            df: DataFrame containing the data
            columns: List of column names to check for uniqueness
            
        Returns:
            Dict[str, float]: Uniqueness scores for each column
        """
        results = {}
        for col in columns:
            if col in df.columns:
                total = len(df[col])
                unique = len(df[col].unique())
                results[col] = unique / total if total > 0 else 0
        return results

    @staticmethod
    def calculate_author_reputation(
        df: pd.DataFrame,
        engagement_cols: Optional[List[str]] = None
    ) -> pd.Series:
        """
        Calculate author reputation scores based on engagement metrics.
        
        Args:
            df: DataFrame containing the data
            engagement_cols: List of engagement metric column names
            
        Returns:
            pd.Series: Reputation scores for each author
        """
        if not engagement_cols:
            engagement_cols = ['likes', 'retweets', 'comments']
            
        # Filter to only existing columns
        available_cols = [col for col in engagement_cols if col in df.columns]
        
        if not available_cols:
            logger.warning("No engagement metrics available for reputation calculation")
            return pd.Series(0.5, index=df.index)
            
        # Normalize and combine engagement metrics
        normalized = df[available_cols].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        return normalized.mean(axis=1)

def create_validator(validator_type: str, config: Dict[str, Any]) -> DataValidator:
    """
    Factory function to create appropriate validator instance.
    
    Args:
        validator_type: Type of validator to create
        config: Configuration dictionary for the validator
        
    Returns:
        DataValidator: Instantiated validator object
    """
    validators = {
        'content_length': ContentLengthValidator,
        'required_fields': RequiredFieldsValidator,
        'author_verification': AuthorVerificationValidator
    }
    
    validator_class = validators.get(validator_type)
    if not validator_class:
        raise ValueError(f"Unknown validator type: {validator_type}")
        
    return validator_class(config)

def validate_data_point(
    data: Dict[str, Any],
    validators: List[DataValidator]
) -> bool:
    """
    Validate a single data point against all validators.
    
    Args:
        data: Dictionary containing the data to validate
        validators: List of validator instances to use
        
    Returns:
        bool: True if all validations pass, False otherwise
    """
    for validator in validators:
        if not validator.validate(data):
            return False
    return True
