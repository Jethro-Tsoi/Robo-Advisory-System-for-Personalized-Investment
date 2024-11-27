from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import glob

def load_csv_data(file_path: str, columns: List[str]) -> Optional[pd.DataFrame]:
    """
    Load specific columns from CSV file
    
    Args:
        file_path: Path to the CSV file
        columns: List of column names to load
        
    Returns:
        DataFrame with specified columns or None if error occurs
    """
    try:
        df = pd.read_csv(file_path, usecols=columns)
        print(f"Successfully loaded CSV {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading CSV {file_path}: {str(e)}")
        return None

def load_all_csvs(data_dir: str, columns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV files from a directory with specific columns
    
    Args:
        data_dir: Directory containing CSV files
        columns: List of column names to load
        
    Returns:
        Dictionary mapping filenames to DataFrames
    """
    csv_files = glob.glob(str(Path(data_dir) / "*.csv"))
    dataframes = {}
    
    for file_path in csv_files:
        df = load_csv_data(file_path, columns)
        if df is not None:
            dataframes[Path(file_path).name] = df
            
    return dataframes

def load_excel_data(file_path: str, columns: List[str]) -> Optional[pd.DataFrame]:
    """
    Load specific columns from Excel file
    
    Args:
        file_path: Path to the Excel file
        columns: List of column names to load
        
    Returns:
        DataFrame with specified columns or None if error occurs
    """
    try:
        df = pd.read_excel(file_path, usecols=columns)
        print(f"Successfully loaded Excel {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading Excel {file_path}: {str(e)}")
        return None

def load_all_excel_files(data_dir: str, columns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load all Excel files from a directory with specific columns
    
    Args:
        data_dir: Directory containing Excel files
        columns: List of column names to load
        
    Returns:
        Dictionary mapping filenames to DataFrames
    """
    excel_files = glob.glob(str(Path(data_dir) / "*.xlsx"))
    excel_files.extend(glob.glob(str(Path(data_dir) / "*.xls")))
    dataframes = {}
    
    for file_path in excel_files:
        df = load_excel_data(file_path, columns)
        if df is not None:
            dataframes[Path(file_path).name] = df
            
    return dataframes

def load_all_files(data_dir: str, columns: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load all CSV and Excel files from a directory with specific columns
    
    Args:
        data_dir: Directory containing files
        columns: List of column names to load
        
    Returns:
        Dictionary mapping filenames to DataFrames
    """
    # Load CSV files
    csv_dataframes = load_all_csvs(data_dir, columns)
    
    # Load Excel files
    excel_dataframes = load_all_excel_files(data_dir, columns)
    
    # Combine both dictionaries
    return {**csv_dataframes, **excel_dataframes}

def initialize_bart_ner() -> Tuple[AutoTokenizer, AutoModelForTokenClassification]:
    """
    Initialize BART-base-NER model and tokenizer
    
    Returns:
        Tuple of (tokenizer, model)
    """
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    model = AutoModelForTokenClassification.from_pretrained("facebook/bart-base")
    return tokenizer, model

def process_text_with_ner(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification
) -> Tuple[List[str], torch.Tensor]:
    """
    Process text using BART-NER
    
    Args:
        text: Input text to process
        tokenizer: BART tokenizer
        model: BART model
        
    Returns:
        Tuple of (tokens, predictions)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)
    
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    return tokens, predictions[0]

def main() -> None:
    # Configuration
    data_dir = "../data"  # Directory containing CSV files
    columns_to_load = ["summary", "text", "title"]  # Replace with your column names
    
    # Load all files (both CSV and Excel)
    dataframes = load_all_files(data_dir, columns_to_load)
    if not dataframes:
        print("No files found or loaded")
        return
    
    # Initialize BART-NER
    tokenizer, model = initialize_bart_ner()
    
    # Process each DataFrame
    for filename, df in dataframes.items():
        print(f"\nProcessing {filename}:")
        if 'text_column' in df.columns:  # Replace with your text column name
            # Process first row as example (adjust as needed)
            sample_text = df['text_column'].iloc[0]
            tokens, predictions = process_text_with_ner(sample_text, tokenizer, model)
            
            # Print results
            print(f"NER Results for {filename}:")
            for token, pred in zip(tokens, predictions):
                print(f"Token: {token}, Prediction: {pred.item()}")

if __name__ == "__main__":
    main() 