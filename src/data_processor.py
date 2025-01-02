from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import glob
import sys
import logging
import json
import csv

def setup_logging():
    """Configure logging to both file and console"""
    log_file = Path("../logs/process.log")
    log_file.parent.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will print to console
        ]
    )

def load_csv_data(file_path: str, columns: List[str]) -> Optional[pd.DataFrame]:
    try:
        # First read the CSV to get available columns
        df = pd.read_csv(file_path)
        # Filter to only use columns that exist in the file
        available_columns = [col for col in columns if col in df.columns]
        
        if not available_columns:
            logging.warning(f"None of the requested columns {columns} found in {file_path}")
            return None
            
        df = df[available_columns]
        logging.info(f"Successfully loaded CSV {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading CSV {file_path}: {str(e)}")
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
    try:
        # First read the Excel to get available columns
        df = pd.read_excel(file_path, engine='openpyxl')
        # Filter to only use columns that exist in the file
        available_columns = [col for col in columns if col in df.columns]
        
        if not available_columns:
            logging.warning(f"None of the requested columns {columns} found in {file_path}")
            return None
            
        df = df[available_columns]
        logging.info(f"Successfully loaded Excel {file_path} with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Error loading Excel {file_path}: {str(e)}")
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

def initialize_ner() -> Tuple[AutoTokenizer, AutoModelForTokenClassification]:
    """
    Initialize BERT-based NER model and tokenizer
    
    Returns:
        Tuple of (tokenizer, model)
    """
    model_name = "dslim/bert-base-NER"
    logging.info(f"Loading NER model: {model_name}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        logging.info("NER model loaded successfully")
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        logging.info(f"Model moved to device: {device}")
        
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error loading NER model: {str(e)}")
        raise

def get_entity_label(prediction_id: int) -> str:
    """Get readable label for prediction ID"""
    labels = {
        0: "O",      # Outside of named entity
        1: "B-MISC", # Beginning of miscellaneous entity
        2: "I-MISC", # Inside of miscellaneous entity
        3: "B-PER",  # Beginning of person name
        4: "I-PER",  # Inside of person name
        5: "B-ORG",  # Beginning of organization
        6: "I-ORG",  # Inside of organization
        7: "B-LOC",  # Beginning of location
        8: "I-LOC"   # Inside of location
    }
    return labels.get(prediction_id, "O")

def process_text_with_ner(
    text: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification
) -> Tuple[List[str], List[str]]:
    """
    Process text using NER model and return formatted results
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=2)[0]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Get entity labels
    entity_labels = [get_entity_label(p.item()) for p in predictions]
    
    return tokens, entity_labels

def format_ner_output(tokens: List[str], labels: List[str]) -> str:
    """Format NER results in a readable way"""
    result = []
    current_entity = None
    current_text = []
    
    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if label == "O":
            if current_entity:
                result.append(f"{' '.join(current_text)} ({current_entity})")
                current_entity = None
                current_text = []
            result.append(token)
        else:
            entity_type = label[2:]  # Remove B- or I- prefix
            if label.startswith("B-"):
                if current_entity:
                    result.append(f"{' '.join(current_text)} ({current_entity})")
                current_entity = entity_type
                current_text = [token.replace("##", "")]
            elif label.startswith("I-"):
                current_text.append(token.replace("##", ""))
                
    if current_entity:
        result.append(f"{' '.join(current_text)} ({current_entity})")
        
    return " ".join(result)

def save_results_multiple_formats(results: Dict[str, List[Dict]], output_dir: Path) -> None:
    """Save NER results in multiple formats"""
    output_dir.mkdir(exist_ok=True)
    
    # Save as text file
    with open(output_dir / "ner_results.txt", "w", encoding="utf-8") as f:
        for filename, entries in results.items():
            f.write(f"\nProcessing {filename}:\n")
            f.write("NER Results:\n")
            for entry in entries:
                f.write(f"{entry['text']}\n")
    
    # Save as JSON
    with open(output_dir / "ner_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save as CSV
    with open(output_dir / "ner_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Text", "Entities"])
        for filename, entries in results.items():
            for entry in entries:
                writer.writerow([
                    filename,
                    entry['text'],
                    "|".join(f"{e['text']}({e['type']})" for e in entry['entities'])
                ])

def extract_entities(tokens: List[str], labels: List[str]) -> List[Dict]:
    """Extract entities from tokens and labels"""
    entities = []
    current_entity = None
    current_text = []
    
    for token, label in zip(tokens, labels):
        if token in ["[CLS]", "[SEP]", "[PAD]"]:
            continue
            
        if label.startswith("B-"):
            if current_entity:
                entities.append({
                    "text": "".join(current_text).replace("##", ""),
                    "type": current_entity
                })
            current_entity = label[2:]  # Remove B- prefix
            current_text = [token]
        elif label.startswith("I-"):
            if current_entity:
                current_text.append(token)
        else:  # O label
            if current_entity:
                entities.append({
                    "text": "".join(current_text).replace("##", ""),
                    "type": current_entity
                })
                current_entity = None
                current_text = []
    
    if current_entity:
        entities.append({
            "text": "".join(current_text).replace("##", ""),
            "type": current_entity
        })
    
    return entities

def main() -> None:
    # Setup logging first
    setup_logging()
    
    # Create output directory
    output_dir = Path("../results")
    output_dir.mkdir(exist_ok=True)
    
    data_dir = "../data"
    columns_to_load = ["summary", "text", "title"]
    
    # Store all results
    all_results = {}
    
    dataframes = load_all_files(data_dir, columns_to_load)
    if not dataframes:
        logging.warning("No files found or loaded")
        return
    
    tokenizer, model = initialize_ner()
    
    for filename, df in dataframes.items():
        all_results[filename] = []
        text_columns = ['text', 'content', 'summary']
        text_col = next((col for col in text_columns if col in df.columns), None)
        
        if text_col:
            # Process all rows in the DataFrame
            for idx, row in df.iterrows():
                text = row[text_col]
                tokens, labels = process_text_with_ner(text, tokenizer, model)
                entities = extract_entities(tokens, labels)
                
                all_results[filename].append({
                    'text': text,
                    'entities': entities
                })
    
    # Save results in multiple formats
    save_results_multiple_formats(all_results, output_dir)

if __name__ == "__main__":
    main() 