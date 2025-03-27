# Financial Sentiment Analysis Notebooks

This directory contains Jupyter notebooks for processing financial tweets, performing sentiment analysis, and training machine learning models for the financial sentiment analysis project.

## Notebook Execution Order

The notebooks should be executed in the following order:

1. `00b_ner_stock_identification.ipynb` - Processes raw tweets with Named Entity Recognition and identifies verified stock symbols
2. `00c_data_labeling_with_stocks.ipynb` - Labels tweets with verified stock symbols using Gemini API
3. `00_data_labeling.ipynb` - (Optional) Original data labeling without stock focus
4. `01_data_preparation.ipynb` - Prepares labeled data for model training
5. `02a_gamma3_training_lora.ipynb` - Trains the Gamma 3 model with LoRA fine-tuning
6. `02b_finbert_training.ipynb` - Trains the FinBERT model
7. `02b_gemma3_training_lora.ipynb` - Trains the Gemma 3 model with LoRA fine-tuning

## Notebook Descriptions

### 00b_ner_stock_identification.ipynb
- Processes raw tweet CSV files using BERT-based Named Entity Recognition (NER)
- Identifies entity types and their values within tweets
- Extracts potential stock symbols (patterns like $XXX)
- Verifies extracted symbols against real stocks using yfinance
- Outputs processed data with NER results and verified stock symbols
- Creates both a full dataset and a filtered dataset of tweets with verified stock symbols

### 00c_data_labeling_with_stocks.ipynb
- Loads the preprocessed data with verified stock symbols
- Labels the sentiment of each tweet using Google's Gemini API
- Focuses the sentiment analysis on the specific stocks mentioned
- Filters out non-relevant tweets
- Creates datasets for stock-specific sentiment analysis and model training

### 00_data_labeling.ipynb
- Original data labeling notebook without stock symbol focus
- Labels general financial sentiment using Google's Gemini API
- Can be used if a broader sentiment analysis is needed

### 01_data_preparation.ipynb
- Performs cleaning and preprocessing on the labeled data
- Prepares features for model training
- Handles data splitting and formatting

### 02a_gamma3_training_lora.ipynb
- Implements Gamma 3 model with LoRA fine-tuning
- Configures training parameters
- Includes multi-metric evaluation
- Saves the trained model

### 02b_finbert_training.ipynb
- Implements FinBERT model training
- Fine-tunes on financial tweet data
- Includes evaluation metrics
- Saves the trained model

### 02b_gemma3_training_lora.ipynb
- Implements Gemma 3 model with LoRA fine-tuning
- Configures training parameters
- Includes multi-metric evaluation
- Saves the trained model

## Output Files

The notebooks will generate various output files in the `../data/` directory:

- `tweets_with_ner_and_stocks.csv` - All tweets with NER and stock symbol information
- `tweets_with_verified_stocks.csv` - Filtered dataset with only tweets mentioning verified stock symbols
- `stock_tweets_labeled.csv` - Tweets with verified stock symbols and their sentiment labels
- `stock_tweets_for_training.csv` - Final dataset for model training, excluding non-relevant tweets
- `stock_tweets_by_symbol.csv` - Expanded dataset for analysis by individual stock symbol

## Required Dependencies

To run these notebooks, you'll need the following Python packages:
- pandas
- numpy
- transformers
- torch
- yfinance
- google-generativeai
- tqdm

You can install them using:
```
pip install pandas numpy transformers torch yfinance google-generativeai tqdm
```

Additionally, you'll need:
- A Google API key for Gemini access (set as environment variable GOOGLE_API_KEY)
- Internet access for yfinance stock symbol verification
- Sufficient disk space for the model weights and output files 