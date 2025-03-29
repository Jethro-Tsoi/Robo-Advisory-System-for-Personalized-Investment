# Active Development Context

## Project Overview
Financial sentiment analysis system using Google's Gemma 3, Gamma 3 and FinBERT models with a modern web interface.

## Current Implementation Status

### 1. Machine Learning Pipeline
- ✅ NER processing with bert-base-ner to identify entity types
- ✅ Stock symbol detection and verification using yfinance
- ✅ Data labeling with Mistral AI API (5-class classification) with Hugging Face's stock_market_tweets dataset
- ✅ Resume-capable data labeling with 00_data_labeling_with_resume.ipynb
- ✅ Gamma 3 model with LoRA fine-tuning and early stopping
- ✅ FinBERT model implementation
- ✅ Multi-metric evaluation system
- 🔄 Gemma 3 model implementation with LoRA (in progress)

### 2. Web Application
- ✅ FastAPI backend with model serving
- ✅ Next.js frontend with TypeScript
- ✅ Real-time visualization components
- ✅ Docker development environment

### 3. Development Environment
- ✅ Docker Compose setup with Makefile commands
- ✅ Comprehensive environment variables in .env
- ✅ Development tools container for enhanced development
- ✅ Hot reloading for both frontend and backend

### 4. Sentiment Classes
1. STRONGLY_POSITIVE
2. POSITIVE
3. NEUTRAL
4. NEGATIVE
5. STRONGLY_NEGATIVE

### 5. Model Training Features
- LoRA configuration (r=8, alpha=16)
- Multi-metric early stopping
- Comprehensive evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Cohen's Kappa
  - Matthews Correlation Coefficient
  - ROC-AUC

### 6. Data Processing Enhancement
- Entity recognition using BERT-based NER model
- Stock symbol extraction and verification
- Focus on stock-specific sentiment analysis
- Direct access to Hugging Face's stock_market_tweets dataset with Polars
- Resume-capable data labeling with Mistral AI API 

### 7. Development Workflow
- Docker-centric development using `make` commands
- Container-based services with proper resource allocation
- Jupyter notebook integration via `make jupyter`
- Testing via `make test-backend` and `make test-frontend`

## Current Working Branches
- main: Primary development branch
- feature/model-training: Model training implementations
- feature/web-interface: Web application development

## Recent Updates
- Replaced OpenRouter Gemini 2.5 Pro API with Mistral AI API for sentiment labeling
- Implemented resume-capable data labeling with the new notebook 00_data_labeling_with_resume.ipynb
- Deprecated 01_data_preparation.ipynb in favor of 00_data_labeling_with_resume.ipynb
- Added KeyManager class for handling multiple Mistral API keys with rotation
- Added parallel processing with ThreadPoolExecutor for better performance
- Restructured data labeling workflow to support resuming from previous runs
- Added 5-class sentiment labeling using Mistral AI API
- Added stock symbol detection and verification step
- Added Named Entity Recognition (NER) preprocessing
- Modified data labeling to focus on stock-specific tweets
- Added Gemma 3 model implementation with LoRA fine-tuning
- Restructured development environment to be Docker-centric
- Replaced setup.sh with comprehensive Makefile
- Added development tools container
- Improved build system and cleanup procedures

## Next Steps
1. Complete data labeling of the stock_market_tweets dataset using the resume-capable approach
2. Train models using the newly labeled data
3. Complete Gemma 3 model implementation and training
4. Implement model versioning system
5. Add model performance comparison visualization
6. Implement real-time sentiment analysis API
7. Add automated testing suite

## Technical Decisions
1. Using 5-class sentiment labeling for more focused sentiment analysis
2. Using Mistral AI API instead of OpenRouter for better reliability in sentiment classification
3. Using Polars instead of Pandas for more efficient dataframe operations
4. Using BERT-based NER to identify entities in financial tweets
5. Focusing ML training on tweets with verified stock symbols
6. Using LoRA for Gamma 3 and Gemma 3 to reduce training resources
7. Implementing resume-capable data labeling to handle large datasets reliably
8. Multi-metric early stopping for better model quality
9. Docker-based development for consistency
10. TypeScript + Tailwind for modern frontend
11. Makefile for standardized commands

## Dependencies
```
Python: 3.9+
Node.js: 18+
Docker & Docker Compose
Frameworks:
- FastAPI
- Next.js
- Transformers
- PyTorch
- Chart.js
- Polars
Additional:
- BERT-based NER
- yfinance
- Mistral AI API
- huggingface_hub
```

## API Endpoints
- GET `/metrics` - Model performance metrics
- GET `/confusion_matrices` - Confusion matrices
- GET `/sample_predictions` - Sample predictions
- GET `/performance_comparison` - Model comparison

## Key Project Files
- `notebooks/00_data_labeling_with_resume.ipynb` - Enhanced data labeling using Mistral AI API with resume capability
- `notebooks/00b_ner_stock_identification.ipynb` - NER and stock symbol detection
- `notebooks/02a_gamma3_training_lora.ipynb` - Gamma 3 model training
- `notebooks/02b_gemma3_training_lora.ipynb` - Gemma 3 model training
- `notebooks/02b_finbert_training.ipynb` - FinBERT model training
- `web/backend/` - FastAPI backend implementation
- `web/frontend/` - Next.js frontend with TypeScript
- `Makefile` - Development commands and workflow
- `docker-compose.yml` - Container configuration
