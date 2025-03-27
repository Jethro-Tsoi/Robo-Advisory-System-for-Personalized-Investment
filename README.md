# Financial Sentiment Analysis with LLMs

This project implements a comprehensive financial sentiment analysis system using Google's Gamma 3 and FinBERT models, with a modern web interface for visualization and comparison.

## Features

- 🤖 Multi-model sentiment analysis (Gamma 3 and FinBERT)
- 📊 Interactive performance visualization dashboard
- 🔄 Real-time sentiment prediction
- 🎯 7-class sentiment classification
- 📈 Comprehensive model evaluation metrics
- 🚀 Modern web interface with TypeScript and Tailwind CSS
- 🐳 Docker-based development environment

## Project Structure

```
.
├── data/                  # Data storage
│   ├── models/           # Trained model files
│   └── tweets/           # Raw and processed tweets
├── notebooks/            # Jupyter notebooks
│   ├── 00_data_labeling.ipynb        # Data labeling with Gemini
│   ├── 01_data_preparation.ipynb     # Data preprocessing
│   ├── 02a_gamma3_training_lora.ipynb # Gamma 3 training
│   └── 02b_finbert_training.ipynb    # FinBERT training
├── web/                  # Web application
│   ├── backend/         # FastAPI backend
│   └── frontend/        # Next.js frontend
└── setup.sh             # Setup script
```

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker and Docker Compose (optional)
- Google Cloud API key (for Gemini)

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-sentiment.git
cd financial-sentiment
```

2. Run the setup script:
```bash
./setup.sh
```

3. Configure your environment:
   - Add your Google Cloud API key to `.env`
   - Adjust other settings as needed

4. Start the development environment:
```bash
cd web
./dev.sh
```

### Manual Setup

If you prefer to set up components individually:

1. Python Environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

2. Frontend:
```bash
cd web/frontend
npm install
npm run dev
```

3. Backend:
```bash
cd web/backend
uvicorn main:app --reload
```

## Model Training

### Data Labeling

1. Run the data labeling notebook:
```bash
jupyter notebook notebooks/00_data_labeling.ipynb
```

2. Configure your Gemini API key in the notebook
3. Run all cells to process and label the tweets

### Training Models

#### Gamma 3 with LoRA

1. Run the training notebook:
```bash
jupyter notebook notebooks/02a_gamma3_training_lora.ipynb
```

Features:
- LoRA for efficient fine-tuning
- Multi-metric early stopping
- Comprehensive evaluation metrics

#### FinBERT

1. Run the training notebook:
```bash
jupyter notebook notebooks/02b_finbert_training.ipynb
```

Features:
- Native fine-tuning
- Early stopping
- Performance metrics

## Web Interface

### Frontend (Next.js + TypeScript)

- Modern React components with TypeScript
- Tailwind CSS for styling
- Chart.js for visualizations
- Real-time model comparison

### Backend (FastAPI)

API Endpoints:
- GET `/metrics` - Model performance metrics
- GET `/confusion_matrices` - Confusion matrices
- GET `/sample_predictions` - Sample predictions
- GET `/performance_comparison` - Model comparison

## Docker Development

1. Start containers:
```bash
docker-compose up --build
```

2. Access services:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Google's Gamma 3 model
- FinBERT team
- HuggingFace Transformers library
- FastAPI framework
- Next.js team
