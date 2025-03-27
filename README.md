# Financial Sentiment Analysis with LLMs

This project implements a comprehensive financial sentiment analysis system using Google's Gamma 3 and FinBERT models, with a modern web interface for visualization and comparison.

## Features

- 🤖 Multi-model sentiment analysis (Gamma 3 and FinBERT)
- 📊 Interactive performance visualization dashboard
- 🔄 Real-time sentiment prediction
- 🎯 7-class sentiment classification
- 📈 Comprehensive model evaluation metrics
- 🚀 Modern web interface with TypeScript and Tailwind CSS
- 🐳 Containerized development environment

## Prerequisites

- Docker and Docker Compose
- Make (optional, but recommended)
- Google Cloud API key (for Gemini)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/financial-sentiment.git
cd financial-sentiment
```

2. Set up the environment:
```bash
make setup
```

3. Configure your API keys in `.env`

4. Start the development environment:
```bash
make up
```

## Available Make Commands

```bash
make help      # Show all available commands
make up        # Start all services
make down      # Stop all services
make build     # Build/rebuild services
make dev       # Start with development tools
make clean     # Remove all containers and volumes
make logs      # View service logs
make jupyter   # Start Jupyter notebook server
```

## Development Environment

The following services will be available:

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Jupyter Notebooks: http://localhost:8888

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
├── Makefile             # Development commands
└── docker-compose.yml   # Container configuration
```

## Model Training

### Data Labeling

1. Start Jupyter server:
```bash
make jupyter
```

2. Open `notebooks/00_data_labeling.ipynb`

3. Configure your Gemini API key in notebook:
```python
os.environ['GOOGLE_API_KEY'] = 'your-api-key'
```

### Training Models

1. Gamma 3 with LoRA:
- Open `notebooks/02a_gamma3_training_lora.ipynb`
- Features:
  - LoRA fine-tuning (r=8, alpha=16)
  - Multi-metric early stopping
  - Comprehensive evaluation

2. FinBERT:
- Open `notebooks/02b_finbert_training.ipynb`
- Features:
  - Native fine-tuning
  - Early stopping
  - Performance metrics

## Development

### Running Tests

```bash
# Backend tests
make test-backend

# Frontend tests
make test-frontend
```

### Development Tools

Start development environment with additional tools:
```bash
make dev
```

This includes:
- Hot reloading
- Development containers
- Debugging tools
- Live code updates

### Making Changes

1. Frontend:
- Edit files in `web/frontend/src/`
- Changes reflect instantly with hot reloading

2. Backend:
- Edit files in `web/backend/`
- Auto-reloads with code changes

3. Notebooks:
- Edit in Jupyter interface
- Auto-saves enabled

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Monitoring

View service logs:
```bash
make logs
```

Monitor specific service:
```bash
make logs service=backend  # or frontend
```

## Cleanup

Remove all containers and volumes:
```bash
make clean
```

## License

MIT License - see LICENSE file for details

## Need Help?

1. Run `make help` to see all available commands
2. Check the API documentation at http://localhost:8000/docs
3. Visit the frontend at http://localhost:3000
