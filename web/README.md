# Financial Sentiment Analysis Dashboard

A modern web application for comparing and visualizing the performance of Gamma 3, Gemma 3, and FinBERT models on financial sentiment analysis.

## Technology Stack

### Frontend
- Next.js 14
- TypeScript
- Tailwind CSS
- Chart.js
- React Query
- Shadcn UI

### Backend
- FastAPI
- Python 3.9
- Polars
- Pandas
- NumPy
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)

## Development Setup

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development without Docker)
- Python 3.9+ (for local development without Docker)
- Mistral AI API keys (for data labeling)

### Quick Start with Docker

1. Clone the repository and navigate to the web directory:
```bash
cd web
```

2. Start the development environment:
```bash
./dev.sh
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Manual Setup (Without Docker)

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Project Structure

```
web/
├── backend/
│   ├── main.py           # FastAPI application
│   └── requirements.txt  # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── app/         # Next.js pages and layouts
│   │   ├── components/  # React components
│   │   └── types/      # TypeScript definitions
│   ├── public/         # Static assets
│   └── package.json    # Node.js dependencies
└── docker-compose.yml  # Docker configuration
```

## Features

- Interactive model performance comparison
- Real-time sentiment analysis visualization
- Side-by-side model predictions
- 5-class sentiment classification (STRONGLY_POSITIVE, POSITIVE, NEUTRAL, NEGATIVE, STRONGLY_NEGATIVE)
- Responsive design with dark mode support
- REST API with automatic documentation

## API Endpoints

- GET `/metrics` - Model performance metrics
- GET `/confusion_matrices` - Confusion matrices for all models
- GET `/sample_predictions` - Sample predictions from all models
- GET `/performance_comparison` - Detailed performance comparison

## Development

### Adding New Components
1. Create component in `frontend/src/components`
2. Use TypeScript interfaces from `frontend/src/types`
3. Import and use in pages as needed

### Modifying the API
1. Update FastAPI endpoints in `backend/main.py`
2. Update TypeScript interfaces in `frontend/src/types/api.d.ts`
3. Update API calls in React components

### Running Tests
```bash
# Frontend tests
cd frontend
npm test

# Backend tests
cd backend
pytest
```

## API Key Management

For data labeling with Mistral AI, configure the API keys in the root `.env` file:

```
MISTRAL_API_KEY=your_primary_key
MISTRAL_API_KEY_1=your_second_key
```

The system includes a KeyManager class that automatically:
- Rotates between keys when rate limits are reached
- Tracks when rate-limited keys become available again
- Implements waiting strategies when all keys are limited

## Contributing

1. Create a new branch for your feature
2. Make changes and test thoroughly
3. Submit a pull request with a clear description

## License

MIT License
