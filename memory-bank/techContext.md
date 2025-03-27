# Technical Context

## Development Environment

### Docker Configuration
1. Core Services
   - Frontend container (Next.js)
   - Backend container (FastAPI)
   - Development tools container

2. Build System
   - Makefile for common tasks
   - Multi-stage Dockerfiles
   - Development/Production configs
   - Hot reloading support

3. Resource Management
   - Volume mounting
   - Container networking
   - Memory/CPU limits
   - GPU access (optional)

## Technology Stack

### Machine Learning
1. Models
   - Google Gemma 3
     - 12B parameter model
     - LoRA fine-tuning (r=8, alpha=16)
     - Target modules: q_proj, v_proj
     - 8-bit quantization for efficient training
     - Gradient clipping for training stability
   - Google Gamma 3
     - LoRA fine-tuning (r=8, alpha=16)
     - Target modules: q_proj, v_proj
     - Early stopping with multiple metrics
   - FinBERT
     - Native fine-tuning
     - Domain-specific pre-training
     - Financial text specialization

2. Libraries
   - PyTorch >= 2.1.2
   - Transformers >= 4.36.2
   - PEFT >= 0.7.1
   - Accelerate >= 0.25.0
   - Scikit-learn >= 1.3.2

### Backend
1. Framework
   - FastAPI >= 0.109.0
   - Uvicorn >= 0.27.0
   - Pydantic >= 2.5.3

2. Package Management
   - uv for fast installation
   - Docker-based dependency management
   - Version pinning

### Frontend
1. Core
   - Next.js 14
   - TypeScript 5
   - React 18

2. UI/UX
   - Tailwind CSS
   - Shadcn UI
   - Chart.js
   - React Query

3. State Management
   - Zustand
   - React Query for API state

## Build and Deployment

### Development Workflow
1. Local Development
   ```bash
   make up        # Start services
   make dev       # Start with tools
   make jupyter   # Access notebooks
   ```

2. Testing
   ```bash
   make test-backend
   make test-frontend
   ```

3. Cleanup
   ```bash
   make clean    # Remove containers
   make down     # Stop services
   ```

### Container Architecture
```
┌─────────────────┐
│    Frontend     │
│    (Next.js)    │
├─────────────────┤
│    Backend      │
│    (FastAPI)    │
├─────────────────┤
│  Dev Tools      │
│  (Optional)     │
└─────────────────┘
```

## System Architecture

### Data Flow
```
Raw Tweets → Preprocessing → Model Inference → API → Frontend Display
```

### Model Pipeline
```
Data Labeling (Gemini) → Training → Evaluation → Deployment
```

### API Architecture
```
Client Request → FastAPI Router → Model Service → Response
```

## Performance Considerations

### Model Optimization
1. Gemma 3 Optimization
   - 8-bit quantization
   - LoRA for parameter-efficient fine-tuning
   - Gradient clipping (max_norm=1.0)
   - Smaller batch size (8)
   - Lower learning rate (1e-5)
   - Linear warmup scheduler

2. LoRA Parameters
   - Rank: 8
   - Alpha: 16
   - Target modules: attention layers
   - Dropout: 0.1

3. Training
   - Batch size: 8-16 (model dependent)
   - Learning rate: 1e-5 to 2e-5
   - Early stopping patience: 3
   - Multi-metric monitoring

### Container Optimization
1. Resource Limits
   - Memory constraints
   - CPU allocation
   - Volume management
   - Network configuration

2. Development Performance
   - Hot reloading
   - Volume mounts
   - Cache utilization
   - Build optimization

## Security Measures

### Container Security
1. Resource Isolation
   - Network segmentation
   - Volume permissions
   - Resource limits
   - Environment separation

2. Access Control
   - API authentication
   - Development restrictions
   - Environment variables
   - Secrets management

## Monitoring and Logging

### Container Monitoring
1. Health Checks
   ```yaml
   healthcheck:
     test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
     interval: 30s
     timeout: 10s
     retries: 3
   ```

2. Resource Tracking
   - Container stats
   - Volume usage
   - Network metrics
   - Application logs

## Testing Strategy

### Development Testing
1. Local Testing
   - Hot reload testing
   - Component testing
   - API testing
   - Integration tests

2. Container Testing
   - Build verification
   - Resource usage
   - Network connectivity
   - Volume persistence
