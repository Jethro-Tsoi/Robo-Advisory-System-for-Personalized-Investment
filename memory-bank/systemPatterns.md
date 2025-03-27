# System Design Patterns and Conventions

## Code Organization

### Directory Structure
```
project/
├── data/           # Data storage and models
├── notebooks/      # Jupyter notebooks
├── web/           # Web application
│   ├── backend/   # FastAPI application
│   └── frontend/  # Next.js application
├── src/           # Core Python modules
└── tests/         # Test suites
```

### File Naming Conventions
- Python: snake_case (e.g., model_utils.py)
- TypeScript: PascalCase for components (e.g., ModelMetrics.tsx)
- CSS: kebab-case (e.g., button-primary.css)
- Notebooks: numbered prefixes (e.g., 01_data_preparation.ipynb)

## Design Patterns

### Frontend Patterns

1. Component Structure
```typescript
// src/components/ComponentName/index.tsx
import styles from './styles.module.css'

interface Props {
  // props definition
}

export const ComponentName: React.FC<Props> = ({ prop }) => {
  // component logic
}
```

2. Custom Hooks
```typescript
// src/hooks/useName.ts
export const useName = (params) => {
  // hook logic
  return { data, methods }
}
```

3. API Integration
```typescript
// src/api/endpoints.ts
export const fetchData = async () => {
  const response = await axios.get('/api/endpoint')
  return response.data
}
```

### Backend Patterns

1. Route Organization
```python
# routes/endpoint.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/endpoint")
async def handle_request():
    # request handling
```

2. Model Services
```python
# services/model_service.py
class ModelService:
    def __init__(self):
        self.model = load_model()
    
    async def predict(self, text: str):
        # prediction logic
```

3. Error Handling
```python
from fastapi import HTTPException

def handle_error(error):
    raise HTTPException(
        status_code=error.status_code,
        detail=str(error)
    )
```

## State Management

### Frontend State
1. API State (React Query)
```typescript
const { data, isLoading } = useQuery({
  queryKey: ['key'],
  queryFn: fetchData
})
```

2. UI State (Zustand)
```typescript
const useStore = create((set) => ({
  state: initialState,
  setState: (newState) => set({ state: newState })
}))
```

### Backend State
1. Model Cache
```python
class ModelCache:
    def __init__(self):
        self._cache = {}
    
    def get_or_compute(self, key, compute_fn):
        if key not in self._cache:
            self._cache[key] = compute_fn()
        return self._cache[key]
```

## API Conventions

### Endpoint Structure
```
GET    /api/metrics
GET    /api/predictions
POST   /api/analyze
```

### Response Format
```json
{
  "success": true,
  "data": {},
  "error": null,
  "metadata": {
    "timestamp": "",
    "version": ""
  }
}
```

### Error Format
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description"
  },
  "data": null
}
```

## Documentation Patterns

### Code Documentation
```python
def function_name(param: type) -> return_type:
    """
    Function description.

    Args:
        param: Parameter description

    Returns:
        Return value description

    Raises:
        ErrorType: Error description
    """
```

### API Documentation
```python
@router.get("/endpoint")
async def endpoint():
    """
    Endpoint description.

    Returns:
        JSON response description

    Raises:
        HTTPException: Error conditions
    """
```

## Testing Patterns

### Frontend Testing
```typescript
describe('Component', () => {
  it('should render correctly', () => {
    render(<Component />)
    expect(screen.getByText('text')).toBeInTheDocument()
  })
})
```

### Backend Testing
```python
def test_endpoint():
    response = client.get("/endpoint")
    assert response.status_code == 200
    assert response.json()["success"] == True
```

## Monitoring Patterns

### Logging
```python
logger.info("Operation completed", extra={
    "operation": "name",
    "duration": time_taken,
    "status": "success"
})
```

### Metrics
```python
def track_metric(name: str, value: float, tags: dict):
    metrics.gauge(
        name,
        value,
        tags=tags
    )
```

## Security Patterns

### Input Validation
```python
def validate_input(text: str) -> bool:
    if not text or len(text) > MAX_LENGTH:
        raise ValidationError("Invalid input")
    return text
```

### Output Sanitization
```python
def sanitize_output(prediction: dict) -> dict:
    return {
        "label": clean_text(prediction["label"]),
        "confidence": min(max(prediction["confidence"], 0), 1)
    }
