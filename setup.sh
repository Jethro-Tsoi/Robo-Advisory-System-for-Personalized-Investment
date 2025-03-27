#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up Financial Sentiment Analysis project...${NC}\n"

# Check if Python 3.9+ is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.9 or later.${NC}"
    exit 1
fi

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo -e "${BLUE}Upgrading pip...${NC}"
pip install --upgrade pip

# Install Python dependencies
echo -e "${BLUE}Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${BLUE}Creating project directories...${NC}"
mkdir -p data/models
mkdir -p logs
mkdir -p notebooks/.ipynb_checkpoints

# Setup frontend
if command -v node &> /dev/null; then
    echo -e "${BLUE}Setting up frontend...${NC}"
    cd web/frontend
    npm install
    cd ../..
else
    echo -e "${RED}Node.js is not installed. Frontend setup skipped.${NC}"
    echo "Please install Node.js and run 'npm install' in the web/frontend directory."
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}Creating .env file...${NC}"
    echo "# API Keys
GOOGLE_API_KEY=your_api_key_here

# Environment
ENVIRONMENT=development

# Model paths
MODEL_PATH=./models

# API settings
API_HOST=0.0.0.0
API_PORT=8000" > .env
fi

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "${BLUE}To activate the virtual environment:${NC}"
echo "source venv/bin/activate"
echo -e "\n${BLUE}To start the development environment:${NC}"
echo "cd web && ./dev.sh"
echo -e "\n${BLUE}To run Jupyter notebooks:${NC}"
echo "jupyter notebook notebooks/"
