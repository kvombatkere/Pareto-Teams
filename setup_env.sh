#!/bin/bash

# Set environment name
ENV_NAME="pareto_env"

# Set requirements file
REQ_FILE="requirements.txt"

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Conda first."
    exit 1
fi

# Create a new Conda environment
echo "Creating Conda environment: $ENV_NAME..."
conda create --name $ENV_NAME python=3.13 -y  # Change Python version if needed

# Activate the environment
echo "Activating environment..."
source activate $ENV_NAME  # Use 'conda activate' if using conda >=4.4

# Install dependencies using Conda first
if [ -f "$REQ_FILE" ]; then
    echo "Installing packages from $REQ_FILE..."
    
    # Install packages with pip
    pip install -r $REQ_FILE
else
    echo "Requirements file '$REQ_FILE' not found!"
    exit 1
fi

echo "Package Installation complete. To activate, run: conda activate $ENV_NAME"