#!/bin/bash

# Single-Cell RNA-seq Analysis - Automated Dependency Installation Script
# This script provides multiple installation options for the required dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}$1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install with conda
install_with_conda() {
    print_header "Installing dependencies with Conda..."
    
    if ! command_exists conda; then
        print_error "Conda is not installed. Please install Anaconda or Miniconda first."
        print_status "Visit: https://docs.conda.io/en/latest/miniconda.html"
        return 1
    fi
    
    print_status "Creating conda environment 'scrna-analysis'..."
    conda env create -f environment.yml
    
    print_status "Environment created successfully!"
    print_warning "To activate the environment, run:"
    echo "    conda activate scrna-analysis"
    
    return 0
}

# Function to install with pip
install_with_pip() {
    print_header "Installing dependencies with pip..."
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed. Please install Python 3.8 or higher."
        return 1
    fi
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "You're not in a virtual environment."
        read -p "Do you want to create a virtual environment? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Creating virtual environment 'scrna-env'..."
            python3 -m venv scrna-env
            print_status "Activating virtual environment..."
            source scrna-env/bin/activate
            print_status "Virtual environment activated."
        fi
    fi
    
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_status "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
    
    print_status "Dependencies installed successfully!"
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        print_warning "Remember to activate your virtual environment before using the code:"
        echo "    source scrna-env/bin/activate"
    fi
    
    return 0
}

# Function to test installation
test_installation() {
    print_header "Testing installation..."
    
    if [[ -f "test_imports.py" ]]; then
        python test_imports.py
    else
        print_error "test_imports.py not found. Cannot verify installation."
        return 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --conda     Install using conda (recommended)"
    echo "  -p, --pip       Install using pip"
    echo "  -t, --test      Test the installation"
    echo "  -h, --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --conda     # Install with conda"
    echo "  $0 --pip       # Install with pip"
    echo "  $0 --test      # Test installation"
}

# Main installation function
main() {
    print_header "=========================================="
    print_header "Single-Cell RNA-seq Analysis Setup"
    print_header "=========================================="
    
    if [[ $# -eq 0 ]]; then
        print_status "No options provided. Showing interactive menu..."
        echo ""
        echo "Choose installation method:"
        echo "1) Conda (recommended for scientific computing)"
        echo "2) Pip (standard Python package manager)"
        echo "3) Test existing installation"
        echo "4) Exit"
        echo ""
        read -p "Enter your choice (1-4): " choice
        
        case $choice in
            1)
                install_with_conda
                if [[ $? -eq 0 ]]; then
                    print_status "Would you like to test the installation? (y/n)"
                    read -p "> " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        test_installation
                    fi
                fi
                ;;
            2)
                install_with_pip
                if [[ $? -eq 0 ]]; then
                    print_status "Would you like to test the installation? (y/n)"
                    read -p "> " -n 1 -r
                    echo
                    if [[ $REPLY =~ ^[Yy]$ ]]; then
                        test_installation
                    fi
                fi
                ;;
            3)
                test_installation
                ;;
            4)
                print_status "Exiting..."
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please run the script again."
                exit 1
                ;;
        esac
    else
        # Parse command line arguments
        while [[ $# -gt 0 ]]; do
            case $1 in
                -c|--conda)
                    install_with_conda
                    shift
                    ;;
                -p|--pip)
                    install_with_pip
                    shift
                    ;;
                -t|--test)
                    test_installation
                    shift
                    ;;
                -h|--help)
                    show_usage
                    exit 0
                    ;;
                *)
                    print_error "Unknown option: $1"
                    show_usage
                    exit 1
                    ;;
            esac
        done
    fi
    
    print_header "=========================================="
    print_status "Setup complete!"
    print_header "=========================================="
}

# Run main function
main "$@"