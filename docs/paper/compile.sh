#!/bin/bash

# Script to compile the SO(3) Tube Smoothing paper
# Run this in a directory with pdflatex installed

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "==================================="
echo "SO(3) Tube Smoothing Paper Compiler"
echo "==================================="
echo ""

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo -e "${RED}Error: pdflatex not found!${NC}"
    echo ""
    echo "Install pdflatex using one of the following methods:"
    echo "  Ubuntu/Debian:  sudo apt-get install texlive-full"
    echo "  macOS:  brew install mactex"
    echo "  Overleaf/Overleaf: Upload files and compile online"
    exit 1
fi

echo -e "${GREEN}pdflatex found, starting compilation...${NC}"
echo ""

# Create build directory
BUILD_DIR="build"
mkdir -p "$BUILD_DIR"

echo "Step 1: First compilation (pdflatex)..."
pdflatex -interaction=nonstopmode main.tex -output-directory="$BUILD_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}First compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}First compilation successful!${NC}"
echo ""

echo "Step 2: Running bibtex..."
bibtex "$BUILD_DIR/main.aux"

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Bibtex failed, continuing without references...${NC}"
else
    echo -e "${GREEN}Bibtex successful!${NC}"
    echo ""

    echo "Step 3: Second compilation (pdflatex)..."
    pdflatex -interaction=nonstopmode main.tex -output-directory="$BUILD_DIR"

    if [ $? -ne 0 ]; then
        echo -e "${RED}Second compilation failed!${NC}"
        exit 1
    fi

    echo -e "${GREEN}Second compilation successful!${NC}"
fi

echo ""
echo "Step 4: Third compilation (pdflatex)..."
pdflatex main.tex -output-directory="$BUILD_DIR"

if [ $? -ne 0 ]; then
    echo -e "${RED}Third compilation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Third compilation successful!${NC}"
echo ""

# Check if PDF was generated
if [ -f "$BUILD_DIR/main.pdf" ]; then
    PDF_SIZE=$(du -h "$BUILD_DIR/main.pdf" | cut -f1)
    echo -e "${GREEN}===================================${NC}"
    echo -e "${GREEN}Compilation successful!${NC}"
    echo -e "${GREEN}===================================${NC}"
    echo ""
    echo "PDF generated: $BUILD_DIR/main.pdf"
    echo "File size: $PDF_SIZE"
    echo ""
    echo "Optional: View PDF with:"
    echo "  xdg-open $BUILD_DIR/main.pdf"
    echo "  open $BUILD_DIR/main.pdf"
    echo "  evince $BUILD_DIR/main.pdf"
    echo ""
    echo "Online compilation alternatives:"
    echo "  1. Overleaf: Upload all files to overleaf.com"
    echo "  2. Papeeria: Upload to papeeria.eu"
    echo "  3. ShareLaTeX: Upload to sharelatex.com"
else
    echo -e "${RED}PDF not found in build directory${NC}"
    exit 1
fi

# Cleanup: Move PDF to current directory
if [ -f "$BUILD_DIR/main.pdf" ]; then
    cp "$BUILD_DIR/main.pdf" .
    echo -e "${GREEN}PDF copied to current directory${NC}"
fi
