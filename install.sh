#!/bin/bash

# Install script for matgen and matmul
# This script builds and installs the matrix generation and multiplication tools

set -e  # Exit on any error

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BUILD_TYPE="${BUILD_TYPE:-release}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc not found. Please install CUDA development tools."
        exit 1
    fi

    print_info "All dependencies satisfied."
}

# Create install directory if it doesn't exist
create_install_dir() {
    if [ ! -d "$INSTALL_DIR" ]; then
        print_info "Creating install directory: $INSTALL_DIR"
        mkdir -p "$INSTALL_DIR"
    fi
}

# Build the applications
build_applications() {
    print_info "Building applications..."

    # Build matgen
    print_info "Building matgen..."
    ./build_single.sh day_008_matgen_main.cu

    # Build matmul
    print_info "Building matmul..."
    ./build_single.sh day_008_matmul_main.cu

    # Build matpose
    print_info "Building matpose..."
    ./build_single.sh day_009_matpose_main.cu

    print_info "Build completed successfully."
}

# Install the applications
install_applications() {
    print_info "Installing applications to $INSTALL_DIR..."

    # Determine which build to install
    if [ "$BUILD_TYPE" = "debug" ]; then
        SUFFIX="_dn"
    else
        SUFFIX="_rn"
    fi

    # Copy executables
    if [ -f "build/day_008_matgen_main.cu${SUFFIX}" ]; then
        cp "build/day_008_matgen_main.cu${SUFFIX}" "$INSTALL_DIR/matgen"
        chmod +x "$INSTALL_DIR/matgen"
        print_info "Installed matgen"
    else
        print_error "matgen executable not found in build directory"
        exit 1
    fi

    if [ -f "build/day_009_matpose_main.cu${SUFFIX}" ]; then
        cp "build/day_009_matpose_main.cu${SUFFIX}" "$INSTALL_DIR/matpose"
        chmod +x "$INSTALL_DIR/matpose"
        print_info "Installed matpose"
    else
        print_error "matpose executable not found in build directory"
        exit 1
    fi

}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    if [ -x "$INSTALL_DIR/matgen" ]; then
        print_info "matgen installed successfully"
    else
        print_error "matgen installation failed"
        exit 1
    fi

    if [ -x "$INSTALL_DIR/matmul" ]; then
        print_info "matmul installed successfully"
    else
        print_error "matmul installation failed"
        exit 1
    fi

    if [ -x "$INSTALL_DIR/matpose" ]; then
        print_info "matpose installed successfully"
    else
        print_error "matpose installation failed"
        exit 1
    fi

    # Test the pipeline
    print_info "Testing installation..."
    if "$INSTALL_DIR/matgen" uniform 12345 2 2 2 2 2 | "$INSTALL_DIR/matpose" | "$INSTALL_DIR/matmul" tiled > /dev/null 2>&1; then
        print_info "Installation test passed"
    else
        print_warning "Installation test failed, but binaries are installed"
    fi
}

# Add to PATH information
path_info() {
    print_info "Installation complete!"
    echo
    echo "The binaries have been installed to: $INSTALL_DIR"
    echo
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        print_warning "Note: $INSTALL_DIR is not in your PATH"
        echo "To use matgen and matmul from anywhere, add this line to your ~/.bashrc or ~/.zshrc:"
        echo "export PATH=\"$INSTALL_DIR:\$PATH\""
        echo
        echo "Or run: echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc"
        echo "Then restart your terminal or run: source ~/.bashrc"
    else
        print_info "$INSTALL_DIR is already in your PATH"
    fi
    echo
    echo "Usage examples:"
    echo "  $INSTALL_DIR/matgen uniform 12345 2 10 20 20 10 | $INSTALL_DIR/matmul simple"
    echo "  $INSTALL_DIR/matgen uniform 54321 2 3 2 2 3 | $INSTALL_DIR/matmul tiled"
}

# Print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --debug         Install debug builds instead of release builds"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  INSTALL_DIR         Installation directory (default: \$HOME/.local/bin)"
    echo "  BUILD_TYPE          Build type: release or debug (default: release)"
    echo ""
    echo "Example:"
    echo "  $0                                    # Install release builds to ~/.local/bin"
    echo "  $0 --debug                           # Install debug builds"
    echo "  INSTALL_DIR=/usr/local/bin $0        # Install to /usr/local/bin"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            BUILD_TYPE="debug"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main installation process
main() {
    print_info "Starting installation of matgen and matmul..."
    print_info "Install directory: $INSTALL_DIR"
    print_info "Build type: $BUILD_TYPE"
    echo

    check_dependencies
    create_install_dir
    build_applications
    install_applications
    verify_installation
    path_info
}

# Run main function
main
