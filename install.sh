#!/bin/bash

# Install script for matrix tools
# This script builds and installs applications based on a simple table configuration

set -e  # Exit on any error

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/bin}"
BUILD_TYPE="${BUILD_TYPE:-release}"

# Application table: "source_file:app_name"
# Add or modify entries here as needed
APPLICATIONS=(
    "day_008_matgen_main.cu:matgen"
    "day_008_matmul_main.cu:matmul"
    "day_009_matpose_main.cu:matpose"
    "day_010_matsum_main.cu:matsum"
)

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

    for app_entry in "${APPLICATIONS[@]}"; do
        source_file="${app_entry%:*}"
        app_name="${app_entry#*:}"
        
        print_info "Building $app_name from $source_file..."
        ./build_single.sh "$source_file"
    done

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

    # Install each application
    for app_entry in "${APPLICATIONS[@]}"; do
        source_file="${app_entry%:*}"
        app_name="${app_entry#*:}"
        
        build_executable="build/${source_file}${SUFFIX}"
        install_path="$INSTALL_DIR/$app_name"
        
        if [ -f "$build_executable" ]; then
            cp "$build_executable" "$install_path"
            chmod +x "$install_path"
            print_info "Installed $app_name"
        else
            print_error "$app_name executable not found: $build_executable"
            exit 1
        fi
    done
}

# Verify installation
verify_installation() {
    print_info "Verifying installation..."

    for app_entry in "${APPLICATIONS[@]}"; do
        app_name="${app_entry#*:}"
        install_path="$INSTALL_DIR/$app_name"
        
        if [ -x "$install_path" ]; then
            print_info "$app_name installed successfully"
        else
            print_error "$app_name installation failed"
            exit 1
        fi
    done

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
    
    # List installed applications
    echo "Installed applications:"
    for app_entry in "${APPLICATIONS[@]}"; do
        app_name="${app_entry#*:}"
        echo "  - $app_name"
    done
    echo
    
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        print_warning "Note: $INSTALL_DIR is not in your PATH"
        echo "To use the installed tools from anywhere, add this line to your ~/.bashrc or ~/.zshrc:"
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
    echo "  $INSTALL_DIR/matgen seq 12345 1 1 10 | $INSTALL_DIR/matsum"
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
    echo "Applications to be installed:"
    for app_entry in "${APPLICATIONS[@]}"; do
        source_file="${app_entry%:*}"
        app_name="${app_entry#*:}"
        echo "  $source_file -> $app_name"
    done
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
    print_info "Starting installation of matrix tools..."
    print_info "Install directory: $INSTALL_DIR"
    print_info "Build type: $BUILD_TYPE"
    echo
    
    print_info "Applications to build and install:"
    for app_entry in "${APPLICATIONS[@]}"; do
        source_file="${app_entry%:*}"
        app_name="${app_entry#*:}"
        echo "  $source_file -> $app_name"
    done
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
