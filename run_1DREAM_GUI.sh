#!/bin/bash
# 1DREAM GUI Launcher for Linux/macOS/WSL
# Optimized for Ubuntu and other Debian-based distributions

set -e  # Exit on error

echo "========================================="
echo "  1DREAM Toolbox - Manifold Learning"
echo "========================================="
echo

# Determine the script directory (works with symlinks)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Function to check Python version
check_python_version() {
    local python_cmd="$1"
    if command -v "$python_cmd" &> /dev/null; then
        local version=$("$python_cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        local major=$(echo "$version" | cut -d. -f1)
        local minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 8 ]; then
            echo "$python_cmd"
            return 0
        fi
    fi
    return 1
}

# Find suitable Python interpreter
PYTHON_CMD=""
for cmd in python3 python python3.12 python3.11 python3.10 python3.9 python3.8; do
    if PYTHON_CMD=$(check_python_version "$cmd"); then
        break
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.8+ not found"
    echo "Please install Python 3.8 or higher:"
    echo "  Ubuntu/Debian: sudo apt-get install python3"
    echo "  Fedora:        sudo dnf install python3"
    echo "  Arch:          sudo pacman -S python"
    exit 1
fi

echo "Using Python: $($PYTHON_CMD --version)"

# Detect operating system
OS_TYPE="linux"
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS_TYPE="macos"
elif grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    OS_TYPE="wsl"
fi

# WSL-specific display configuration
if [ "$OS_TYPE" = "wsl" ]; then
    echo "WSL detected..."
    
    # Check if WSLg is available (WAYLAND_DISPLAY or built-in DISPLAY)
    if [ -n "$WAYLAND_DISPLAY" ] || [ -e "/mnt/wslg" ]; then
        echo "WSLg detected - GUI should work natively"
    elif [ -z "$DISPLAY" ]; then
        # For older WSL without WSLg, try to set DISPLAY for X server
        WIN_HOST=$(grep -m 1 nameserver /etc/resolv.conf 2>/dev/null | awk '{print $2}')
        if [ -n "$WIN_HOST" ]; then
            export DISPLAY="${WIN_HOST}:0.0"
            echo "Set DISPLAY=$DISPLAY (requires X server like VcXsrv on Windows)"
            echo ""
            echo "If the GUI doesn't appear, ensure you have an X server running:"
            echo "  1. Install VcXsrv or X410 on Windows"
            echo "  2. Run XLaunch with 'Disable access control' checked"
            echo ""
        fi
    fi
fi

# Linux-specific: Check for display server
if [ "$OS_TYPE" = "linux" ]; then
    if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
        echo "WARNING: No display server detected (DISPLAY and WAYLAND_DISPLAY are unset)"
        echo "The GUI requires a graphical environment (X11 or Wayland)"
        echo ""
    fi
fi

# Check if tkinter is available
if ! $PYTHON_CMD -c "import tkinter" 2>/dev/null; then
    echo "ERROR: tkinter not found"
    echo ""
    echo "Install tkinter for your system:"
    if [ "$OS_TYPE" = "macos" ]; then
        echo "  brew install python-tk"
    else
        echo "  Ubuntu/Debian: sudo apt-get install python3-tk"
        echo "  Fedora:        sudo dnf install python3-tkinter"
        echo "  Arch:          sudo pacman -S tk"
    fi
    exit 1
fi

# Check for required Python packages
echo "Checking Python dependencies..."
MISSING_PACKAGES=""
for pkg in numpy pandas scipy sklearn networkx matplotlib; do
    if ! $PYTHON_CMD -c "import $pkg" 2>/dev/null; then
        if [ "$pkg" = "sklearn" ]; then
            MISSING_PACKAGES="$MISSING_PACKAGES scikit-learn"
        else
            MISSING_PACKAGES="$MISSING_PACKAGES $pkg"
        fi
    fi
done

if [ -n "$MISSING_PACKAGES" ]; then
    echo ""
    echo "WARNING: Missing Python packages:$MISSING_PACKAGES"
    echo "Install with: pip install$MISSING_PACKAGES"
    echo ""
fi

# Check if LAAT_MBMS is installed
if ! $PYTHON_CMD -c "import LAAT_MBMS" 2>/dev/null; then
    echo "WARNING: LAAT_MBMS module not found"
    echo "LAAT and MBMS algorithms will not be available."
    echo "To install: cd LAAT_MBMS && pip install ."
    echo ""
fi

# Launch the GUI
echo "Starting 1DREAM GUI..."
echo ""

$PYTHON_CMD 1DREAM_GUI.py

exit_code=$?
if [ $exit_code -ne 0 ]; then
    echo ""
    echo "An error occurred (exit code: $exit_code)"
    echo "Check that all dependencies are installed:"
    echo "  pip install numpy pandas scipy scikit-learn networkx matplotlib"
    echo ""
fi

exit $exit_code
