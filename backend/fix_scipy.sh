#!/bin/bash

# Fix scipy installation issues

echo "Fixing scipy installation..."

# Uninstall problematic packages
pip uninstall -y scipy numpy

# Reinstall with specific versions
pip install numpy==1.26.4
pip install scipy==1.13.1

# Verify installation
python3.11 -c "import scipy; import numpy; print('scipy and numpy installed successfully')" || {
    echo "Installation failed. Trying alternative approach..."
    pip install --no-cache-dir scipy numpy
}

echo "Done!"

