#!/bin/bash
# WSL script to test real BLS12-381 library usage

echo "=== Testing Real BLS12-381 Library in WSL ==="
echo "============================================================"

# Check if we're in WSL
if [[ -n "$WSL_DISTRO_NAME" ]]; then
    echo "✅ Running in WSL: $WSL_DISTRO_NAME"
else
    echo "❌ Not running in WSL"
    exit 1
fi

# Check for required files
echo "Checking for required files..."
if [ -f "bls12_381_ctypes.so" ]; then
    echo "✅ Found bls12_381_ctypes.so"
    ls -la bls12_381_ctypes.so
else
    echo "❌ Missing bls12_381_ctypes.so"
    exit 1
fi

if [ -f "zkCNN_multi_models.py" ]; then
    echo "✅ Found zkCNN_multi_models.py"
else
    echo "❌ Missing zkCNN_multi_models.py"
    exit 1
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import ctypes; print('✅ ctypes available')" || {
    echo "❌ ctypes not available"
    exit 1
}

# Test library loading
echo "Testing library loading..."
python3 -c "
import ctypes
try:
    lib = ctypes.CDLL('./bls12_381_ctypes.so')
    print('✅ Successfully loaded bls12_381_ctypes.so')
    
    # Test initialization
    if hasattr(lib, 'init_bls12_381'):
        lib.init_bls12_381()
        print('✅ Successfully called init_bls12_381()')
    else:
        print('❌ init_bls12_381 function not found')
        
except Exception as e:
    print(f'❌ Failed to load library: {e}')
    exit(1)
"

# Run the comprehensive test
echo "Running comprehensive test..."
python3 test_real_bls12_381_wsl.py

echo "=== Test completed ==="



