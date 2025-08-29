#!/bin/bash

echo "Building BLS12-381 library for Python integration using WSL..."

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCL_INCLUDE="$SCRIPT_DIR/3rd/hyrax-bls12-381/3rd/mcl/include"
HYRAX_INCLUDE="$SCRIPT_DIR/3rd/hyrax-bls12-381/src"
MCL_LIB="$SCRIPT_DIR/3rd/hyrax-bls12-381/build/lib"
HYRAX_LIB="$SCRIPT_DIR/3rd/hyrax-bls12-381/build/src"

echo "Using MCL include: $MCL_INCLUDE"
echo "Using Hyrax include: $HYRAX_INCLUDE"
echo "Using MCL lib: $MCL_LIB"
echo "Using Hyrax lib: $HYRAX_LIB"

# Check if libraries exist
if [ ! -f "$MCL_LIB/libmcl.a" ]; then
    echo "Error: MCL library not found at $MCL_LIB/libmcl.a"
    exit 1
fi

if [ ! -f "$HYRAX_LIB/libhyrax_lib.a" ]; then
    echo "Error: Hyrax library not found at $HYRAX_LIB/libhyrax_lib.a"
    exit 1
fi

# Try to compile with g++ - dynamic linking for GMP
echo "Attempting compilation with g++ (dynamic GMP linking)..."

g++ -shared -fPIC -std=c++17 \
    -I"$MCL_INCLUDE" \
    -I"$HYRAX_INCLUDE" \
    -L"$MCL_LIB" \
    -L"$HYRAX_LIB" \
    -Wl,-Bstatic -lmcl -lhyrax_lib \
    -Wl,-Bdynamic -lgmp \
    -o bls12_381_ctypes.so \
    bls12_381_ctypes_wrapper_minimal.cpp

if [ $? -eq 0 ]; then
    echo "✅ Compilation successful!"
    echo "Library created: bls12_381_ctypes.so"
    
    # Check library dependencies
    echo "Checking library dependencies..."
    ldd bls12_381_ctypes.so
    
    echo ""
    echo "Next steps:"
    echo "1. Run: python3 bls12_381_ctypes_interface.py"
    echo "2. Run: python3 zkCNN_multi_models.py"
    echo ""
    echo "The Python code will now use real BLS12-381 operations from the C++ implementation!"
else
    echo "❌ Compilation failed with g++"
    echo "Trying with clang++..."
    
    clang++ -shared -fPIC -std=c++17 \
        -I"$MCL_INCLUDE" \
        -I"$HYRAX_INCLUDE" \
        -L"$MCL_LIB" \
        -L"$HYRAX_LIB" \
        -Wl,-Bstatic -lmcl -lhyrax_lib \
        -Wl,-Bdynamic -lgmp \
        -o bls12_381_ctypes.so \
        bls12_381_ctypes_wrapper_minimal.cpp
    
    if [ $? -eq 0 ]; then
        echo "✅ Compilation successful with clang++!"
        echo "Library created: bls12_381_ctypes.so"
        
        # Check library dependencies
        echo "Checking library dependencies..."
        ldd bls12_381_ctypes.so
        
        echo ""
        echo "Next steps:"
        echo "1. Run: python3 bls12_381_ctypes_interface.py"
        echo "2. Run: python3 zkCNN_multi_models.py"
        echo ""
        echo "The Python code will now use real BLS12-381 operations from the C++ implementation!"
    else
        echo "❌ Compilation failed with both g++ and clang++"
        echo "The Python code will use the fallback implementation."
        exit 1
    fi
fi
