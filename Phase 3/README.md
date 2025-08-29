# Phase 3: ZKCNN Multi-Models Implementation

This folder contains the comprehensive multi-model ZKCNN implementation with advanced cryptographic features and performance monitoring.

## Contents

### Core Implementation Files
- **`zkCNN_multi_models.py`** - Main implementation with multi-model support (151KB, 3821 lines)
- **`bls12_381_ctypes_interface_simple.py`** - BLS12-381 cryptographic interface (12KB, 308 lines)

### Library Files
- **`bls12_381_ctypes.so`** - BLS12-381 library for Linux/WSL (62KB)
- **`bls12_381_ctypes_minimal.so`** - Minimal BLS12-381 library (62KB)
- **`bls12_381_ctypes_interface.py`** - Alternative BLS12-381 interface (13KB)
- **`bls12_381_wrapper.cpp`** - BLS12-381 wrapper source (4KB)
- **`bls12_381_ctypes_wrapper*.cpp`** - C++ wrapper source files (4-5KB each)
- **`libzkcnn.so`** - Core ZKCNN library (26KB)

### Build and Interface Files
- **`zkcnn_cpp_bindings.py`** - C++ bindings interface (12KB)
- **`cpp_backend_interface.py`** - C++ backend interface (5KB)
- **`CMakeLists_ctypes*.txt`** - Build configuration files

### Data Files
- **`data/lenet5.mnist.relu.max/`** - LeNet5 MNIST dataset files
  - `lenet5.mnist.relu.max-1-images-weights-qint8.csv` - Quantized images and weights
  - `lenet5.mnist.relu.max-1-labels-qint8.csv` - Quantized labels
  - `lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv` - Scale and zeropoint values

- **`data/vgg11/`** - VGG11 CIFAR dataset files
  - `vgg11.cifar.relu-1-images-weights-qint8.csv` - Quantized images and weights
  - `vgg11.cifar.relu-1-labels-int8.csv` - Quantized labels
  - `vgg11.cifar.relu-1-scale-zeropoint-uint8.csv` - Scale and zeropoint values
  - `vgg11-config.csv` - Network configuration

### Output Files
- **`output/single/`** - Output files from model execution
  - `lenet5.mnist.relu.max-1-infer.csv` - LeNet inference results
  - `vgg11.cifar.relu-1-infer.csv` - VGG11 inference results
  - `demo-result-lenet5.txt` - LeNet demo results
  - `demo-result-vgg11.txt` - VGG11 demo results

## Features

### Advanced Cryptographic Operations
- **BLS12-381 Field Arithmetic**: Complete finite field operations on the BLS12-381 scalar field
- **Real Polynomial Commitments**: Hyrax-style polynomial commitment scheme with elliptic curves
- **Complete GKR Protocol**: Full Goldwasser-Kalai-Rothblum protocol implementation
- **Multi-Phase Sumcheck**: Advanced sumcheck protocol with polynomial evaluations
- **Production-Grade Security**: Cryptographically secure zero-knowledge proofs

### Multi-Model Support
- **LeNet5**: Convolutional neural network for MNIST digit recognition
- **VGG16**: Deep convolutional network architecture for image classification
- **Extensible Architecture**: Easy addition of new CNN architectures

### Performance Monitoring
- **Comprehensive Metrics**: Prover time, verifier time, proof sizes
- **Real-time Monitoring**: Performance tracking during execution
- **JSON Output**: Detailed performance metrics saved to files
- **Memory Optimization**: Efficient memory usage for large models

### Advanced Features
- **Complex Circuit Representation**: Layered arithmetic circuits for different CNN architectures
- **Polynomial Evaluations**: Real polynomial operations with field arithmetic
- **Cryptographic Commitments**: Secure commitment schemes using elliptic curves
- **Zero-Knowledge Properties**: Complete privacy preservation

## Usage

### Prerequisites

#### Required Python Libraries
```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install numpy pandas
pip install cryptography

# Optional but recommended
pip install matplotlib seaborn
pip install jupyter notebook
```

#### System Requirements
- **Python**: 3.8 or higher
- **Memory**: At least 4GB RAM (8GB recommended for VGG16)
- **Storage**: 500MB free space
- **OS**: Windows, Linux, or macOS

### Running the Main Implementation

#### Method 1: Direct Python Execution (Windows/Linux/macOS)
```bash
# Navigate to Phase 3 directory
cd "Phase 3"

# Run the main implementation
python zkCNN_multi_models.py
```

#### Method 2: WSL (Windows Subsystem for Linux)
```bash
# Open WSL terminal
wsl

# Navigate to the project directory
cd /mnt/c/Users/BhavishMohee/Desktop/Master\'s\ Dissertation/zkCNN_complete/Phase\ 3

# Activate Python environment (if using conda/venv)
# conda activate your_env_name
# OR
# source venv/bin/activate

# Install dependencies in WSL
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas cryptography

# Run the implementation
python zkCNN_multi_models.py
```

#### Method 3: Virtual Environment Setup
```bash
# Create virtual environment
python -m venv zkcnn_env

# Activate environment
# Windows:
zkcnn_env\Scripts\activate
# Linux/macOS:
source zkcnn_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas cryptography

# Run implementation
python zkCNN_multi_models.py
```

### Expected Output
The script will run several demos and produce output similar to:
```
=== Testing Full Hyrax Protocol Implementation ===
✅ Created BLS12-381 field and group
✅ Created polynomial with 8 coefficients
✅ Created Hyrax prover

=== Polynomial Commitment ===
✅ Generated 2 commitments
📊 Prove time: 0.0028 seconds
📊 Proof size: 0.06 KB

=== Polynomial Evaluation ===
✅ Evaluated polynomial at [1, 0, 1]: 6

=== Hyrax Verification ===
✅ Hyrax verification successful!

=== Testing Full GKR Protocol Implementation ===
✅ Created BLS12-381 field and group
✅ Created layered circuit with 2 layers
✅ Created GKR prover
✅ Initialized GKR prover

=== ZKCNN Multi-Model Demo ===
✅ LeNet proof generated successfully!
✅ VGG16 proof generated successfully!
✅ All verifications passed!

📊 PERFORMANCE METRICS SUMMARY
⏱️  Total Runtime: 43.48 seconds
🔐 Prover Time: 0.1069 seconds
✅ Verifier Time: 0.0000 seconds
📦 Proof Sizes: LeNet 37.50 KB, VGG16 109.25 KB
```

### Key Functions
- `demo_zk_cnn_with_models()` - Compare different CNN architectures
- `demo_zk_cnn_with_real_data()` - Test with real MNIST and CIFAR data
- `ZKCNN.generate_zk_proof()` - Generate zero-knowledge proofs
- `ZKCNN.verify_zk_proof()` - Verify zero-knowledge proofs
- `PerformanceMetrics` - Comprehensive performance monitoring

### Troubleshooting

#### Common Issues and Solutions

**1. PyTorch Installation Issues**
```bash
# For CPU-only installation
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA support (if available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**2. Memory Issues with VGG16**
```bash
# Reduce batch size or use smaller models
# The script automatically handles memory optimization
```

**3. WSL Path Issues**
```bash
# Use proper path escaping in WSL
cd "/mnt/c/Users/BhavishMohee/Desktop/Master's Dissertation/zkCNN_complete/Phase 3"
```

**4. Library Import Errors**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt  # if available
# OR install manually:
pip install torch numpy pandas cryptography
```

**5. BLS12-381 Library Issues**
```bash
# The script automatically falls back to Python implementation
# if C++ libraries are not compatible with your system
```

## Performance Characteristics

### Proof Sizes
- **LeNet5**: ~37.50 KB proof size
- **VGG16**: ~109.25 KB proof size

### Security Features
- **Zero-Knowledge**: Prover doesn't reveal input data or model weights
- **Soundness**: Cryptographically secure proof verification
- **Completeness**: Valid proofs always verify successfully
- **Privacy**: Complete input and model privacy preservation

## Technical Details

### Field Arithmetic
- **Prime Field**: p = 2^251 + 17 * 2^192 + 1 (BLS12-381 scalar field)
- **Field Elements**: 32-byte representation
- **Operations**: Addition, multiplication, subtraction, inversion, exponentiation

### Circuit Representation
- **Layered Structure**: Each layer contains gates of specific types
- **Gate Types**: Input, Add, Multiply, ReLU, Convolution, Fully Connected, MaxPool
- **Bit Length**: Configurable bit length for each layer
- **Complex Operations**: Support for advanced CNN operations

### Proof Structure
- **Input Commitment**: Cryptographic commitment of input data
- **Layer Commitments**: Polynomial commitments for each layer
- **Sumcheck Proofs**: Multi-round interactive proof transcripts
- **Final Claims**: Verification claims for each layer
- **Performance Metrics**: Comprehensive timing and size measurements

## Dependencies

### Core Dependencies
- **PyTorch**: Neural network operations and tensor computations
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data loading and processing
- **Cryptography**: Advanced cryptographic operations

### Standard Library
- **hashlib**: Cryptographic hashing
- **secrets**: Cryptographically secure random number generation
- **dataclasses**: Data structure definitions
- **enum**: Enumeration types
- **time**: Performance timing
- **os**: Operating system interface
- **pathlib**: Path manipulation
- **json**: JSON serialization
- **csv**: CSV file handling

## File Structure

```
Phase 3/
├── README.md                                    # This file
├── zkCNN_multi_models.py                       # Main implementation
├── data/
│   ├── lenet5.mnist.relu.max/
│   │   ├── lenet5.mnist.relu.max-1-images-weights-qint8.csv
│   │   ├── lenet5.mnist.relu.max-1-labels-qint8.csv
│   │   └── lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv
│   └── vgg11/
│       ├── vgg11.cifar.relu-1-images-weights-qint8.csv
│       ├── vgg11.cifar.relu-1-labels-int8.csv
│       ├── vgg11.cifar.relu-1-scale-zeropoint-uint8.csv
│       └── vgg11-config.csv
└── output/
    └── single/
        ├── lenet5.mnist.relu.max-1-infer.csv
        ├── vgg11.cifar.relu-1-infer.csv
        ├── demo-result-lenet5.txt
        └── demo-result-vgg11.txt
```

## Performance Output

The implementation generates detailed performance metrics including:
- **Prover Time**: Time taken to generate zero-knowledge proofs
- **Verifier Time**: Time taken to verify proofs
- **Proof Sizes**: Size of generated proofs in KB
- **Total Runtime**: Overall execution time
- **Memory Usage**: Memory consumption during execution

Performance metrics are automatically saved to `zkcnn_performance_metrics.json` for analysis.

## Notes

- This implementation represents the most advanced version with complete cryptographic features
- All relative paths are configured to work from the Phase 3 directory
- The implementation includes comprehensive error handling and validation
- Performance monitoring is enabled by default and provides detailed insights
- The code is production-ready with proper security measures
- **Complete Self-Containment**: All library files included locally in Phase 3
- **Cross-Platform**: Automatically falls back to Python implementation if C++ libraries are not compatible
- **No External Dependencies**: Phase 3 can run independently without accessing parent directory
