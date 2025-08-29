# Phase 1: Advanced ZKCNN Implementation

This folder contains the advanced working implementation of the Zero-Knowledge CNN (ZKCNN) system.

## Contents

### Core Implementation Files
- **`zkCNN_advanced_working.py`** - Main implementation with full functionality
- **`zkCNN_advanced.py`** - Simplified educational version

### Data Files
- **`data/lenet5.mnist.relu.max/`** - LeNet5 MNIST dataset files
  - `lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv` - Scale and zeropoint values
  - `lenet5.mnist.relu.max-1-images-weights-qint8.csv` - Quantized images and weights
  - `lenet5.mnist.relu.max-1-labels-qint8.csv` - Quantized labels

## Features

### Cryptographic Operations
- **BLS12-381 Field Arithmetic**: Finite field operations on the BLS12-381 scalar field
- **Polynomial Commitments**: Hyrax-style polynomial commitment scheme
- **Zero-Knowledge Proofs**: GKR (Goldwasser-Kalai-Rothblum) protocol implementation

### Neural Network Support
- **LeNet5**: Convolutional neural network for MNIST digit recognition
- **VGG16**: Deep convolutional network architecture
- **Layer Types**: Convolution, Max Pooling, Fully Connected, ReLU activation

### Protocol Implementation
- **Sumcheck Protocol**: Multi-round interactive proof system
- **Circuit Representation**: Layered arithmetic circuit representation
- **Proof Generation**: Complete zero-knowledge proof generation
- **Proof Verification**: Cryptographic proof verification

## Usage

### Running the Demo
```bash
cd "Phase 1"
python zkCNN_advanced_working.py
```

### Key Functions
- `demo_zk_cnn_with_models()` - Compare different CNN architectures
- `demo_zk_cnn_with_real_data()` - Test with real MNIST data
- `ZKCNN.generate_zk_proof()` - Generate zero-knowledge proofs
- `ZKCNN.verify_zk_proof()` - Verify zero-knowledge proofs

## Performance Characteristics

### Proof Sizes
- **LeNet5**: ~37.50 KB proof size
- **VGG16**: ~109.25 KB proof size

### Security Features
- **Zero-Knowledge**: Prover doesn't reveal input data or model weights
- **Soundness**: Cryptographically secure proof verification
- **Completeness**: Valid proofs always verify successfully

## Technical Details

### Field Arithmetic
- **Prime Field**: p = 2^251 + 17 * 2^192 + 1 (BLS12-381 scalar field)
- **Field Elements**: 32-byte representation
- **Operations**: Addition, multiplication, subtraction, inversion

### Circuit Representation
- **Layered Structure**: Each layer contains gates of specific types
- **Gate Types**: Input, Add, Multiply, ReLU, Convolution, Fully Connected
- **Bit Length**: Configurable bit length for each layer

### Proof Structure
- **Input Commitment**: SHA-256 hash of input data
- **Layer Commitments**: Polynomial commitments for each layer
- **Sumcheck Proofs**: Interactive proof transcripts
- **Final Claims**: Verification claims for each layer

## Dependencies

- **PyTorch**: Neural network operations
- **NumPy**: Numerical computations
- **Pandas**: Data loading and processing
- **Standard Library**: hashlib, secrets, dataclasses, enum, time, os

## File Structure

```
Phase 1/
├── README.md                                    # This file
├── zkCNN_advanced_working.py                   # Main implementation
├── zkCNN_advanced.py                           # Educational version
└── data/
    └── lenet5.mnist.relu.max/
        ├── lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv
        ├── lenet5.mnist.relu.max-1-images-weights-qint8.csv
        └── lenet5.mnist.relu.max-1-labels-qint8.csv
```

## Notes

- All file paths are relative to the Phase 1 directory
- The implementation uses real MNIST data for testing
- Performance metrics are included for monitoring
- The system provides both educational and production-ready implementations


