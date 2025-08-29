# Phase 2: ZKCNN Subprocess Wrapper

This folder contains the subprocess wrapper implementation that interfaces with the existing C++ ZKCNN binaries without requiring Python extensions.

## Contents

### Core Implementation Files
- **`subprocess_demo.py`** - Demo script showing how to use the wrapper (9.5KB, 255 lines)
- **`zkcnn_subprocess_wrapper.py`** - Main wrapper class for C++ binaries (17KB, 454 lines)

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
- **`output/single/`** - Output files from C++ binaries
  - `lenet5.mnist.relu.max-1-infer.csv` - LeNet inference results
  - `vgg11.cifar.relu-1-infer.csv` - VGG11 inference results
  - `demo-result-lenet5.txt` - LeNet demo results
  - `demo-result-vgg11.txt` - VGG11 demo results

## Features

### Subprocess Integration
- **Direct C++ Binary Execution**: Calls existing C++ binaries via subprocess
- **WSL Support**: Automatic Windows-to-WSL path conversion
- **Cross-Platform**: Works on Windows, Linux, and macOS
- **No Extensions Required**: No need to build Python extensions

### Model Support
- **LeNet5**: MNIST digit recognition with quantized data
- **VGG11**: CIFAR image classification with quantized data
- **AlexNet**: Support for AlexNet architecture (if binary available)

### Data Handling
- **Quantized Data**: Supports 8-bit quantized neural networks
- **CSV Format**: Standard CSV input/output format
- **Scale/Zeropoint**: Proper handling of quantization parameters
- **Batch Processing**: Support for multiple images

## Usage

### Running the Demo
```bash
cd "Phase 2"
python subprocess_demo.py
```

### Using the Wrapper Class
```python
from zkcnn_subprocess_wrapper import ZKCNNSubprocessWrapper

# Create wrapper instance
wrapper = ZKCNNSubprocessWrapper()

# Run LeNet demo
result = wrapper.run_lenet_demo()
print(f"Success: {result['success']}")
print(f"Execution time: {result['execution_time']:.4f} seconds")

# Run VGG demo
result = wrapper.run_vgg_demo()
print(f"Success: {result['success']}")
print(f"Execution time: {result['execution_time']:.4f} seconds")
```

### Key Functions
- `run_lenet_demo()` - Run LeNet5 inference with default data
- `run_vgg_demo()` - Run VGG11 inference with default data
- `verify_proof()` - Check if proof verification passed
- `get_last_result()` - Get results from last execution
- `get_last_timing()` - Get timing information from last execution

## Dependencies

### External Dependencies
- **C++ Binaries**: Requires compiled C++ ZKCNN binaries
  - `../cmake-build-release/src/demo_lenet_run`
  - `../cmake-build-release/src/demo_vgg_run`
  - `../build/src/demo_lenet_run`
  - `../build/src/demo_vgg_run`

### Python Dependencies
- **Standard Library**: subprocess, os, sys, time, json, pathlib
- **No External Libraries**: Pure Python implementation

## Technical Details

### Binary Path Resolution
The wrapper automatically searches for C++ binaries in the following order:
1. `../cmake-build-release/src/demo_*_run`
2. `../build/src/demo_*_run`
3. Local directory `demo_*_run`

### WSL Integration
On Windows, the wrapper automatically:
- Converts Windows paths to WSL format
- Escapes special characters in paths
- Uses `wsl -e bash -c` for command execution

### Error Handling
- **Binary Not Found**: Clear error messages with build instructions
- **Execution Errors**: Captures stdout/stderr for debugging
- **Path Conversion**: Handles Windows/WSL path differences
- **Timeout Protection**: Prevents hanging on long operations

## File Structure

```
Phase 2/
├── README.md                                    # This file
├── subprocess_demo.py                           # Demo script
├── zkcnn_subprocess_wrapper.py                  # Main wrapper class
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

## Notes

- All file paths are relative to the Phase 2 directory
- C++ binaries are expected to be in `../cmake-build-release/src/` or `../build/src/`
- The implementation uses real quantized data for testing
- WSL is required for Windows users
- No Python extensions need to be built


