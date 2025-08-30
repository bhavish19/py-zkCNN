# py-zkCNN

# Install on Ubuntu 

```bash
git clone --recurse-submodules https://github.com/vio1etus/zkCNN_complete.git
sudo apt-get update
sudo apt-get install libgmp3-dev cmake gcc g++
```

## Introduction

This is the implementation of [zkCNN paper](https://eprint.iacr.org/2021/673), which is a GKR-based zero-knowledge proof for CNN reference, containing some common CNN models such as LeNet5, vgg11 and vgg16.

## Implementation Phases

This repository contains three distinct phases of py-zkCNN implementation, each with different features and complexity levels:

### Phase 1: Basic ZKCNN Implementation
**Location:** `Phase 1/zkCNN_advanced_working.py`

**Features:**
- Basic zero-knowledge proof implementation
- Support for LeNet5 and VGG11 models
- Simple cryptographic operations
- Basic performance monitoring

**How to Run:**
```bash
# Navigate to Phase 1 directory
cd "Phase 1"

# Install required dependencies
pip install torch torchvision numpy pandas

# Run the implementation
python zkCNN_advanced_working.py
```

**Expected Output:**
- Basic ZKCNN operations with LeNet5 and VGG11
- Simple performance metrics
- Model inference results

### Phase 2: Subprocess Wrapper Implementation
**Location:** `Phase 2/subprocess_demo.py`

**Features:**
- Enhanced subprocess wrapper for C++ integration
- Improved performance through C++ backend
- Advanced data processing capabilities
- Better error handling and logging

**How to Run:**
```bash
# Navigate to Phase 2 directory
cd "Phase 2"

# Install required dependencies
pip install torch torchvision numpy pandas subprocess32

# Run the implementation
python subprocess_demo.py
```

**Expected Output:**
- Enhanced ZKCNN with C++ backend integration
- Improved performance metrics
- Detailed logging and error handling

### Phase 3: Advanced Multi-Model Implementation
**Location:** `Phase 3/zkCNN_multi_models.py`

**Features:**
- Complete BLS12-381 cryptographic operations
- Real polynomial commitments using elliptic curves
- Full GKR protocol implementation
- Multi-phase sumcheck protocol
- Production-grade security features
- Comprehensive performance monitoring
- Self-contained with all dependencies

**How to Run:**

#### Method 1: Direct Python Execution
```bash
# Navigate to Phase 3 directory
cd "Phase 3"

# Install dependencies
pip install torch torchvision torchaudio numpy pandas cryptography

# Run the implementation
python zkCNN_multi_models.py
```

#### Method 2: Using Installation Scripts
```bash
# Navigate to Phase 3 directory
cd "Phase 3"

# For Windows:
install.bat

# For Linux/macOS/WSL:
chmod +x install.sh
./install.sh

# Run the implementation
python zkCNN_multi_models.py
```

#### Method 3: Virtual Environment Setup
```bash
# Navigate to Phase 3 directory
cd "Phase 3"

# Create virtual environment
python -m venv zkcnn_env

# Activate environment
# Windows:
zkcnn_env\Scripts\activate
# Linux/macOS:
source zkcnn_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run implementation
python zkCNN_multi_models.py
```

**Expected Output:**
```
=== Testing Full Hyrax Protocol Implementation ===
✅ Created BLS12-381 field and group
✅ Created polynomial with 8 coefficients
✅ Created Hyrax prover

=== Polynomial Commitment ===
✅ Generated 2 commitments
📊 Prove time: 0.0028 seconds
📊 Proof size: 0.06 KB

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

## Requirement

- C++14
- cmake >= 3.10
- GMP library

## Input Format

The input has two part which are data and weight in the matrix.

### Data Part

This part is the picture data, a vector reshaped from its original matrix by

![formula1](https://render.githubusercontent.com/render/math?math=ch_{in}%20%5Ccdot%20h\times%20w)

where ![formula2](https://render.githubusercontent.com/render/math?math=ch_{in}) is the number of channel, ![formula3](https://render.githubusercontent.com/render/math?math=h) is the height, ![formula4](https://render.githubusercontent.com/render/math?math=w) is the width.

### Weight Part

This part is the set of parameters in the neural network, which contains

- convolution kernel of size ![formula10](https://render.githubusercontent.com/render/math?math=ch_{out}%20\times%20ch_{in}%20\times%20m%20\times%20m)

  where ![formula11](https://render.githubusercontent.com/render/math?math=ch_{out}) and ![formula12](https://render.githubusercontent.com/render/math?math=ch_{in}) are the number of output and input channels, ![formula13](https://render.githubusercontent.com/render/math?math=m) is the sideness of the kernel (here we only support square kernel).

- convolution bias of size ![formula16](https://render.githubusercontent.com/render/math?math=ch_{out}).

- fully-connected kernel of size ![formula14](https://render.githubusercontent.com/render/math?math=ch_{out}\times%20ch_{in}).

- fully-connected bias of size ![formula15](https://render.githubusercontent.com/render/math?math=ch_{out}).

## Experiment Script
### Clone the repo
To run the code, make sure you clone with
```bash
git clone --recurse-submodules git@github.com:TAMUCrypto/zkCNN.git
```
since the polynomial commitment is included as a submodule.

### Run a demo of LeNet5
The script to run LeNet5 model (please run the script in ``script/`` directory).
```bash
./demo_lenet.sh
```

- The input data is in ``data/lenet5.mnist.relu.max/``.
- The experiment evaluation is ``output/single/demo-result-lenet5.txt``.
- The inference result is ``output/single/lenet5.mnist.relu.max-1-infer.csv``.

### Run a demo of vgg11
The script to run vgg11 model (please run the script in ``script/`` directory).
```bash
./demo_vgg.sh
```

- The input data is in ``data/vgg11/``.
- The experiment evaluation is ``output/single/demo-result.txt``.
- The inference result is ``output/single/vgg11.cifar.relu-1-infer.csv``.

## Polynomial Commitment

Here we implement a [hyrax polynomial commitment](https://eprint.iacr.org/2017/1132.pdf) based on BLS12-381 elliptic curve. It is a submodule and someone who is interested can refer to this repo [hyrax-bls12-381](https://github.com/TAMUCrypto/hyrax-bls12-381).

## Reference
- [zkCNN: Zero knowledge proofs for convolutional neural network predictions and accuracy](https://doi.org/10.1145/3460120.3485379).
  Liu, T., Xie, X., & Zhang, Y. (CCS 2021).

- [Doubly-efficient zksnarks without trusted setup](https://doi.org/10.1109/SP.2018.00060). Wahby, R. S., Tzialla, I., Shelat, A., Thaler, J., & Walfish, M. (S&P 2018)

- [Hyrax](https://github.com/hyraxZK/hyraxZK.git)

- [mcl](https://github.com/herumi/mcl)
