# ZKCNN Demo Guide - Using WSL

This guide explains how to run the LeNet and VGG demos for the zkCNN project using WSL (Windows Subsystem for Linux).

## Prerequisites

### 1. Install WSL
If you don't have WSL installed, follow these steps:

```powershell
# Open PowerShell as Administrator and run:
wsl --install
```

This will install Ubuntu by default. Restart your computer when prompted.

### 2. Install Required Packages in WSL
Open WSL terminal and run:

```bash
sudo apt-get update
sudo apt-get install cmake gcc g++ libgmp3-dev git
```

## Running the Demos

### Option 1: Using Original Shell Scripts (Recommended)

1. **Open WSL Terminal**
   - Press `Win + R`, type `wsl`, and press Enter
   - Or open Ubuntu from Start Menu

2. **Navigate to Project Directory**
   ```bash
   cd /mnt/c/Users/BhavishMohee/Desktop/Master\'s\ Dissertation/zkCNN_complete
   ```

3. **Run LeNet Demo**
   ```bash
   cd script
   bash demo_lenet.sh
   ```

4. **Run VGG Demo**
   ```bash
   bash demo_vgg.sh
   ```

### Option 2: Manual Build and Run

1. **Build the Project**
   ```bash
   cd /mnt/c/Users/BhavishMohee/Desktop/Master\'s\ Dissertation/zkCNN_complete
   mkdir -p cmake-build-release
   cd cmake-build-release
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --target demo_lenet_run -- -j 4
   cmake --build . --target demo_vgg_run -- -j 4
   ```

2. **Run LeNet Demo**
   ```bash
   cd ..
   ./cmake-build-release/src/demo_lenet_run \
     data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-images-weights-qint8.csv \
     data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv \
     output/single/lenet5.mnist.relu.max-1-infer.csv 1
   ```

3. **Run VGG Demo**
   ```bash
   ./cmake-build-release/src/demo_vgg_run \
     data/vgg11/vgg11.cifar.relu-1-images-weights-qint8.csv \
     data/vgg11/vgg11.cifar.relu-1-scale-zeropoint-uint8.csv \
     output/single/vgg11.cifar.relu-1-infer.csv \
     data/vgg11/vgg11-config.csv 1
   ```

## What Each Demo Does

### LeNet Demo
- **Model**: LeNet5 CNN architecture
- **Dataset**: MNIST (handwritten digits)
- **Purpose**: Demonstrates zero-knowledge proof for a simple CNN
- **Expected Runtime**: 1-5 minutes
- **Output**: 
  - `output/single/lenet5.mnist.relu.max-1-infer.csv` - Inference results
  - `output/single/demo-result-lenet5.txt` - Performance metrics

### VGG Demo
- **Model**: VGG11 CNN architecture  
- **Dataset**: CIFAR-10 (color images)
- **Purpose**: Demonstrates zero-knowledge proof for a larger CNN
- **Expected Runtime**: 10-30 minutes
- **Output**:
  - `output/single/vgg11.cifar.relu-1-infer.csv` - Inference results
  - `output/single/demo-result-vgg11.txt` - Performance metrics

## Understanding the Output

The demos generate zero-knowledge proofs for CNN inference. The output includes:

1. **Inference Results**: The actual CNN predictions
2. **Proof Generation Time**: How long it took to generate the ZK proof
3. **Verification Time**: How long it took to verify the proof
4. **Proof Size**: Size of the generated proof in KB
5. **Performance Metrics**: Various timing and efficiency measurements

## Troubleshooting

### Common Issues:

1. **"Permission denied"**
   ```bash
   chmod +x script/*.sh
   ```

2. **"CMake not found"**
   ```bash
   sudo apt-get install cmake
   ```

3. **"GMP not found"**
   ```bash
   sudo apt-get install libgmp3-dev
   ```

4. **"Build failed"**
   - Make sure you have enough disk space
   - Try building with fewer parallel jobs: `-j 2` instead of `-j 4`

5. **"Timeout expired"**
   - VGG demo can take 30+ minutes on slower machines
   - Be patient and let it complete

### Getting Help:

- Check the output files in `output/single/` for detailed error messages
- Look at the console output for specific error details
- Ensure all prerequisites are installed correctly

## Expected Results

When successful, you should see output like:

```
=== Running LeNet Demo ===
Running: ./cmake-build-release/src/demo_lenet_run data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-images-weights-qint8.csv data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv output/single/lenet5.mnist.relu.max-1-infer.csv 1

lenet (relu), 1, 45.23, 1024.5, 0.12, 0.08
```

The last line contains: model_name, pic_count, prove_time, proof_size_KB, verifier_time, verifier_slow_time

## Tips for WSL Usage

1. **File Access**: Your Windows files are accessible in WSL at `/mnt/c/Users/...`
2. **Performance**: WSL2 provides near-native Linux performance
3. **Terminal**: Use Windows Terminal for better WSL experience
4. **VS Code**: You can use VS Code with WSL extension for development

## Quick Commands Reference

```bash
# Check WSL version
wsl --version

# Update WSL
wsl --update

# Access WSL from Windows
wsl

# Exit WSL
exit

# Run specific command in WSL
wsl -e bash -c "cd /mnt/c/path/to/project && ./script/demo_lenet.sh"
```

