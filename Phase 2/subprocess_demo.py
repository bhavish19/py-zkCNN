#!/usr/bin/env python3
"""
Demo for the ZKCNN Subprocess Wrapper.
This shows how to use the existing C++ binaries from Python without building extensions.
"""

import os
import sys
from pathlib import Path

def convert_to_wsl_path(windows_path):
    """Convert Windows path to WSL path"""
    # Remove drive letter and convert backslashes
    path = windows_path.replace('\\', '/')
    if ':' in path:
        # Convert C:\path\to\file to /mnt/c/path/to/file
        drive, rest = path.split(':', 1)
        return f"/mnt/{drive.lower()}{rest}"
    return path

def check_binary():
    """Check if the C++ binary exists"""
    print("=== Checking C++ Binary ===")
    
    try:
        from zkcnn_subprocess_wrapper import check_binary_exists, get_binary_path
        
        if check_binary_exists():
            binary_path = get_binary_path()
            print(f"✅ C++ binary found: {binary_path}")
            return True
        else:
            print("❌ C++ binary not found")
            print("\nTo build the C++ binary:")
            print("1. Make sure you have WSL installed")
            print("2. Run: wsl -e bash -c \"cd /mnt/c/Users/BhavishMohee/Desktop/Master\\'s\\ Dissertation/zkCNN_complete && cd script && bash demo_lenet.sh\"")
            print("3. This will build the binaries automatically")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing wrapper: {e}")
        return False

def demo_lenet():
    """Demonstrate using the ZKCNNSubprocessWrapper class for LeNet"""
    print("\n=== LeNet Demo ===")
    
    try:
        from zkcnn_subprocess_wrapper import ZKCNNSubprocessWrapper
        
        # Create wrapper instance
        wrapper = ZKCNNSubprocessWrapper()
        print(f"✅ Wrapper created successfully")
        print(f"Binary path: {wrapper.binary_path}")
        
        # Use default file paths (same as C++ script)
        print("✅ Using default data files (same as C++ script)")
        print("Input file: data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-images-weights-qint8.csv")
        print("Config file: data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv")
        print("Output file: output/single/lenet5.mnist.relu.max-1-infer.csv")
        
        # Run the demo with default paths
        print("\nRunning LeNet demo...")
        result = wrapper.run_lenet_demo()
        
        # Display results
        print(f"\n=== LeNet Demo Results ===")
        print(f"Success: {result['success']}")
        if 'return_code' in result:
            print(f"Return code: {result['return_code']}")
        print(f"Execution time: {result['execution_time']:.4f} seconds")
        
        if result.get('stdout'):
            print(f"\n=== Output (last 10 lines) ===")
            lines = result['stdout'].strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
        
        if result.get('stderr'):
            print(f"\n=== Error Output ===")
            print(result['stderr'])
        
        # Check verification
        print(f"\n=== Verification ===")
        verification_result = wrapper.verify_proof(None, None, None, 1, "lenet")
        print(f"Proof verification: {'✅ Success' if verification_result else '❌ Failed'}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error in LeNet demo: {e}")
        return False

def demo_vgg():
    """Demonstrate using the ZKCNNSubprocessWrapper class for VGG"""
    print("\n=== VGG Demo ===")
    
    try:
        from zkcnn_subprocess_wrapper import ZKCNNSubprocessWrapper
        
        # Create wrapper instance
        wrapper = ZKCNNSubprocessWrapper()
        print(f"✅ Wrapper created successfully")
        print(f"Binary path: {wrapper.binary_path}")
        
        # Use default file paths (same as C++ script)
        print("✅ Using default data files (same as C++ script)")
        print("Input file: data/vgg11/vgg11.cifar.relu-1-images-weights-qint8.csv")
        print("Config file: data/vgg11/vgg11.cifar.relu-1-scale-zeropoint-uint8.csv")
        print("Network file: data/vgg11/vgg11-config.csv")
        print("Output file: output/single/vgg11.cifar.relu-1-infer.csv")
        
        # Run the demo with default paths
        print("\nRunning VGG demo...")
        result = wrapper.run_vgg_demo()
        
        # Display results
        print(f"\n=== VGG Demo Results ===")
        print(f"Success: {result['success']}")
        if 'return_code' in result:
            print(f"Return code: {result['return_code']}")
        print(f"Execution time: {result['execution_time']:.4f} seconds")
        
        if result.get('stdout'):
            print(f"\n=== Output (last 10 lines) ===")
            lines = result['stdout'].strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
        
        if result.get('stderr'):
            print(f"\n=== Error Output ===")
            print(result['stderr'])
        
        # Check verification
        print(f"\n=== Verification ===")
        verification_result = wrapper.verify_proof(None, None, None, 1, "vgg")
        print(f"Proof verification: {'✅ Success' if verification_result else '❌ Failed'}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error in VGG demo: {e}")
        return False

def demo_wrapper_class():
    """Demonstrate using the ZKCNNSubprocessWrapper class"""
    print("\n=== ZKCNNSubprocessWrapper Class Demo ===")
    
    # Run both demos
    lenet_success = demo_lenet()
    vgg_success = demo_vgg()
    
    return lenet_success and vgg_success

def demo_function_call():
    """Demonstrate using the simple function call"""
    print("\n=== Function Call Demo ===")
    
    try:
        from zkcnn_subprocess_wrapper import run_lenet_demo_simple, run_vgg_demo_simple
        
        print("Running LeNet demo with simple function call (using default files)...")
        lenet_result = run_lenet_demo_simple()
        
        print("Running VGG demo with simple function call (using default files)...")
        vgg_result = run_vgg_demo_simple()
        
        print(f"\n=== Function Call Results ===")
        print(f"LeNet Success: {lenet_result['success']}")
        print(f"LeNet Execution time: {lenet_result['execution_time']:.4f} seconds")
        print(f"VGG Success: {vgg_result['success']}")
        print(f"VGG Execution time: {vgg_result['execution_time']:.4f} seconds")
        
        return lenet_result['success'] and vgg_result['success']
        
    except Exception as e:
        print(f"❌ Error in function call demo: {e}")
        return False

def compare_approaches():
    """Compare different approaches"""
    print("\n=== Comparison of Approaches ===")
    
    print("Subprocess Wrapper (WSL):")
    print("✅ No compilation required")
    print("✅ Uses existing C++ binaries")
    print("✅ Full C++ performance")
    print("✅ Real cryptographic security")
    print("✅ Easy Python integration")
    print("✅ Works on Windows with WSL")
    print("⚠️ Requires WSL and C++ binary to be built")
    
    print("\nPython Extension Wrapper:")
    print("✅ Direct Python integration")
    print("✅ Full C++ performance")
    print("✅ Real cryptographic security")
    print("❌ Requires C++ compiler")
    print("❌ Complex build process")
    print("❌ Platform-specific compilation")
    
    print("\nPython-only Implementation:")
    print("✅ Easy to use and modify")
    print("✅ No compilation required")
    print("❌ No cryptographic security")
    print("❌ Slow performance")
    print("❌ Educational only")

def main():
    """Main demo function"""
    print("=== ZKCNN Subprocess Wrapper Demo ===")
    print("This demo shows how to use the C++ ZKCNN binaries from Python")
    print("without building Python extensions, using WSL.")
    print()
    
    # Check if binary exists
    if not check_binary():
        print("\n❌ C++ binary not found. Please build the C++ code first.")
        print("\nYou can still use the Python-only implementations:")
        print("- zkCNN.py (basic hash commitments)")
        print("- zkCNN_zk.py (educational ZK structure)")
        print("- zkCNN_advanced.py (advanced educational implementation)")
        return False
    
    # Demo 1: Using the wrapper class
    success1 = demo_wrapper_class()
    
    # Demo 2: Using the function call
    success2 = demo_function_call()
    
    # Demo 3: Comparison
    compare_approaches()
    
    print("\n=== Summary ===")
    print(f"Wrapper class demo: {'✅ Success' if success1 else '❌ Failed'}")
    print(f"Function call demo: {'✅ Success' if success2 else '❌ Failed'}")
    
    if success1 and success2:
        print("\n🎉 All demos completed successfully!")
        print("\nYou can now use the C++ ZKCNN implementation from Python!")
        print("\nExample usage:")
        print("from zkcnn_subprocess_wrapper import ZKCNNSubprocessWrapper")
        print("wrapper = ZKCNNSubprocessWrapper()")
        print("# Run LeNet demo")
        print("result = wrapper.run_lenet_demo('input.csv', 'config.csv', 'output.csv')")
        print("# Run VGG demo")
        print("result = wrapper.run_vgg_demo('input.csv', 'config.csv', 'output.csv', 'network.csv')")
        print("print(f'Success: {result[\"success\"]}')")
    else:
        print("\n❌ Some demos failed. Check the error messages above.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 