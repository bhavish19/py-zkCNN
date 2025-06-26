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
            print("1. Make sure you have a C++ compiler installed")
            print("2. Run: mkdir cmake-build-release")
            print("3. Run: cd cmake-build-release")
            print("4. Run: cmake ..")
            print("5. Run: cmake --build .")
            return False
            
    except ImportError as e:
        print(f"❌ Error importing wrapper: {e}")
        return False

def demo_wrapper_class():
    """Demonstrate using the ZKCNNSubprocessWrapper class"""
    print("\n=== ZKCNNSubprocessWrapper Class Demo ===")
    
    try:
        from zkcnn_subprocess_wrapper import ZKCNNSubprocessWrapper
        
        # Create wrapper instance
        wrapper = ZKCNNSubprocessWrapper()
        print(f"✅ Wrapper created successfully")
        print(f"Binary path: {wrapper.binary_path}")
        
        # Define file paths (Windows format)
        current_dir = Path(__file__).parent
        input_file = str(current_dir / "data" / "lenet5.mnist.relu.max" / "lenet5.mnist.relu.max-1-images-weights-qint8.csv")
        config_file = str(current_dir / "data" / "lenet5.mnist.relu.max" / "lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv")
        output_file = str(current_dir / "output" / "single" / "lenet5.mnist.relu.max-1-infer.csv")
        
        # Convert to WSL paths
        input_file_wsl = convert_to_wsl_path(input_file)
        config_file_wsl = convert_to_wsl_path(config_file)
        output_file_wsl = convert_to_wsl_path(output_file)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Check if files exist
        if not os.path.exists(input_file) or not os.path.exists(config_file):
            print("❌ Input files not found. Please make sure the data is extracted.")
            return False
        
        print("✅ Input files found")
        print(f"Input file: {input_file}")
        print(f"Config file: {config_file}")
        print(f"Output file: {output_file}")
        print(f"WSL Input file: {input_file_wsl}")
        print(f"WSL Config file: {config_file_wsl}")
        print(f"WSL Output file: {output_file_wsl}")
        
        # Run the demo with WSL paths
        print("\nRunning LeNet demo...")
        result = wrapper.run_lenet_demo(input_file_wsl, config_file_wsl, output_file_wsl, 1)
        
        # Display results
        print(f"\n=== Demo Results ===")
        print(f"Success: {result['success']}")
        if 'return_code' in result:
            print(f"Return code: {result['return_code']}")
        print(f"Execution time: {result['execution_time']:.4f} seconds")
        
        if result.get('stdout'):
            print(f"\n=== Output ===")
            print(result['stdout'])
        
        if result.get('stderr'):
            print(f"\n=== Error Output ===")
            print(result['stderr'])
        
        # Check verification
        print(f"\n=== Verification ===")
        verification_result = wrapper.verify_proof(input_file_wsl, config_file_wsl, output_file_wsl, 1)
        print(f"Proof verification: {'✅ Success' if verification_result else '❌ Failed'}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error in demo: {e}")
        return False

def demo_function_call():
    """Demonstrate using the simple function call"""
    print("\n=== Function Call Demo ===")
    
    try:
        from zkcnn_subprocess_wrapper import run_lenet_demo_simple
        
        # Define file paths (Windows format)
        current_dir = Path(__file__).parent
        input_file = str(current_dir / "data" / "lenet5.mnist.relu.max" / "lenet5.mnist.relu.max-1-images-weights-qint8.csv")
        config_file = str(current_dir / "data" / "lenet5.mnist.relu.max" / "lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv")
        output_file = str(current_dir / "output" / "single" / "lenet5.mnist.relu.max-1-infer.csv")
        
        # Convert to WSL paths
        input_file_wsl = convert_to_wsl_path(input_file)
        config_file_wsl = convert_to_wsl_path(config_file)
        output_file_wsl = convert_to_wsl_path(output_file)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        print("Running LeNet demo with simple function call...")
        result = run_lenet_demo_simple(input_file_wsl, config_file_wsl, output_file_wsl, 1)
        
        print(f"\n=== Function Call Results ===")
        print(f"Success: {result['success']}")
        print(f"Execution time: {result['execution_time']:.4f} seconds")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Error in function call demo: {e}")
        return False

def compare_approaches():
    """Compare different approaches"""
    print("\n=== Comparison of Approaches ===")
    
    print("Subprocess Wrapper:")
    print("✅ No compilation required")
    print("✅ Uses existing C++ binaries")
    print("✅ Full C++ performance")
    print("✅ Real cryptographic security")
    print("✅ Easy Python integration")
    print("⚠️ Requires C++ binary to be built separately")
    
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
    print("without building Python extensions.")
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
        print("result = wrapper.run_lenet_demo('input.csv', 'config.csv', 'output.csv')")
        print("print(f'Success: {result[\"success\"]}')")
    else:
        print("\n❌ Some demos failed. Check the error messages above.")
    
    return success1 and success2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 