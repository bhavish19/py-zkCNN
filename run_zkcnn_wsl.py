#!/usr/bin/env python3
"""
Run zkCNN with real BLS12-381 implementation from WSL
This script should be run from within WSL environment
"""

import os
import sys
import platform

def main():
    print("=== Running zkCNN with Real BLS12-381 Implementation ===")
    print(f"Platform: {platform.system()}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check if we're in WSL/Linux
    if platform.system() != "Linux":
        print("❌ This script should be run from within WSL/Linux environment")
        print("Please run: wsl python3 run_zkcnn_wsl.py")
        return
    
    # Check if the BLS12-381 library exists
    library_name = "bls12_381_ctypes.so"
    if not os.path.exists(library_name):
        print(f"❌ BLS12-381 library not found: {library_name}")
        print("Please build the library first using: bash build_bls12_381_wsl.sh")
        return
    
    print(f"✅ Found BLS12-381 library: {library_name}")
    
    # Import and run the main zkCNN module
    try:
        print("\nImporting zkCNN_multi_models...")
        from zkCNN_multi_models import demo_multi_model_zkcnn
        
        print("✅ Successfully imported zkCNN_multi_models")
        print("\nRunning zkCNN demo with real BLS12-381 implementation...")
        
        # Run the demo
        demo_multi_model_zkcnn()
        
        print("\n✅ zkCNN demo completed successfully!")
        print("The real BLS12-381 implementation was used for cryptographic operations.")
        
    except Exception as e:
        print(f"❌ Error running zkCNN: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()







