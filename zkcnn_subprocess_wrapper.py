#!/usr/bin/env python3
"""
ZKCNN Subprocess Wrapper
This wrapper calls the existing C++ binaries directly using subprocess,
avoiding the need to build Python extensions.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

class ZKCNNSubprocessWrapper:
    """Wrapper for ZKCNN C++ binaries using subprocess"""
    
    def __init__(self, binary_path: Optional[str] = None):
        """
        Initialize the wrapper
        
        Args:
            binary_path: Path to the C++ binary (e.g., 'cmake-build-release/lenet_demo')
        """
        self.binary_path = binary_path or self._find_binary()
        self.last_result = None
        self.last_timing = {}
        
    def _find_binary(self) -> str:
        """Find the C++ binary automatically"""
        possible_paths = [
            "build_linux/src/demo_lenet_run",
            "build_linux/src/demo_vgg_run", 
            "build_linux/src/demo_alexnet_run",
            "cmake-build-release/src/demo_lenet_run.exe",
            "cmake-build-release/src/demo_lenet_run",
            "build/lenet_demo.exe", 
            "build/lenet_demo",
            "lenet_demo.exe",
            "lenet_demo"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Could not find ZKCNN binary. Tried: {possible_paths}\n"
            "Please build the C++ code first or specify the binary path."
        )
    
    def run_lenet_demo(self, input_file: str, config_file: str, 
                       output_file: str, pic_cnt: int = 1) -> Dict[str, Any]:
        """
        Run the LeNet demo using subprocess
        
        Args:
            input_file: Path to input CSV file
            config_file: Path to config CSV file  
            output_file: Path to output CSV file
            pic_cnt: Number of pictures to process
            
        Returns:
            Dictionary with results and timing information
        """
        # Validate inputs
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Prepare command
        cmd = [
            self.binary_path,
            input_file,
            config_file, 
            output_file,
            str(pic_cnt)
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the binary
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            end_time = time.time()
            
            # Parse results
            success = result.returncode == 0
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            
            # Parse timing information from output
            timing_info = self._parse_timing_output(stdout)
            
            # Store results
            self.last_result = {
                "success": success,
                "return_code": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "execution_time": end_time - start_time,
                "timing_info": timing_info
            }
            
            self.last_timing = timing_info
            
            return self.last_result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 5 minutes",
                "execution_time": 300
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _parse_timing_output(self, stdout: str) -> Dict[str, Any]:
        """Parse timing information from the C++ binary output"""
        timing_info = {}
        
        # Look for timing patterns in the output
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for common timing patterns
            if 'time' in line.lower() or 'seconds' in line.lower():
                timing_info['raw_timing_line'] = line
            
            # Look for proof size information
            if 'kb' in line.lower() or 'size' in line.lower():
                timing_info['raw_size_line'] = line
        
        return timing_info
    
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the result from the last run"""
        return self.last_result
    
    def get_last_timing(self) -> Dict[str, Any]:
        """Get timing information from the last run"""
        return self.last_timing
    
    def verify_proof(self, input_file: str, config_file: str, 
                    output_file: str, pic_cnt: int = 1) -> bool:
        """
        Run the demo and check if proof verification passed
        
        Returns:
            True if proof verification was successful
        """
        result = self.run_lenet_demo(input_file, config_file, output_file, pic_cnt)
        
        if not result["success"]:
            return False
        
        # Check if verification passed by looking at output
        stdout = result["stdout"]
        
        # Look for success indicators in the output
        success_indicators = [
            "verification passed",
            "proof verified", 
            "success",
            "true"
        ]
        
        for indicator in success_indicators:
            if indicator.lower() in stdout.lower():
                return True
        
        # If no clear success indicator, assume success if no errors
        return "error" not in stdout.lower() and result["return_code"] == 0

def run_lenet_demo_simple(input_file: str, config_file: str, 
                         output_file: str, pic_cnt: int = 1) -> Dict[str, Any]:
    """
    Simple function to run LeNet demo
    
    Args:
        input_file: Path to input CSV file
        config_file: Path to config CSV file
        output_file: Path to output CSV file  
        pic_cnt: Number of pictures to process
        
    Returns:
        Dictionary with results
    """
    wrapper = ZKCNNSubprocessWrapper()
    return wrapper.run_lenet_demo(input_file, config_file, output_file, pic_cnt)

def check_binary_exists() -> bool:
    """Check if the C++ binary exists"""
    try:
        wrapper = ZKCNNSubprocessWrapper()
        return True
    except FileNotFoundError:
        return False

def get_binary_path() -> Optional[str]:
    """Get the path to the C++ binary if it exists"""
    try:
        wrapper = ZKCNNSubprocessWrapper()
        return wrapper.binary_path
    except FileNotFoundError:
        return None 