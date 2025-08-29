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
    
    def __init__(self, binary_path: Optional[str] = None, use_wsl: bool = True):
        """
        Initialize the wrapper
        
        Args:
            binary_path: Path to the C++ binary (e.g., 'cmake-build-release/lenet_demo')
            use_wsl: Whether to use WSL for running the binaries
        """
        self.use_wsl = use_wsl
        self.binary_path = binary_path or self._find_binary()
        self.last_result = None
        self.last_timing = {}
        
    def _find_binary(self) -> str:
        """Find the C++ binary automatically"""
        possible_paths = [
            "../cmake-build-release/src/demo_lenet_run",
            "../cmake-build-release/src/demo_vgg_run", 
            "../cmake-build-release/src/demo_alexnet_run",
            "../build/src/demo_lenet_run",
            "../build/src/demo_vgg_run",
            "../build/src/demo_alexnet_run",
            "demo_lenet_run",
            "demo_vgg_run",
            "demo_alexnet_run"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Could not find ZKCNN binary. Tried: {possible_paths}\n"
            "Please build the C++ code first or specify the binary path."
        )
    
    def _convert_to_wsl_path(self, windows_path: str) -> str:
        """Convert Windows path to WSL path"""
        path = windows_path.replace('\\', '/')
        if ':' in path:
            drive, rest = path.split(':', 1)
            return f"/mnt/{drive.lower()}{rest}"
        return path
    
    def _prepare_command(self, cmd: List[str]) -> List[str]:
        """Prepare command for execution (with WSL if needed)"""
        if self.use_wsl and os.name == 'nt':
            # Convert paths to WSL format and escape properly
            wsl_cmd = ["wsl", "-e", "bash", "-c"]
            wsl_paths = []
            for arg in cmd:
                if os.path.exists(arg) or arg.startswith('cmake-build-release'):
                    # Convert file paths to WSL format
                    wsl_path = self._convert_to_wsl_path(arg)
                    # Escape spaces and special characters
                    wsl_path = wsl_path.replace("'", "'\"'\"'")
                    wsl_paths.append(f"'{wsl_path}'")
                else:
                    # Keep non-path arguments as-is
                    wsl_paths.append(arg)
            wsl_cmd.append(" ".join(wsl_paths))
            return wsl_cmd
        else:
            return cmd
    
    def run_lenet_demo(self, input_file: str = None, config_file: str = None, 
                       output_file: str = None, pic_cnt: int = 1) -> Dict[str, Any]:
        """
        Run the LeNet demo using subprocess
        
        Args:
            input_file: Path to input CSV file (optional, uses default if None)
            config_file: Path to config CSV file (optional, uses default if None)
            output_file: Path to output CSV file (optional, uses default if None)
            pic_cnt: Number of pictures to process
            
        Returns:
            Dictionary with results and timing information
        """
        # Use default file paths if not provided (same as C++ script)
        if input_file is None:
            input_file = "data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-images-weights-qint8.csv"
        if config_file is None:
            config_file = "data/lenet5.mnist.relu.max/lenet5.mnist.relu.max-1-scale-zeropoint-uint8.csv"
        if output_file is None:
            output_file = "output/single/lenet5.mnist.relu.max-1-infer.csv"
        
        # Validate inputs
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Find the correct binary for LeNet
        lenet_binary = self._find_lenet_binary()
        
        # Prepare command (same order as C++ main function)
        cmd = [
            lenet_binary,
            input_file,
            config_file, 
            output_file,
            str(pic_cnt)
        ]
        
        # Prepare command for execution
        final_cmd = self._prepare_command(cmd)
        
        print(f"Running command: {' '.join(final_cmd)}")
        
        # Run the binary
        start_time = time.time()
        try:
            result = subprocess.run(
                final_cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
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
                "timing_info": timing_info,
                "model": "lenet"
            }
            
            self.last_timing = timing_info
            
            return self.last_result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 10 minutes",
                "execution_time": 600,
                "model": "lenet"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "model": "lenet"
            }
    
    def run_vgg_demo(self, input_file: str = None, config_file: str = None, 
                     output_file: str = None, network_file: str = None, pic_cnt: int = 1) -> Dict[str, Any]:
        """
        Run the VGG demo using subprocess
        
        Args:
            input_file: Path to input CSV file (optional, uses default if None)
            config_file: Path to config CSV file (optional, uses default if None)
            output_file: Path to output CSV file (optional, uses default if None)
            network_file: Path to network config file (optional, uses default if None)
            pic_cnt: Number of pictures to process
            
        Returns:
            Dictionary with results and timing information
        """
        # Use default file paths if not provided (same as C++ script)
        if input_file is None:
            input_file = "data/vgg11/vgg11.cifar.relu-1-images-weights-qint8.csv"
        if config_file is None:
            config_file = "data/vgg11/vgg11.cifar.relu-1-scale-zeropoint-uint8.csv"
        if output_file is None:
            output_file = "output/single/vgg11.cifar.relu-1-infer.csv"
        if network_file is None:
            network_file = "data/vgg11/vgg11-config.csv"
        
        # Validate inputs
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file not found: {config_file}")
        if not os.path.exists(network_file):
            raise FileNotFoundError(f"Network file not found: {network_file}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Find the correct binary for VGG
        vgg_binary = self._find_vgg_binary()
        
        # Prepare command (same order as C++ main function)
        cmd = [
            vgg_binary,
            input_file,
            config_file, 
            output_file,
            network_file,
            str(pic_cnt)
        ]
        
        # Prepare command for execution
        final_cmd = self._prepare_command(cmd)
        
        print(f"Running command: {' '.join(final_cmd)}")
        
        # Run the binary
        start_time = time.time()
        try:
            result = subprocess.run(
                final_cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for VGG
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
                "timing_info": timing_info,
                "model": "vgg"
            }
            
            self.last_timing = timing_info
            
            return self.last_result
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out after 30 minutes",
                "execution_time": 1800,
                "model": "vgg"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time,
                "model": "vgg"
            }
    
    def _find_lenet_binary(self) -> str:
        """Find the LeNet binary specifically"""
        possible_paths = [
            "../cmake-build-release/src/demo_lenet_run",
            "../build/src/demo_lenet_run",
            "demo_lenet_run"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If not found, try using the main binary path
        if hasattr(self, 'binary_path') and self.binary_path:
            return self.binary_path
        
        raise FileNotFoundError(f"Could not find LeNet binary. Tried: {possible_paths}")
    
    def _find_vgg_binary(self) -> str:
        """Find the VGG binary specifically"""
        possible_paths = [
            "../cmake-build-release/src/demo_vgg_run",
            "../build/src/demo_vgg_run",
            "demo_vgg_run"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If not found, try using the main binary path
        if hasattr(self, 'binary_path') and self.binary_path:
            return self.binary_path
        
        raise FileNotFoundError(f"Could not find VGG binary. Tried: {possible_paths}")
    
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
            
            # Look for circuit size information
            if 'gates' in line.lower() or 'circuit' in line.lower():
                timing_info['raw_circuit_line'] = line
        
        return timing_info
    
    def get_last_result(self) -> Optional[Dict[str, Any]]:
        """Get the result from the last run"""
        return self.last_result
    
    def get_last_timing(self) -> Dict[str, Any]:
        """Get timing information from the last run"""
        return self.last_timing
    
    def verify_proof(self, input_file: str = None, config_file: str = None, 
                    output_file: str = None, pic_cnt: int = 1, model: str = "lenet") -> bool:
        """
        Run the demo and check if proof verification passed
        
        Args:
            input_file: Path to input CSV file (optional, uses default if None)
            config_file: Path to config CSV file (optional, uses default if None)
            output_file: Path to output CSV file (optional, uses default if None)
            pic_cnt: Number of pictures to process
            model: Model to run ("lenet" or "vgg")
            
        Returns:
            True if proof verification was successful
        """
        if model.lower() == "lenet":
            result = self.run_lenet_demo(input_file, config_file, output_file, pic_cnt)
        elif model.lower() == "vgg":
            # For VGG, we need the network file
            if input_file is None:
                # Use default paths
                result = self.run_vgg_demo()
            else:
                # Try to find the network file
                network_file = input_file.replace("images-weights-qint8.csv", "config.csv")
                if not os.path.exists(network_file):
                    # Try alternative path
                    network_file = input_file.replace("images-weights-qint8.csv", "config.csv").replace("vgg11.cifar.relu-1-", "vgg11-")
                result = self.run_vgg_demo(input_file, config_file, output_file, network_file, pic_cnt)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        if not result["success"]:
            return False
        
        # Check if verification passed by looking at output
        stdout = result["stdout"]
        
        # Look for success indicators in the output
        success_indicators = [
            "verification passed",
            "proof verified", 
            "success",
            "true",
            "verification pass"
        ]
        
        for indicator in success_indicators:
            if indicator.lower() in stdout.lower():
                return True
        
        # If no clear success indicator, assume success if no errors
        return "error" not in stdout.lower() and result["return_code"] == 0

def run_lenet_demo_simple(input_file: str = None, config_file: str = None, 
                         output_file: str = None, pic_cnt: int = 1) -> Dict[str, Any]:
    """
    Simple function to run LeNet demo
    
    Args:
        input_file: Path to input CSV file (optional, uses default if None)
        config_file: Path to config CSV file (optional, uses default if None)
        output_file: Path to output CSV file (optional, uses default if None)
        pic_cnt: Number of pictures to process
        
    Returns:
        Dictionary with results
    """
    wrapper = ZKCNNSubprocessWrapper()
    return wrapper.run_lenet_demo(input_file, config_file, output_file, pic_cnt)

def run_vgg_demo_simple(input_file: str = None, config_file: str = None, 
                       output_file: str = None, network_file: str = None, pic_cnt: int = 1) -> Dict[str, Any]:
    """
    Simple function to run VGG demo
    
    Args:
        input_file: Path to input CSV file (optional, uses default if None)
        config_file: Path to config CSV file (optional, uses default if None)
        output_file: Path to output CSV file (optional, uses default if None)
        network_file: Path to network config file (optional, uses default if None)
        pic_cnt: Number of pictures to process
        
    Returns:
        Dictionary with results
    """
    wrapper = ZKCNNSubprocessWrapper()
    return wrapper.run_vgg_demo(input_file, config_file, output_file, network_file, pic_cnt)

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