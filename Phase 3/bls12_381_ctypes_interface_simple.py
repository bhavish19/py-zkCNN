#!/usr/bin/env python3
"""
Simplified BLS12-381 ctypes interface
"""

import os
import ctypes
import platform
from typing import Optional

# Global library instance to avoid multiple loading
_global_lib = None

class BLS12_381_Field:
    """Simplified BLS12-381 Field implementation using ctypes"""
    
    def __init__(self, library_path: Optional[str] = None):
        self.field_elements = []
        self._load_library(library_path)
        self._initialize()
    
    def _load_library(self, library_path: Optional[str] = None):
        """Load the BLS12-381 library"""
        global _global_lib
        
        # Use global library if already loaded
        if _global_lib is not None:
            self.lib = _global_lib
            return
        
        if library_path is None:
            # Check for available library files (local directory first, then parent)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            
            possible_paths = [
                # Local directory paths (Phase 3)
                os.path.join(current_dir, "bls12_381_ctypes.dll"),  # Windows
                os.path.join(current_dir, "bls12_381_ctypes.so"),   # Linux/WSL
                os.path.join(current_dir, "bls12_381_ctypes_minimal.so"),  # Alternative Linux/WSL
                # Current directory
                "bls12_381_ctypes.dll",  # Windows
                "bls12_381_ctypes.so",   # Linux/WSL
                "bls12_381_ctypes_minimal.so",  # Alternative Linux/WSL
                "./bls12_381_ctypes.so",
                "./bls12_381_ctypes_minimal.so",
                # Parent directory paths (fallback)
                os.path.join(parent_dir, "bls12_381_ctypes.dll"),
                os.path.join(parent_dir, "bls12_381_ctypes.so"),
                os.path.join(parent_dir, "bls12_381_ctypes_minimal.so")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    library_path = path
                    break
            
            if library_path is None:
                # Default fallback (try local directory first, then parent)
                if platform.system() == "Windows":
                    local_path = os.path.join(current_dir, "bls12_381_ctypes.dll")
                    parent_path = os.path.join(parent_dir, "bls12_381_ctypes.dll")
                    if os.path.exists(local_path):
                        library_path = local_path
                    elif os.path.exists(parent_path):
                        library_path = parent_path
                    else:
                        library_path = "bls12_381_ctypes.dll"
                else:
                    local_path = os.path.join(current_dir, "bls12_381_ctypes.so")
                    parent_path = os.path.join(parent_dir, "bls12_381_ctypes.so")
                    if os.path.exists(local_path):
                        library_path = local_path
                    elif os.path.exists(parent_path):
                        library_path = parent_path
                    else:
                        library_path = "./bls12_381_ctypes.so"
        
        print(f"Trying to load library: {library_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Library exists: {os.path.exists(library_path)}")
        
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"BLS12-381 library not found: {library_path}")
        
        try:
            # Try direct loading first
            self.lib = ctypes.CDLL(library_path)
            print("✅ Library loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading library: {e}")
            # Try with absolute path
            try:
                abs_path = os.path.abspath(library_path)
                self.lib = ctypes.CDLL(abs_path)
                print(f"✅ Library loaded with absolute path: {abs_path}")
            except Exception as e2:
                print(f"❌ Error loading library with absolute path: {e2}")
                raise RuntimeError(f"Failed to load BLS12-381 library: {e}")
        
        # Store in global variable
        _global_lib = self.lib
    
    def _initialize(self):
        """Initialize BLS12-381"""
        try:
            self.lib.init_bls12_381()
            print("✅ BLS12-381 initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing BLS12-381: {e}")
            raise RuntimeError(f"Failed to initialize BLS12-381: {e}")
    
    def get_field_order(self) -> str:
        """Get the field order as a string"""
        try:
            buffer = ctypes.create_string_buffer(256)
            self.lib.get_field_order(buffer, 256)
            return buffer.value.decode('utf-8')
        except Exception as e:
            print(f"❌ Error getting field order: {e}")
            return "2147483647"  # Fallback
    
    def create_field_element(self, value: int) -> int:
        """Create a field element from an integer"""
        try:
            idx = self.lib.create_field_element(value)
            if idx == -1:
                raise RuntimeError("Failed to create field element")
            self.field_elements.append(idx)
            return idx
        except Exception as e:
            print(f"❌ Error creating field element: {e}")
            raise RuntimeError(f"Failed to create field element: {e}")
    
    def create_element(self, value: int) -> int:
        """Alias for create_field_element for compatibility"""
        return self.create_field_element(value)
    
    def field_add(self, a: int, b: int) -> int:
        """Add two field elements"""
        try:
            return self.lib.field_add(a, b)
        except Exception as e:
            print(f"❌ Error adding field elements: {e}")
            raise RuntimeError(f"Failed to add field elements: {e}")
    
    def field_mul(self, a: int, b: int) -> int:
        """Multiply two field elements"""
        try:
            return self.lib.field_mul(a, b)
        except Exception as e:
            print(f"❌ Error multiplying field elements: {e}")
            raise RuntimeError(f"Failed to multiply field elements: {e}")
    
    def field_sub(self, a: int, b: int) -> int:
        """Subtract two field elements"""
        try:
            return self.lib.field_sub(a, b)
        except Exception as e:
            print(f"❌ Error subtracting field elements: {e}")
            raise RuntimeError(f"Failed to subtract field elements: {e}")
    
    def field_inv(self, a: int) -> int:
        """Compute multiplicative inverse of field element"""
        try:
            return self.lib.field_inv(a)
        except Exception as e:
            print(f"❌ Error computing field inverse: {e}")
            raise RuntimeError(f"Failed to compute field inverse: {e}")
    
    def field_random(self) -> int:
        """Generate a random field element"""
        try:
            return self.lib.field_random()
        except Exception as e:
            print(f"❌ Error generating random field element: {e}")
            raise RuntimeError(f"Failed to generate random field element: {e}")
    
    # Compatibility methods for the main code
    def add(self, a: int, b: int) -> int:
        """Alias for field_add for compatibility"""
        return self.field_add(a, b)
    
    def mul(self, a: int, b: int) -> int:
        """Alias for field_mul for compatibility"""
        return self.field_mul(a, b)
    
    def sub(self, a: int, b: int) -> int:
        """Alias for field_sub for compatibility"""
        return self.field_sub(a, b)
    
    def inv(self, a: int) -> int:
        """Alias for field_inv for compatibility"""
        return self.field_inv(a)
    
    def random_element(self) -> int:
        """Alias for field_random for compatibility"""
        return self.field_random()

class BLS12_381_Group:
    """Simplified BLS12-381 Group implementation using ctypes"""
    
    def __init__(self, library_path: Optional[str] = None):
        self.group_elements = []
        self._load_library(library_path)
        self._initialize()
    
    def _load_library(self, library_path: Optional[str] = None):
        """Load the BLS12-381 library"""
        global _global_lib
        
        # Use global library if already loaded
        if _global_lib is not None:
            self.lib = _global_lib
            return
        
        if library_path is None:
            # Check for available library files
            possible_paths = [
                "bls12_381_ctypes.dll",  # Windows
                "bls12_381_ctypes.so",   # Linux/WSL
                "bls12_381_ctypes_minimal.so",  # Alternative Linux/WSL
                "./bls12_381_ctypes.so",
                "./bls12_381_ctypes_minimal.so"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    library_path = path
                    break
            
            if library_path is None:
                # Default fallback
                if platform.system() == "Windows":
                    library_path = "bls12_381_ctypes.dll"
                else:
                    library_path = "./bls12_381_ctypes.so"
        
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"BLS12-381 library not found: {library_path}")
        
        try:
            self.lib = ctypes.CDLL(library_path)
        except Exception as e:
            # Try with absolute path
            try:
                abs_path = os.path.abspath(library_path)
                print(f"✅ Library loaded with absolute path: {abs_path}")
                self.lib = ctypes.CDLL(abs_path)
            except Exception as e2:
                raise RuntimeError(f"Failed to load BLS12-381 library: {e}")
        
        # Store in global variable
        _global_lib = self.lib
    
    def _initialize(self):
        """Initialize BLS12-381"""
        try:
            self.lib.init_bls12_381()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BLS12-381: {e}")
    
    def create_group_element(self) -> int:
        """Create a group element"""
        try:
            idx = self.lib.create_group_element()
            if idx == -1:
                raise RuntimeError("Failed to create group element")
            self.group_elements.append(idx)
            return idx
        except Exception as e:
            raise RuntimeError(f"Failed to create group element: {e}")
    
    def group_scalar_mul(self, scalar: int, element: int) -> int:
        """Multiply a group element by a scalar"""
        try:
            return self.lib.group_scalar_mul(scalar, element)
        except Exception as e:
            raise RuntimeError(f"Failed to multiply group element by scalar: {e}")
    
    def group_add(self, a: int, b: int) -> int:
        """Add two group elements"""
        try:
            return self.lib.group_add(a, b)
        except Exception as e:
            raise RuntimeError(f"Failed to add group elements: {e}")
    
    def group_to_string(self, element: int) -> str:
        """Convert group element to string"""
        try:
            buffer = ctypes.create_string_buffer(256)
            self.lib.group_to_string(element, buffer, 256)
            return buffer.value.decode('utf-8')
        except Exception as e:
            return f"GroupElement({element})"
    
    def random_generator(self) -> int:
        """Generate a random group generator"""
        try:
            idx = self.lib.random_generator()
            if idx == -1:
                raise RuntimeError("Failed to create random generator")
            self.group_elements.append(idx)
            return idx
        except Exception as e:
            # Fallback: create a simple group element
            return self.create_group_element()
