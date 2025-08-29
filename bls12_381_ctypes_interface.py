#!/usr/bin/env python3
"""
Python interface to BLS12-381 operations using ctypes
This provides real BLS12-381 cryptographic operations from the C++ implementation
"""

import ctypes
import os
import sys
from typing import List, Optional, Tuple
import platform

class BLS12_381_CTypes:
    """Python interface to BLS12-381 operations using ctypes"""
    
    def __init__(self, library_path: Optional[str] = None):
        """
        Initialize BLS12-381 interface
        
        Args:
            library_path: Path to the compiled library (optional)
        """
        self._load_library(library_path)
        self._initialize()
        
        # Storage for field and group element indices
        self.field_elements = []
        self.group_elements = []
    
    def _load_library(self, library_path: Optional[str] = None):
        """Load the BLS12-381 library"""
        if library_path is None:
            # Try to find the library automatically
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
        
        print(f"Trying to load library: {library_path}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Library exists: {os.path.exists(library_path)}")
        
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"BLS12-381 library not found: {library_path}")
        
        try:
            self.lib = ctypes.CDLL(library_path)
            print("✅ Library loaded successfully!")
            self._setup_function_signatures()
        except Exception as e:
            print(f"❌ Error loading library: {e}")
            raise RuntimeError(f"Failed to load BLS12-381 library: {e}")
    
    def _setup_function_signatures(self):
        """Setup function signatures for ctypes"""
        # Initialize function
        self.lib.init_bls12_381.argtypes = []
        self.lib.init_bls12_381.restype = None
        
        # Field element functions
        self.lib.create_field_element.argtypes = [ctypes.c_int64]
        self.lib.create_field_element.restype = ctypes.c_int64
        
        self.lib.field_add.argtypes = [ctypes.c_int64, ctypes.c_int64]
        self.lib.field_add.restype = ctypes.c_int64
        
        self.lib.field_mul.argtypes = [ctypes.c_int64, ctypes.c_int64]
        self.lib.field_mul.restype = ctypes.c_int64
        
        self.lib.field_sub.argtypes = [ctypes.c_int64, ctypes.c_int64]
        self.lib.field_sub.restype = ctypes.c_int64
        
        self.lib.field_inv.argtypes = [ctypes.c_int64]
        self.lib.field_inv.restype = ctypes.c_int64
        
        self.lib.field_random.argtypes = []
        self.lib.field_random.restype = ctypes.c_int64
        
        # Group element functions
        self.lib.create_group_element.argtypes = []
        self.lib.create_group_element.restype = ctypes.c_int64
        
        self.lib.group_scalar_mul.argtypes = [ctypes.c_int64, ctypes.c_int64]
        self.lib.group_scalar_mul.restype = ctypes.c_int64
        
        self.lib.group_add.argtypes = [ctypes.c_int64, ctypes.c_int64]
        self.lib.group_add.restype = ctypes.c_int64
        
        # String conversion functions
        self.lib.field_to_string.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_int]
        self.lib.field_to_string.restype = None
        
        self.lib.group_to_string.argtypes = [ctypes.c_int64, ctypes.c_char_p, ctypes.c_int]
        self.lib.group_to_string.restype = None
        
        self.lib.get_field_order.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.get_field_order.restype = None
        
        # Storage management functions
        self.lib.clear_storage.argtypes = []
        self.lib.clear_storage.restype = None
        
        self.lib.get_field_elements_count.argtypes = []
        self.lib.get_field_elements_count.restype = ctypes.c_int64
        
        self.lib.get_group_elements_count.argtypes = []
        self.lib.get_group_elements_count.restype = ctypes.c_int64
    
    def _initialize(self):
        """Initialize BLS12-381"""
        self.lib.init_bls12_381()
    
    def create_field_element(self, value: int) -> int:
        """Create a field element from an integer"""
        idx = self.lib.create_field_element(value)
        if idx == -1:
            raise RuntimeError("Failed to create field element")
        self.field_elements.append(idx)
        return idx
    
    def field_add(self, a: int, b: int) -> int:
        """Add two field elements"""
        idx = self.lib.field_add(a, b)
        if idx == -1:
            raise RuntimeError("Failed to add field elements")
        return idx
    
    def field_mul(self, a: int, b: int) -> int:
        """Multiply two field elements"""
        idx = self.lib.field_mul(a, b)
        if idx == -1:
            raise RuntimeError("Failed to multiply field elements")
        return idx
    
    def field_sub(self, a: int, b: int) -> int:
        """Subtract two field elements"""
        idx = self.lib.field_sub(a, b)
        if idx == -1:
            raise RuntimeError("Failed to subtract field elements")
        return idx
    
    def field_inv(self, a: int) -> int:
        """Compute multiplicative inverse of field element"""
        idx = self.lib.field_inv(a)
        if idx == -1:
            raise RuntimeError("Failed to compute field inverse")
        return idx
    
    def field_random(self) -> int:
        """Generate random field element"""
        idx = self.lib.field_random()
        if idx == -1:
            raise RuntimeError("Failed to generate random field element")
        return idx
    
    def create_group_element(self) -> int:
        """Create random group element"""
        idx = self.lib.create_group_element()
        if idx == -1:
            raise RuntimeError("Failed to create group element")
        self.group_elements.append(idx)
        return idx
    
    def group_scalar_mul(self, scalar: int, point: int) -> int:
        """Multiply group element by scalar"""
        idx = self.lib.group_scalar_mul(scalar, point)
        if idx == -1:
            raise RuntimeError("Failed to perform scalar multiplication")
        self.group_elements.append(idx)
        return idx
    
    def group_add(self, a: int, b: int) -> int:
        """Add two group elements"""
        idx = self.lib.group_add(a, b)
        if idx == -1:
            raise RuntimeError("Failed to add group elements")
        self.group_elements.append(idx)
        return idx
    
    def field_to_string(self, idx: int) -> str:
        """Convert field element to string"""
        buffer = ctypes.create_string_buffer(256)
        self.lib.field_to_string(idx, buffer, 256)
        return buffer.value.decode('utf-8')
    
    def group_to_string(self, idx: int) -> str:
        """Convert group element to string"""
        buffer = ctypes.create_string_buffer(256)
        self.lib.group_to_string(idx, buffer, 256)
        return buffer.value.decode('utf-8')
    
    def get_field_order(self) -> str:
        """Get field order as string"""
        buffer = ctypes.create_string_buffer(256)
        self.lib.get_field_order(buffer, 256)
        return buffer.value.decode('utf-8')
    
    def clear_storage(self):
        """Clear all stored elements"""
        self.lib.clear_storage()
        self.field_elements.clear()
        self.group_elements.clear()
    
    def get_field_elements_count(self) -> int:
        """Get number of field elements in storage"""
        return self.lib.get_field_elements_count()
    
    def get_group_elements_count(self) -> int:
        """Get number of group elements in storage"""
        return self.lib.get_group_elements_count()
    
    def get_stats(self) -> dict:
        """Get statistics about stored elements"""
        return {
            'field_elements': self.get_field_elements_count(),
            'group_elements': self.get_group_elements_count(),
            'field_order': self.get_field_order()
        }

# Convenience class for field arithmetic
class BLS12_381_Field:
    """Real BLS12-381 field arithmetic using C++ implementation"""
    
    def __init__(self, library_path: Optional[str] = None):
        self.bls = BLS12_381_CTypes(library_path)
        self._operation_count = 0
    
    def add(self, a: int, b: int) -> int:
        """Add two field elements"""
        self._operation_count += 1
        return self.bls.field_add(a, b)
    
    def mul(self, a: int, b: int) -> int:
        """Multiply two field elements"""
        self._operation_count += 1
        return self.bls.field_mul(a, b)
    
    def sub(self, a: int, b: int) -> int:
        """Subtract two field elements"""
        self._operation_count += 1
        return self.bls.field_sub(a, b)
    
    def inv(self, a: int) -> int:
        """Compute multiplicative inverse"""
        self._operation_count += 1
        return self.bls.field_inv(a)
    
    def random_element(self) -> int:
        """Generate random field element"""
        return self.bls.field_random()
    
    def create_element(self, value: int) -> int:
        """Create field element from integer"""
        return self.bls.create_field_element(value)
    
    def to_string(self, element: int) -> str:
        """Convert field element to string"""
        return self.bls.field_to_string(element)
    
    def get_field_order(self) -> str:
        """Get field order"""
        return self.bls.get_field_order()
    
    def get_stats(self) -> dict:
        """Get operation statistics"""
        stats = self.bls.get_stats()
        stats['total_operations'] = self._operation_count
        return stats

# Convenience class for group operations
class BLS12_381_Group:
    """Real BLS12-381 group operations using C++ implementation"""
    
    def __init__(self, library_path: Optional[str] = None):
        self.bls = BLS12_381_CTypes(library_path)
        self._operation_count = 0
    
    def random_generator(self) -> int:
        """Generate random group element"""
        self._operation_count += 1
        return self.bls.create_group_element()
    
    def scalar_mul(self, scalar: int, point: int) -> int:
        """Multiply group element by scalar"""
        self._operation_count += 1
        return self.bls.group_scalar_mul(scalar, point)
    
    def point_add(self, point1: int, point2: int) -> int:
        """Add two group elements"""
        self._operation_count += 1
        return self.bls.group_add(point1, point2)
    
    def to_string(self, element: int) -> str:
        """Convert group element to string"""
        return self.bls.group_to_string(element)
    
    def get_stats(self) -> dict:
        """Get operation statistics"""
        stats = self.bls.get_stats()
        stats['total_operations'] = self._operation_count
        return stats

# Demo function
def demo_bls12_381_ctypes():
    """Demo the BLS12-381 ctypes interface"""
    print("=== BLS12-381 CTypes Interface Demo ===")
    
    try:
        # Initialize field arithmetic
        field = BLS12_381_Field()
        group = BLS12_381_Group()
        
        print(f"Field order: {field.get_field_order()}")
        print()
        
        # Test field operations
        print("=== Field Operations ===")
        a = field.create_element(5)
        b = field.create_element(3)
        
        c = field.add(a, b)
        d = field.mul(a, b)
        e = field.sub(a, b)
        f = field.inv(a)
        
        print(f"a = {field.to_string(a)}")
        print(f"b = {field.to_string(b)}")
        print(f"a + b = {field.to_string(c)}")
        print(f"a * b = {field.to_string(d)}")
        print(f"a - b = {field.to_string(e)}")
        print(f"a^(-1) = {field.to_string(f)}")
        print()
        
        # Test group operations
        print("=== Group Operations ===")
        g1 = group.random_generator()
        g2 = group.random_generator()
        
        g3 = group.point_add(g1, g2)
        g4 = group.scalar_mul(a, g1)
        
        print(f"G1 = {group.to_string(g1)}")
        print(f"G2 = {group.to_string(g2)}")
        print(f"G1 + G2 = {group.to_string(g3)}")
        print(f"a * G1 = {group.to_string(g4)}")
        print()
        
        # Show statistics
        print("=== Statistics ===")
        print(f"Field operations: {field.get_stats()}")
        print(f"Group operations: {group.get_stats()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the BLS12-381 library is compiled and available")
        return False

if __name__ == "__main__":
    demo_bls12_381_ctypes()

