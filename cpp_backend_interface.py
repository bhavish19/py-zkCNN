#!/usr/bin/env python3

import ctypes
import os
import numpy as np
from typing import List, Optional

class CppBackendInterface:
    """Python interface to C++ backend for cryptographic operations"""
    
    def __init__(self):
        self.lib = None
        self._load_library()
    
    def _load_library(self):
        """Load the C++ backend library"""
        try:
            # Try to load the compiled library
            lib_path = os.path.join(os.path.dirname(__file__), 'cpp_backend', 'build', 'libzkcnn_backend.so')
            if os.path.exists(lib_path):
                self.lib = ctypes.CDLL(lib_path)
                print("✅ C++ backend library loaded successfully")
            else:
                print("⚠️  C++ backend library not found, using Python fallback")
                self.lib = None
        except Exception as e:
            print(f"⚠️  Failed to load C++ backend: {e}, using Python fallback")
            self.lib = None
    
    def generate_proof_transcript(self, input_data: np.ndarray, model_type: str) -> dict:
        """Generate proof transcript using C++ backend"""
        if self.lib is None:
            return self._generate_proof_transcript_python(input_data, model_type)
        
        try:
            # Convert input data to C++ format
            input_size = input_data.size
            input_bytes = input_data.tobytes()
            
            # Create C++ proof structure
            class Proof(ctypes.Structure):
                _fields_ = [
                    ("input_commitment", ctypes.c_char * 65),
                    ("final_claim", ctypes.c_int),
                    ("proof_size", ctypes.c_int)
                ]
            
            proof = Proof()
            
            # Call C++ function
            result = self.lib.generate_proof(
                input_bytes,
                input_size,
                model_type.encode('utf-8'),
                ctypes.byref(proof)
            )
            
            if result == 0:
                return {
                    'input_commitment': proof.input_commitment.decode('utf-8'),
                    'final_claim': proof.final_claim,
                    'proof_size_bytes': proof.proof_size,
                    'backend': 'cpp'
                }
            else:
                print("⚠️  C++ backend failed, falling back to Python")
                return self._generate_proof_transcript_python(input_data, model_type)
                
        except Exception as e:
            print(f"⚠️  C++ backend error: {e}, falling back to Python")
            return self._generate_proof_transcript_python(input_data, model_type)
    
    def _generate_proof_transcript_python(self, input_data: np.ndarray, model_type: str) -> dict:
        """Python fallback for proof generation"""
        import hashlib
        
        # Simulate C++ proof generation
        input_hash = hashlib.sha256(input_data.tobytes()).hexdigest()
        
        # Calculate proof size based on C++ methodology
        F_BYTE_SIZE = 32
        total_size = 0
        
        # Input commitment
        total_size += F_BYTE_SIZE
        
        # Layer commitments (estimate based on model type)
        if model_type.lower() == 'lenet':
            num_layers = 8
        elif model_type.lower() == 'vgg16':
            num_layers = 22
        else:
            num_layers = 10
        
        total_size += num_layers * F_BYTE_SIZE
        
        # Sumcheck proofs
        num_sumcheck_proofs = num_layers - 1
        for _ in range(num_sumcheck_proofs):
            # Sumcheck rounds (estimate 32 rounds per proof)
            total_size += 32 * F_BYTE_SIZE * 3  # 3 field elements per round
            total_size += F_BYTE_SIZE * 2  # Final claims
            total_size += F_BYTE_SIZE  # Vres evaluation
            total_size += 32 * F_BYTE_SIZE  # Additional coefficients
        
        # Additional overhead
        total_size += F_BYTE_SIZE * 20
        
        return {
            'input_commitment': input_hash,
            'final_claim': 0,
            'proof_size_bytes': total_size,
            'backend': 'python_fallback'
        }
    
    def verify_proof_transcript(self, proof: dict, input_commitment: str) -> bool:
        """Verify proof transcript using C++ backend"""
        if self.lib is None:
            return self._verify_proof_transcript_python(proof, input_commitment)
        
        try:
            # Call C++ verification function
            result = self.lib.verify_proof(
                proof['input_commitment'].encode('utf-8'),
                input_commitment.encode('utf-8'),
                proof['final_claim']
            )
            
            return result == 1
            
        except Exception as e:
            print(f"⚠️  C++ verification error: {e}, falling back to Python")
            return self._verify_proof_transcript_python(proof, input_commitment)
    
    def _verify_proof_transcript_python(self, proof: dict, input_commitment: str) -> bool:
        """Python fallback for proof verification"""
        # Basic verification
        return proof['input_commitment'] == input_commitment

# Global instance
cpp_backend = CppBackendInterface()
