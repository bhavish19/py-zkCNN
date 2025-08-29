#!/usr/bin/env python3
"""
Usage Guide Demo: How to Use the Updated C++ Backend
This demo shows practical usage patterns for the C++ backend
"""

import sys
import os
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zkcnn_cpp_bindings import ZKCNNBackend

def demo_basic_usage():
    """Demo of basic C++ backend usage"""
    print("📚 Basic Usage Guide")
    print("=" * 40)
    
    # 1. Initialize the backend
    print("1. Initialize the C++ backend:")
    backend = ZKCNNBackend()
    print("   ✅ Backend initialized successfully")
    
    # 2. Field arithmetic operations
    print("\n2. Perform field arithmetic operations:")
    a, b = 12345, 67890
    result_add = backend.field_add(a, b)
    result_mul = backend.field_mul(a, b)
    result_inv = backend.field_inv(a)
    
    print(f"   Field addition: {a} + {b} = {result_add}")
    print(f"   Field multiplication: {a} × {b} = {result_mul}")
    print(f"   Field inversion: {a}⁻¹ = {result_inv}")
    
    # 3. Polynomial operations
    print("\n3. Perform polynomial operations:")
    coeffs = [1, 2, 3, 4, 5]
    x = 2
    
    # Evaluate polynomial
    result_eval = backend.poly_evaluate(coeffs, x)
    print(f"   Polynomial evaluation: P({x}) = {result_eval}")
    
    # Create polynomial commitment
    commitment = backend.poly_commit(coeffs)
    print(f"   Polynomial commitment: {commitment[:20]}...")
    
    # 4. Zero-knowledge proof generation
    print("\n4. Generate zero-knowledge proof:")
    input_data = b"Sample input data for ZK proof"
    model_type = "lenet"
    
    proof = backend.generate_proof(input_data, len(input_data), model_type)
    print(f"   ✅ Proof generated successfully")
    print(f"   Proof size: {proof.proof_size} bytes")
    print(f"   Layer commitments: {proof.num_layer_commitments}")
    
    # 5. Proof verification
    print("\n5. Verify zero-knowledge proof:")
    import hashlib
    input_hash = hashlib.sha256(input_data).hexdigest()
    is_valid = backend.verify_proof(proof, input_hash)
    print(f"   Verification result: {'VALID' if is_valid else 'INVALID'}")
    
    # 6. Cleanup
    print("\n6. Clean up resources:")
    backend.cleanup_proof(proof)
    backend.cleanup()
    print("   ✅ Resources cleaned up successfully")
    
    print("\n✅ Basic usage demonstration completed!")

def demo_advanced_usage():
    """Demo of advanced C++ backend usage patterns"""
    print("\n🔧 Advanced Usage Patterns")
    print("=" * 40)
    
    backend = ZKCNNBackend()
    
    # Pattern 1: Batch field operations
    print("1. Batch field operations:")
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    results = []
    
    start_time = time.time()
    for i in range(len(numbers) - 1):
        result = backend.field_add(numbers[i], numbers[i + 1])
        results.append(result)
    end_time = time.time()
    
    print(f"   Processed {len(results)} field additions in {end_time - start_time:.4f}s")
    print(f"   Results: {results}")
    
    # Pattern 2: Polynomial evaluation at multiple points
    print("\n2. Polynomial evaluation at multiple points:")
    coeffs = [1, 2, 3, 4, 5]
    points = [0, 1, 2, 3, 4, 5]
    evaluations = []
    
    for point in points:
        result = backend.poly_evaluate(coeffs, point)
        evaluations.append(result)
    
    print(f"   Polynomial: {coeffs}")
    print(f"   Evaluations at points {points}: {evaluations}")
    
    # Pattern 3: Multiple polynomial commitments
    print("\n3. Multiple polynomial commitments:")
    polynomials = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9, 10]
    ]
    
    commitments = []
    for poly in polynomials:
        commitment = backend.poly_commit(poly)
        commitments.append(commitment[:20] + "...")
    
    print(f"   Polynomials: {polynomials}")
    print(f"   Commitments: {commitments}")
    
    # Pattern 4: Proof generation for different model types
    print("\n4. Proof generation for different model types:")
    input_data = b"Test input data"
    model_types = ["lenet", "vgg16", "alexnet"]
    
    for model_type in model_types:
        proof = backend.generate_proof(input_data, len(input_data), model_type)
        print(f"   {model_type}: {proof.proof_size} bytes, {proof.num_layer_commitments} layers")
        backend.cleanup_proof(proof)
    
    backend.cleanup()
    print("\n✅ Advanced usage demonstration completed!")

def demo_error_handling():
    """Demo of error handling patterns"""
    print("\n⚠️ Error Handling Patterns")
    print("=" * 40)
    
    backend = ZKCNNBackend()
    
    # Handle invalid field operations
    print("1. Handling invalid field operations:")
    try:
        # Division by zero
        result = backend.field_inv(0)
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error caught: {e}")
    
    # Handle invalid polynomial evaluation
    print("\n2. Handling invalid polynomial evaluation:")
    try:
        result = backend.poly_evaluate([], 1)  # Empty coefficients
        print(f"   Result: {result}")
    except Exception as e:
        print(f"   Error caught: {e}")
    
    # Handle proof verification with wrong commitment
    print("\n3. Handling proof verification errors:")
    input_data = b"Test data"
    proof = backend.generate_proof(input_data, len(input_data), "lenet")
    
    # Wrong commitment
    wrong_commitment = "wrong_hash"
    is_valid = backend.verify_proof(proof, wrong_commitment)
    print(f"   Verification with wrong commitment: {'VALID' if is_valid else 'INVALID'}")
    
    backend.cleanup_proof(proof)
    backend.cleanup()
    print("\n✅ Error handling demonstration completed!")

def main():
    """Main usage guide demo"""
    print("📖 C++ Backend Usage Guide Demo")
    print("=" * 50)
    print("This demo shows how to use the updated C++ backend in practice")
    print("=" * 50)
    
    try:
        # Basic usage
        demo_basic_usage()
        
        # Advanced usage
        demo_advanced_usage()
        
        # Error handling
        demo_error_handling()
        
        print("\n🎉 Usage guide demonstration completed!")
        print("\n📋 Key Takeaways:")
        print("✅ Always initialize the backend before use")
        print("✅ Use field arithmetic for cryptographic operations")
        print("✅ Use polynomial operations for ZK proof components")
        print("✅ Generate and verify proofs for zero-knowledge functionality")
        print("✅ Always clean up resources after use")
        print("✅ Handle errors gracefully in production code")
        print("\n🔧 The C++ backend is now ready for production use!")
        
    except Exception as e:
        print(f"❌ Usage guide failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



