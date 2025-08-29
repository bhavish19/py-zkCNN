#!/usr/bin/env python3

from zkCNN_multi_models import *

def test_verification():
    print("=== Testing Verification ===")
    
    # Create a simple model
    field = BLS12_381_Field()
    group = BLS12_381_Group()
    circuit = LayeredCircuit()
    circuit.init(2, 4)
    model = EnhancedZKCNN(ModelType.LENET, input_channels=1, num_classes=10)
    
    # Create test input
    input_data = torch.randn(1, 1, 28, 28)
    output = model(input_data)
    
    # Generate proof
    print("Generating proof...")
    proof = model.generate_zk_proof(input_data, output)
    
    # Verify proof
    print("Verifying proof...")
    input_commitment = model._commit_input(input_data)
    result = model.verify_zk_proof(proof, input_commitment)
    
    # Show result
    print(f"Verification result: {'✅ VALID' if result else '❌ INVALID'}")
    print(f"Proof time: {proof['proof_time']:.4f} seconds")
    print(f"Verify time: {proof['verify_time']:.4f} seconds")
    print(f"Proof size: {len(str(proof))} characters")
    
    return result

if __name__ == "__main__":
    test_verification()


