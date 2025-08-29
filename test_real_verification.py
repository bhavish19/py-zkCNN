#!/usr/bin/env python3

from zkCNN_multi_models import *

def test_real_verification():
    print("=== Testing Real Verification with Sumcheck Protocol ===")
    
    # Create model and generate proof
    print("1. Creating LeNet model...")
    model = EnhancedZKCNN(ModelType.LENET, input_channels=1, num_classes=10)
    
    print("2. Generating input data...")
    input_data = torch.randn(1, 1, 28, 28)
    output = model(input_data)
    
    print("3. Generating zero-knowledge proof...")
    proof = model.generate_zk_proof(input_data, output)
    
    print("4. Creating input commitment...")
    input_commitment = model._commit_input(input_data)
    
    print("5. Running real verification with sumcheck...")
    result = model.verify_zk_proof(proof, input_commitment)
    
    print("\n=== Results ===")
    print(f"✅ Verification Result: {'VALID' if result else 'INVALID'}")
    print(f"⏱️  Prover Time: {proof['proof_time']:.4f} seconds")
    print(f"⏱️  Verifier Time: {proof['verify_time']:.4f} seconds")
    print(f"📦 Proof Size: {proof['proof_size_bytes']/1024:.2f} KB")
    
    # Verify proof structure
    print(f"\n📊 Proof Structure:")
    print(f"   - Layer commitments: {len(proof['layer_commitments'])}")
    print(f"   - Sumcheck proofs: {len(proof['sumcheck_proofs'])}")
    print(f"   - Model type: {proof['model_type']}")
    print(f"   - Input commitment: {len(proof['input_commitment'])} hex chars")
    
    return result

if __name__ == "__main__":
    success = test_real_verification()
    if success:
        print("\n🎉 All tests passed! Real verification with sumcheck is working correctly.")
    else:
        print("\n❌ Verification failed!")

