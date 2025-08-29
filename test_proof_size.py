#!/usr/bin/env python3

from zkCNN_multi_models import *

def test_proof_size():
    print("=== Testing Updated Proof Size Calculation ===")
    
    # Test LeNet
    print("1. Testing LeNet...")
    lenet_model = EnhancedZKCNN(ModelType.LENET, input_channels=1, num_classes=10)
    lenet_input = torch.randn(1, 1, 28, 28)
    lenet_output = lenet_model(lenet_input)
    lenet_proof = lenet_model.generate_zk_proof(lenet_input, lenet_output)
    
    print(f"LeNet proof size: {lenet_proof['proof_size_bytes']/1024:.2f} KB")
    print(f"LeNet layers: {len(lenet_proof['layer_commitments'])}")
    print(f"LeNet sumcheck proofs: {len(lenet_proof['sumcheck_proofs'])}")
    
    # Test VGG16
    print("\n2. Testing VGG16...")
    vgg16_model = EnhancedZKCNN(ModelType.VGG16, input_channels=3, num_classes=10)
    vgg16_input = torch.randn(1, 3, 32, 32)
    vgg16_output = vgg16_model(vgg16_input)
    vgg16_proof = vgg16_model.generate_zk_proof(vgg16_input, vgg16_output)
    
    print(f"VGG16 proof size: {vgg16_proof['proof_size_bytes']/1024:.2f} KB")
    print(f"VGG16 layers: {len(vgg16_proof['layer_commitments'])}")
    print(f"VGG16 sumcheck proofs: {len(vgg16_proof['sumcheck_proofs'])}")
    
    print("\n=== Comparison with C++ ===")
    print("Expected C++ sizes:")
    print("- LeNet: ~71 KB")
    print("- VGG16: ~304 KB")
    print("\nCurrent Python sizes:")
    print(f"- LeNet: {lenet_proof['proof_size_bytes']/1024:.2f} KB")
    print(f"- VGG16: {vgg16_proof['proof_size_bytes']/1024:.2f} KB")

if __name__ == "__main__":
    test_proof_size()
