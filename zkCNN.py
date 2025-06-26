import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import numpy as np

# 1. Define a simple CNN (LeNet-like)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.fc1 = nn.Linear(6*26*26, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 6*26*26)
        x = self.fc1(x)
        return x

# 2. Generate random input and weights
model = SimpleCNN()
input_data = torch.randn(1, 1, 28, 28)
output = model(input_data)

print("=== Debug: Input Data ===")
print(input_data)
print("=== Debug: Model Weights (flattened) ===")
all_weights = torch.cat([p.flatten() for p in model.parameters()])
print(all_weights)

# 3. Commit to input and weights (using SHA256)
def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

input_commit = tensor_hash(input_data)
weights_commit = tensor_hash(all_weights)

print(f"Input Commitment (SHA256): {input_commit}")
print(f"Weights Commitment (SHA256): {weights_commit}")

print("=== Debug: Model Output ===")
print(output)

# 4. Prover sends (input_commit, weights_commit, output)
# 5. Verifier recomputes output and checks commitments
def verify(model, input_data, output, input_commit, weights_commit):
    print("\n[Verifier] Checking input commitment...")
    input_hash = tensor_hash(input_data)
    print(f"[Verifier] Computed input hash: {input_hash}")
    if input_hash != input_commit:
        print("[Verifier] Input commitment mismatch!")
        return False
    print("[Verifier] Input commitment matches.")

    print("[Verifier] Checking weights commitment...")
    weights_hash = tensor_hash(torch.cat([p.flatten() for p in model.parameters()]))
    print(f"[Verifier] Computed weights hash: {weights_hash}")
    if weights_hash != weights_commit:
        print("[Verifier] Weights commitment mismatch!")
        return False
    print("[Verifier] Weights commitment matches.")

    print("[Verifier] Checking model output...")
    expected_output = model(input_data)
    print(f"[Verifier] Expected output: {expected_output}")
    print(f"[Verifier] Provided output: {output}")
    if not torch.allclose(expected_output, output):
        print("[Verifier] Output mismatch!")
        return False
    print("[Verifier] Output matches.")
    return True

# Simulate proof and verification
is_valid = verify(model, input_data, output, input_commit, weights_commit)
print("\nProof valid?", is_valid)