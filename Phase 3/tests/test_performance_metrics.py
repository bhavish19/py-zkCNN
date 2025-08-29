#!/usr/bin/env python3
"""
Test script to demonstrate performance metrics integration in zkCNN_multi_models.py
"""

import time
import json

def test_performance_metrics():
    """Test the performance metrics integration"""
    print("=== Testing Performance Metrics Integration ===")
    print("=" * 60)
    
    try:
        # Import the main module
        import zkCNN_multi_models
        
        print("✅ Successfully imported zkCNN_multi_models")
        print(f"📊 BLS12-381 Configuration: USE_REAL_BLS12_381 = {zkCNN_multi_models.USE_REAL_BLS12_381}")
        
        # Test basic field operations with metrics
        print("\n🔢 Testing Field Operations with Metrics:")
        from zkCNN_multi_models import BLS12_381_Field, BLS12_381_Group
        
        field = BLS12_381_Field()
        group = BLS12_381_Group()
        
        # Perform some field operations
        a = 5
        b = 3
        
        print(f"   Testing field operations: {a} + {b}, {a} * {b}, {a} - {b}, {a}^(-1)")
        
        c = field.add(a, b)
        d = field.mul(a, b)
        e = field.sub(a, b)
        f = field.inv(a)
        
        print(f"   Results: {a}+{b}={c}, {a}*{b}={d}, {a}-{b}={e}, {a}^(-1)={f}")
        
        # Test polynomial operations with metrics
        print("\n📈 Testing Polynomial Operations with Metrics:")
        from zkCNN_multi_models import Polynomial
        
        coeffs = [1, 2, 3, 4, 5, 6, 7, 8]
        poly = Polynomial(coeffs, field)
        print(f"   Created polynomial with {len(coeffs)} coefficients")
        
        # Test polynomial evaluation
        x = 2
        result = poly.evaluate(x)
        print(f"   Evaluated polynomial at x={x}: {result}")
        
        # Test Hyrax protocol with metrics
        print("\n🔐 Testing Hyrax Protocol with Metrics:")
        from zkCNN_multi_models import HyraxPolyCommitment
        
        hyrax_prover = HyraxPolyCommitment(field, group)
        commitments = hyrax_prover.commit(poly)
        print(f"   Generated {len(commitments)} commitments")
        
        # Test evaluation
        x_point = [1, 0, 1]
        evaluation = hyrax_prover.evaluate(x_point)
        print(f"   Evaluated polynomial at {x_point}: {evaluation}")
        
        # Test GKR protocol with metrics
        print("\n🧮 Testing GKR Protocol with Metrics:")
        from zkCNN_multi_models import FullGKRProver, LayeredCircuit
        
        # Create a simple circuit
        circuit = LayeredCircuit()
        circuit.init(220, 2)
        
        gkr_prover = FullGKRProver(circuit, field, group)
        gkr_prover.init()
        print("   Initialized GKR prover")
        
        # Test sumcheck initialization
        r_0_from_v = [field.random_element() for _ in range(4)]
        gkr_prover.sumcheck_init_all(r_0_from_v)
        print("   Initialized sumcheck for all layers")
        
        # Print current metrics
        print("\n📊 Current Performance Metrics:")
        zkCNN_multi_models.performance_metrics.print_summary()
        
        # Save metrics to file
        zkCNN_multi_models.performance_metrics.save_to_file("test_performance_metrics.json")
        
        print("\n✅ Performance metrics test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_performance_data():
    """Analyze the performance data from the JSON file"""
    try:
        with open("test_performance_metrics.json", "r") as f:
            data = json.load(f)
        
        print("\n📊 PERFORMANCE DATA ANALYSIS")
        print("=" * 60)
        
        print(f"⏱️  Total Runtime: {data['total_runtime']:.4f} seconds")
        
        # Field operations analysis
        field_ops = data['field_operations']
        print(f"\n🔢 Field Operations Analysis:")
        print(f"   Total Operations: {field_ops['total_count']}")
        print(f"   Total Time: {field_ops['total_time']:.6f} seconds")
        
        for op, metrics in field_ops['operations'].items():
            if metrics['count'] > 0:
                print(f"   {op.upper()}:")
                print(f"     - Count: {metrics['count']}")
                print(f"     - Average Time: {metrics['avg_time_ms']:.6f} ms")
                print(f"     - Operations/Second: {metrics['ops_per_second']:.0f}")
        
        # Protocol operations analysis
        protocol_ops = data['protocol_operations']
        print(f"\n🔐 Protocol Operations Analysis:")
        print(f"   Total Operations: {protocol_ops['total_count']}")
        print(f"   Total Time: {protocol_ops['total_time']:.6f} seconds")
        
        for op, metrics in protocol_ops['operations'].items():
            if metrics['count'] > 0:
                print(f"   {op.upper()}:")
                print(f"     - Count: {metrics['count']}")
                print(f"     - Average Time: {metrics['avg_time_ms']:.4f} ms")
        
        # Proof sizes analysis
        proof_sizes = data['proof_sizes']
        print(f"\n📦 Proof Sizes Analysis:")
        print(f"   Total Proofs: {proof_sizes['total_proofs']}")
        print(f"   Hyrax Average: {proof_sizes['hyrax_avg_kb']:.2f} KB")
        print(f"   GKR Average: {proof_sizes['gkr_avg_kb']:.2f} KB")
        print(f"   Overall Average: {proof_sizes['total_avg_kb']:.2f} KB")
        
        # Memory usage analysis
        memory = data['memory_usage']
        print(f"\n💾 Memory Usage Analysis:")
        for component, count in memory.items():
            print(f"   {component.replace('_', ' ').title()}: {count}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Performance metrics file not found. Run the test first.")
        return False
    except Exception as e:
        print(f"❌ Error analyzing performance data: {e}")
        return False

if __name__ == "__main__":
    # Run the performance metrics test
    success = test_performance_metrics()
    
    if success:
        # Analyze the performance data
        analyze_performance_data()
        
        print("\n🎉 Performance metrics integration test completed!")
        print("📊 Check 'test_performance_metrics.json' for detailed metrics")
    else:
        print("\n❌ Performance metrics integration test failed!")


