#!/usr/bin/env python3
"""
Simple test script to check BLS12-381 library usage in WSL
"""

def test_bls12_381_in_wsl():
    """Test BLS12-381 library usage in WSL"""
    print("=== Testing BLS12-381 in WSL ===")
    
    try:
        # Import the main module
        import zkCNN_multi_models
        
        print(f"USE_REAL_BLS12_381: {zkCNN_multi_models.USE_REAL_BLS12_381}")
        
        if zkCNN_multi_models.USE_REAL_BLS12_381:
            print("🎉 SUCCESS: Using REAL BLS12-381 C++ library!")
            print("✅ The implementation is NOT using the Python fallback")
        else:
            print("⚠️  Using Python fallback implementation")
            print("❌ The implementation is using the Python fallback")
        
        # Test field creation
        from zkCNN_multi_models import BLS12_381_Field, BLS12_381_Group
        
        field = BLS12_381_Field()
        group = BLS12_381_Group()
        
        print(f"Field order: {field.get_field_order()}")
        
        # Test if using real implementation
        if hasattr(field, 'real_field') and field.real_field is not None:
            print("✅ Confirmed: Using REAL BLS12-381 C++ implementation!")
            return True
        else:
            print("❌ Confirmed: Using Python fallback implementation")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bls12_381_in_wsl()
    
    if success:
        print("\n🎉 SUCCESS: Real BLS12-381 library is working in WSL!")
        print("✅ The implementation is using the C++ library, not the Python fallback")
    else:
        print("\n❌ FAILED: Real BLS12-381 library test failed")
        print("⚠️  The implementation may be using the Python fallback")


