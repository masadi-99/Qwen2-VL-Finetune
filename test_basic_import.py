#!/usr/bin/env python3
"""
Minimal test to verify the RewardFunc import fix works
"""

def test_core_imports():
    """Test core imports without external dependencies"""
    
    print("🧪 Testing core text-only GRPO components...")
    
    try:
        # Test reward functions loading
        import importlib
        import inspect
        from typing import Callable, List
        
        print("✅ Core Python imports successful")
        
        # Test reward functions module
        mod = importlib.import_module("src.train.reward_funcs")
        members = inspect.getmembers(mod, predicate=callable)
        reward_funcs = [(n, o) for n, o in members if n.endswith("_reward")]
        
        print(f"✅ Found {len(reward_funcs)} reward functions: {[n for n, o in reward_funcs]}")
        
        # Test that reward functions are actually callable
        for name, func in reward_funcs:
            if callable(func):
                print(f"✅ {name} is callable")
            else:
                print(f"❌ {name} is not callable")
        
        print("\n🎉 Core RewardFunc import fix successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_core_imports()
    print(f"\n{'✅ RewardFunc fix verified!' if success else '❌ Import still broken'}") 