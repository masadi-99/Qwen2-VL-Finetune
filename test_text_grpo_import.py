#!/usr/bin/env python3
"""
Test script to verify that the text-only GRPO trainer imports correctly
"""

def test_import():
    """Test that all components import without errors"""
    
    print("🧪 Testing text-only GRPO imports...")
    
    try:
        # Test reward functions import
        from src.utils import load_reward_funcs
        print("✅ load_reward_funcs imported successfully")
        
        # Test loading reward functions
        reward_funcs = load_reward_funcs("src.train.reward_funcs")
        print(f"✅ Loaded {len(reward_funcs)} reward functions: {[f.__name__ for f in reward_funcs]}")
        
        # Test text trainer import
        from src.trainer import QwenGRPOTextTrainer
        print("✅ QwenGRPOTextTrainer imported successfully")
        
        # Test text dataset import
        from src.dataset import GRPOTextDataset, make_grpo_text_data_module
        print("✅ GRPOTextDataset and make_grpo_text_data_module imported successfully")
        
        # Test parameter classes import
        from src.params import TextModelArguments, TextDataArguments, GRPOTextArguments
        print("✅ Text-only parameter classes imported successfully")
        
        print("\n🎉 All imports successful! Text-only GRPO is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_reward_function():
    """Test that reward functions work correctly"""
    
    print("\n🧪 Testing reward function calls...")
    
    try:
        from src.utils import load_reward_funcs
        
        # Load reward functions
        reward_funcs = load_reward_funcs("src.train.reward_funcs")
        
        # Test sample data
        sample_prompts = ["What is 2+2?", "Explain Python"]
        sample_completions = ["2+2 = 4", "Python is a programming language"]
        
        for reward_func in reward_funcs:
            try:
                # Test calling the reward function
                rewards = reward_func(
                    prompts=sample_prompts,
                    completions=sample_completions
                )
                print(f"✅ {reward_func.__name__} returned: {rewards}")
            except Exception as e:
                print(f"⚠️  {reward_func.__name__} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Reward function test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_import()
    if success:
        test_reward_function()
    
    print(f"\n{'🎯 Ready to train!' if success else '❌ Fix imports first'}") 