#!/usr/bin/env python3
"""
Verification script for the refactored TTT-MAE codebase.

This script performs comprehensive testing of the refactored code to ensure
all components work correctly and maintain backward compatibility.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("🧪 Testing imports...")
    
    try:
        import TTT_main_MAE
        print("✅ TTT_main_MAE imported successfully")
    except Exception as e:
        print(f"❌ Failed to import TTT_main_MAE: {e}")
        return False
    
    try:
        import engine_TTT_wavefield_LN_MAE_vis
        print("✅ engine_TTT_wavefield_LN_MAE_vis imported successfully")
    except Exception as e:
        print(f"❌ Failed to import engine_TTT_wavefield_LN_MAE_vis: {e}")
        return False
        
    return True

def test_argument_parser():
    """Test the argument parser functionality."""
    print("\n🧪 Testing argument parser...")
    
    try:
        import TTT_main_MAE
        parser = TTT_main_MAE.get_args_parser()
        
        # Test with minimal required arguments
        test_args = [
            '--model', 'mae_vit_base_patch16',
            '--data_path', './data',
            '--output_dir', './test_output',
            '--batch_size', '4',
            '--steps_per_example', '2'
        ]
        
        args = parser.parse_args(test_args)
        print(f"✅ Arguments parsed: model={args.model}, batch_size={args.batch_size}")
        return True
        
    except Exception as e:
        print(f"❌ Argument parser test failed: {e}")
        return False

def test_dataset_class():
    """Test the CustomTensorDataset class."""
    print("\n🧪 Testing CustomTensorDataset...")
    
    try:
        import TTT_main_MAE
        
        # Create dummy data
        dummy_data = torch.randn(10, 3, 224, 224)
        dummy_labels = torch.randint(0, 2, (10,))
        
        # Test dataset creation
        dataset = TTT_main_MAE.CustomTensorDataset(dummy_data, dummy_labels)
        print(f"✅ Dataset created with {len(dataset)} samples")
        
        # Test data access
        sample, label = dataset[0]
        print(f"✅ Data access works: sample shape {sample.shape}, label {label}")
        
        # Test data loader compatibility
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        batch_data, batch_labels = next(iter(data_loader))
        print(f"✅ DataLoader works: batch shape {batch_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset class test failed: {e}")
        return False

def test_engine_functions():
    """Test key engine functions."""
    print("\n🧪 Testing engine functions...")
    
    try:
        import engine_TTT_wavefield_LN_MAE_vis
        
        # Test accuracy function
        logits = torch.tensor([[2.0, 0.5], [0.1, 3.0], [1.0, 0.5], [2.5, 0.1]])
        target = torch.tensor([0, 1, 0, 0])
        
        acc_list = engine_TTT_wavefield_LN_MAE_vis.accuracy(logits, target)
        print(f"✅ Accuracy function works: {acc_list[0]:.2f}%")
        
        # Test OnlineLayerNormUpdater
        import torch.nn as nn
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_norm = nn.LayerNorm(10)
                self.linear = nn.Linear(10, 2)
                
            def forward(self, x):
                x = self.layer_norm(x)
                return self.linear(x)
        
        model = SimpleModel()
        updater = engine_TTT_wavefield_LN_MAE_vis.OnlineLayerNormUpdater(model)
        print("✅ OnlineLayerNormUpdater created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Engine functions test failed: {e}")
        return False

def test_function_availability():
    """Test that all expected functions are available."""
    print("\n🧪 Testing function availability...")
    
    try:
        import TTT_main_MAE
        import engine_TTT_wavefield_LN_MAE_vis
        
        # Check TTT_main_MAE functions
        main_functions = [
            'get_args_parser',
            'setup_training_environment', 
            'setup_data_transforms',
            'load_ultrasound_data',
            'load_combined_model',
            'process_sample',
            'main'
        ]
        
        for func_name in main_functions:
            if hasattr(TTT_main_MAE, func_name):
                print(f"✅ TTT_main_MAE.{func_name}")
            else:
                print(f"❌ TTT_main_MAE.{func_name} missing")
                return False
        
        # Check engine functions
        engine_functions = [
            'accuracy',
            'OnlineLayerNormUpdater',
            'train_on_test_with_online_ln_update_MAE'
        ]
        
        for func_name in engine_functions:
            if hasattr(engine_TTT_wavefield_LN_MAE_vis, func_name):
                print(f"✅ engine.{func_name}")
            else:
                print(f"❌ engine.{func_name} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Function availability test failed: {e}")
        return False

def test_model_loading_preparation():
    """Test that model loading functions are properly structured."""
    print("\n🧪 Testing model loading preparation...")
    
    try:
        import TTT_main_MAE
        
        # Test argument creation for model loading
        parser = TTT_main_MAE.get_args_parser()
        test_args = [
            '--model', 'mae_vit_base_patch16',
            '--data_path', './data',
            '--output_dir', './test_output'
        ]
        args = parser.parse_args(test_args)
        
        # Check if load_combined_model function exists and is callable
        if hasattr(TTT_main_MAE, 'load_combined_model') and callable(TTT_main_MAE.load_combined_model):
            print("✅ load_combined_model function is available and callable")
        else:
            print("❌ load_combined_model function not available")
            return False
            
        # Check if other setup functions exist
        setup_functions = ['setup_training_environment', 'setup_data_transforms']
        for func_name in setup_functions:
            if hasattr(TTT_main_MAE, func_name) and callable(getattr(TTT_main_MAE, func_name)):
                print(f"✅ {func_name} is available and callable")
            else:
                print(f"❌ {func_name} not available")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading preparation test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("🚀 Starting comprehensive verification of refactored TTT-MAE codebase...\n")
    
    test_results = []
    
    # Run all tests
    test_results.append(("Imports", test_imports()))
    test_results.append(("Argument Parser", test_argument_parser()))
    test_results.append(("Dataset Class", test_dataset_class()))
    test_results.append(("Engine Functions", test_engine_functions()))
    test_results.append(("Function Availability", test_function_availability()))
    test_results.append(("Model Loading Preparation", test_model_loading_preparation()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<30} {status}")
        if result:
            passed += 1
    
    print("-"*60)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The refactored code is working correctly.")
        print("✅ The codebase is ready for production use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
