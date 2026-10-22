#!/usr/bin/env python3
"""
Quick verification script for new model configurations.
"""

def main():
    try:
        # Test imports from unchain
        from unchain.schemas.models import GPT_4O, CLAUDE_SONNET_35, GEMINI_PRO_15
        print("✓ Successfully imported from unchain.schemas.models")
        
        # Test imports from miso
        from miso.schemas.models import GPT_4O as MISO_GPT_4O, CLAUDE_SONNET_35 as MISO_CLAUDE_SONNET_35, GEMINI_PRO_15 as MISO_GEMINI_PRO_15
        print("✓ Successfully imported from miso.schemas.models")
        
        # Test that models are the same
        assert GPT_4O.name == MISO_GPT_4O.name
        assert CLAUDE_SONNET_35.name == MISO_CLAUDE_SONNET_35.name
        assert GEMINI_PRO_15.name == MISO_GEMINI_PRO_15.name
        print("✓ Models are consistent between unchain and miso")
        
        # Test basic functionality
        models = [GPT_4O, CLAUDE_SONNET_35, GEMINI_PRO_15]
        
        for model in models:
            # Test serialization
            model_dict = model.to_dict()
            assert 'name' in model_dict
            assert 'capabilities' in model_dict
            assert 'default_payload' in model_dict
            
            # Test basic properties
            assert model.name is not None
            assert model.capabilities.provider is not None
            assert model.capabilities.max_context_window_tokens > 0
            assert isinstance(model.capabilities.supports_tools, bool)
            assert len(model.capabilities.input_modalities) > 0
            assert isinstance(model.default_payload.payload, dict)
            
            print(f"✓ {model.name} configuration is valid")
        
        print("\n🎉 All new model configurations are working correctly!")
        
        # Print summary
        print("\nModel Summary:")
        print("-" * 60)
        for model in models:
            print(f"{model.name:20} | {model.capabilities.provider:10} | {model.capabilities.max_context_window_tokens:>8,} tokens")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
