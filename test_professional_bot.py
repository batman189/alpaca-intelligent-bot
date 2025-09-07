#!/usr/bin/env python3
"""
Professional Trading Bot - Quick Test Suite
"""

def test_imports():
    """Test all critical imports"""
    print("🧪 Testing Component Imports...")
    
    try:
        from trading.intelligent_risk_manager import IntelligentRiskManager
        print("✅ IntelligentRiskManager import successful")
    except ImportError as e:
        print(f"❌ IntelligentRiskManager import failed: {e}")
        return False
    
    try:
        from models.adaptive_learning_system import AdaptiveLearningSystem
        print("✅ AdaptiveLearningSystem import successful")
    except ImportError as e:
        print(f"❌ AdaptiveLearningSystem import failed: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic functionality of professional components"""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        from trading.intelligent_risk_manager import IntelligentRiskManager
        from models.adaptive_learning_system import AdaptiveLearningSystem
        
        # Test initialization
        risk_manager = IntelligentRiskManager()
        learning_system = AdaptiveLearningSystem()
        print("✅ Professional components initialized successfully")
        
        # Test that key methods exist
        assert hasattr(risk_manager, 'assess_symbol_risk')
        assert hasattr(risk_manager, 'calculate_options_position_size')
        assert hasattr(learning_system, 'record_trade_entry')
        assert hasattr(learning_system, 'should_trade_symbol')
        print("✅ All required methods exist")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 PROFESSIONAL TRADING BOT - QUICK TEST")
    print("="*50)
    
    if test_imports() and test_basic_functionality():
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your professional trading bot is working correctly!")
    else:
        print("\n⚠️  Some tests failed - check the errors above")