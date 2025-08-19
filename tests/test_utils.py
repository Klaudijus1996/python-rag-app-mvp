import pandas as pd

from utils import DataConverter


class TestDataConverter:
    """Test DataConverter utility functions."""
    
    def test_to_string_with_valid_data(self):
        """Test string conversion with valid data."""
        assert DataConverter.to_string("test") == "test"
        assert DataConverter.to_string(123) == "123"
        assert DataConverter.to_string("  spaced  ") == "spaced"
    
    def test_to_string_with_invalid_data(self):
        """Test string conversion with invalid data."""
        assert DataConverter.to_string(None) == ""
        assert DataConverter.to_string("") == ""
        assert DataConverter.to_string("   ") == ""
        assert DataConverter.to_string("null") == ""
        assert DataConverter.to_string("NaN") == ""
        assert DataConverter.to_string(pd.NA) == ""
    
    def test_to_string_with_custom_default(self):
        """Test string conversion with custom default."""
        assert DataConverter.to_string(None, "default") == "default"
        assert DataConverter.to_string("", "fallback") == "fallback"
    
    def test_to_float_with_valid_data(self):
        """Test float conversion with valid data."""
        assert DataConverter.to_float(123) == 123.0
        assert DataConverter.to_float(45.67) == 45.67
        assert DataConverter.to_float("89.12") == 89.12
        assert DataConverter.to_float("150") == 150.0
    
    def test_to_float_with_invalid_data(self):
        """Test float conversion with invalid data."""
        assert DataConverter.to_float(None) == 0.0
        assert DataConverter.to_float("") == 0.0
        assert DataConverter.to_float("invalid") == 0.0
        assert DataConverter.to_float("null") == 0.0
        assert DataConverter.to_float("n/a") == 0.0
        assert DataConverter.to_float(pd.NA) == 0.0
    
    def test_to_float_with_custom_default(self):
        """Test float conversion with custom default."""
        assert DataConverter.to_float(None, 99.9) == 99.9
        assert DataConverter.to_float("invalid", -1.0) == -1.0
    
    def test_to_int_with_valid_data(self):
        """Test int conversion with valid data."""
        assert DataConverter.to_int(123) == 123
        assert DataConverter.to_int(45.67) == 45
        assert DataConverter.to_int("89") == 89
        assert DataConverter.to_int("150.5") == 150
    
    def test_to_int_with_invalid_data(self):
        """Test int conversion with invalid data."""
        assert DataConverter.to_int(None) == 0
        assert DataConverter.to_int("") == 0
        assert DataConverter.to_int("invalid") == 0
        assert DataConverter.to_int("null") == 0
        assert DataConverter.to_int(pd.NA) == 0
    
    def test_to_bool_with_valid_data(self):
        """Test bool conversion with valid data."""
        assert DataConverter.to_bool(True) is True
        assert DataConverter.to_bool(False) is False
        assert DataConverter.to_bool("true") is True
        assert DataConverter.to_bool("1") is True
        assert DataConverter.to_bool("yes") is True
        assert DataConverter.to_bool("false") is False
        assert DataConverter.to_bool("0") is False
        assert DataConverter.to_bool("no") is False
    
    def test_to_bool_with_invalid_data(self):
        """Test bool conversion with invalid data."""
        assert DataConverter.to_bool(None) is False
        assert DataConverter.to_bool("") is False
        assert DataConverter.to_bool("invalid") is False
        assert DataConverter.to_bool("maybe") is False
    
    def test_pandas_nan_handling(self):
        """Test handling of pandas NaN values."""
        import numpy as np
        
        # Test with numpy NaN
        assert DataConverter.to_string(np.nan) == ""
        assert DataConverter.to_float(np.nan) == 0.0
        assert DataConverter.to_int(np.nan) == 0
        assert DataConverter.to_bool(np.nan) is False
        
        # Test with pandas Series containing NaN
        series = pd.Series([1, 2, np.nan, 4])
        assert DataConverter.to_float(series.iloc[2]) == 0.0
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Very large numbers
        large_num = "999999999999999999"
        assert DataConverter.to_float(large_num) == 999999999999999999.0
        
        # Negative numbers
        assert DataConverter.to_float("-123.45") == -123.45
        assert DataConverter.to_int("-100") == -100
        
        # Scientific notation
        assert DataConverter.to_float("1e5") == 100000.0
        assert DataConverter.to_float("1.5e-3") == 0.0015
        
        # Boolean edge cases
        assert DataConverter.to_bool("TRUE") is True
        assert DataConverter.to_bool("FALSE") is False
        assert DataConverter.to_bool("Yes") is True
        assert DataConverter.to_bool("No") is False