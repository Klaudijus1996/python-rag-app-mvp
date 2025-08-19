import logging
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)


class DataConverter:
    """Utility class for safe data type conversions with comprehensive error handling."""

    @staticmethod
    def to_string(value: Any, default: str = "") -> str:
        """Safely convert value to string, handling None, NaN, and empty values."""
        if pd.isna(value) or value is None:
            return default

        str_value = str(value).strip()
        if not str_value or str_value.lower() in ["nan", "null", "none", ""]:
            return default

        return str_value

    @staticmethod
    def to_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float, handling None, NaN, and invalid strings."""
        if pd.isna(value) or value is None:
            return default

        if isinstance(value, (int, float)):
            return float(value)

        str_value = str(value).strip()
        if not str_value or str_value.lower() in ["nan", "null", "none", "", "n/a"]:
            return default

        try:
            return float(str_value)
        except (ValueError, TypeError):
            logger.warning(
                f"Could not convert '{value}' to float, using default {default}"
            )
            return default

    @staticmethod
    def to_int(value: Any, default: int = 0) -> int:
        """Safely convert value to int, handling None, NaN, and invalid strings."""
        if pd.isna(value) or value is None:
            return default

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        str_value = str(value).strip()
        if not str_value or str_value.lower() in ["nan", "null", "none", "", "n/a"]:
            return default

        try:
            return int(
                float(str_value)
            )  # Convert through float first to handle "1.0" strings
        except (ValueError, TypeError):
            logger.warning(
                f"Could not convert '{value}' to int, using default {default}"
            )
            return default

    @staticmethod
    def to_bool(value: Any, default: bool = False) -> bool:
        """Safely convert value to bool, handling various string representations."""
        if pd.isna(value) or value is None:
            return default

        if isinstance(value, bool):
            return value

        str_value = str(value).strip().lower()
        if not str_value or str_value in ["nan", "null", "none", ""]:
            return default

        if str_value in ["true", "1", "yes", "on", "y"]:
            return True
        elif str_value in ["false", "0", "no", "off", "n"]:
            return False
        else:
            logger.warning(
                f"Could not convert '{value}' to bool, using default {default}"
            )
            return default
