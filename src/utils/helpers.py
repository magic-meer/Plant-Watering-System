"""Helper utilities for Plant Watering System."""


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal value as a percentage string.

    Args:
        value: Decimal value to format (e.g., 0.75 for 75%).
        decimals: Number of decimal places.

    Returns:
        Formatted percentage string.
    """
    return f"{value * 100:.{decimals}f}%"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on division by zero.

    Args:
        numerator: The numerator.
        denominator: The denominator.
        default: Default value if denominator is zero.

    Returns:
        Result of division or default value.
    """
    if denominator == 0:
        return default
    return numerator / denominator
