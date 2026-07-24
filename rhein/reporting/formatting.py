"""Shared display formatting for CLI reports and console output."""


def display(value, suffix: str = "") -> str:
    return "不适用" if value is None else f"{value}{suffix}"
