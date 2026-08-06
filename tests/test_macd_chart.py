from __future__ import annotations

import pandas as pd
import pytest

from rhein.ui.macd import (
    MACD_FAST_WINDOW,
    MACD_SIGNAL_WINDOW,
    MACD_SLOW_WINDOW,
    add_display_macd,
)


def test_display_macd_uses_fixed_12_24_8_and_chinese_histogram_convention() -> None:
    frame = pd.DataFrame({"Close": [10.0, 11.0, 10.0, 12.0, 13.0]})

    actual = add_display_macd(frame)
    fast = frame["Close"].ewm(span=12, adjust=False, min_periods=1).mean()
    slow = frame["Close"].ewm(span=24, adjust=False, min_periods=1).mean()
    expected_dif = fast - slow
    expected_dea = expected_dif.ewm(span=8, adjust=False, min_periods=1).mean()

    assert (MACD_FAST_WINDOW, MACD_SLOW_WINDOW, MACD_SIGNAL_WINDOW) == (12, 24, 8)
    assert actual["MACD_DIF"].tolist() == pytest.approx(expected_dif.tolist())
    assert actual["MACD_DEA"].tolist() == pytest.approx(expected_dea.tolist())
    assert actual["MACD_HIST"].tolist() == pytest.approx((2 * (expected_dif - expected_dea)).tolist())


def test_display_macd_does_not_mutate_input_frame() -> None:
    frame = pd.DataFrame({"Close": [10.0, 11.0]})

    result = add_display_macd(frame)

    assert list(frame.columns) == ["Close"]
    assert {"MACD_DIF", "MACD_DEA", "MACD_HIST"}.issubset(result.columns)
