from __future__ import annotations

import pandas as pd
import pytest

from rhein.ui.macd import (
    MACD_FAST_WINDOW,
    MACD_SIGNAL_WINDOW,
    MACD_SLOW_WINDOW,
    add_display_macd,
    add_display_mmacd,
)
from rhein.ui.dmi import DMI_ADX_WINDOW, DMI_DIRECTIONAL_WINDOW, add_display_dmi


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


def test_display_mmacd_uses_normalised_dd_and_formula_histogram() -> None:
    frame = pd.DataFrame({"Close": [10.0, 11.0, 10.0, 12.0, 13.0]})

    actual = add_display_mmacd(frame)
    dd = (
        frame["Close"].ewm(span=10, adjust=False, min_periods=1).mean()
        - frame["Close"].ewm(span=24, adjust=False, min_periods=1).mean()
    ) / frame["Close"] * 100
    ded = dd.ewm(span=8, adjust=False, min_periods=1).mean()

    assert actual["MMACD_DD"].tolist() == pytest.approx(dd.tolist())
    assert actual["MMACD_DED"].tolist() == pytest.approx(ded.tolist())
    assert actual["MMACD_HIST"].tolist() == pytest.approx((2 * (dd - ded)).tolist())
    assert list(frame.columns) == ["Close"]


def test_display_dmi_uses_fixed_10_6_and_does_not_mutate_input_frame() -> None:
    frame = pd.DataFrame(
        {
            "High": [10.0, 12.0, 11.0, 13.0],
            "Low": [8.0, 9.0, 8.0, 10.0],
            "Close": [9.0, 11.0, 9.0, 12.0],
        }
    )

    result = add_display_dmi(frame)

    assert (DMI_DIRECTIONAL_WINDOW, DMI_ADX_WINDOW) == (10, 6)
    assert list(frame.columns) == ["High", "Low", "Close"]
    assert {"DMI_PDI", "DMI_MDI", "DMI_ADX", "DMI_ADXR"}.issubset(result.columns)
    assert result["DMI_PDI"].tolist() == pytest.approx([0.0, 40.0, 25.0, 33.3333333333])
    assert result["DMI_MDI"].tolist() == pytest.approx([0.0, 0.0, 12.5, 8.3333333333])
    assert result[["DMI_ADX", "DMI_ADXR"]].notna().all().all()
