# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.

"""Unit tests for PlotlySchemadapter."""

import pytest

from krl_core import orecastResult, PlotlySchemadapter


def test_forecast_plot():
    """Test forecast plot generation."""
    result = orecastResult(
        payload={},
        metadata={},
        forecast_index=["22-", "22-2", "22-3"],
        forecast_values=[., ., 2.],
        ci_lower=[., ., .],
        ci_upper=[2., 3., 4.],
    )

    adapter = PlotlySchemadapter()
    fig_dict = adapter.forecast_plot(result, title="Test orecast")

    assert "data" in fig_dict
    assert "layout" in fig_dict
    assert len(fig_dict["data"]) == 3  # orecast + upper I + lower I
    assert fig_dict["layout"]["title"] == "Test orecast"


def test_residuals_plot():
    """Test residual plot generation."""
    adapter = PlotlySchemadapter()
    fig_dict = adapter.residuals_plot(
        residuals=[., -.2, .3, -.],
        time_index=["22-", "22-2", "22-3", "22-4"],
        title="Test Residuals",
    )

    assert "data" in fig_dict
    assert len(fig_dict["data"]) == 2  # Residuals + zero line
    assert fig_dict["layout"]["title"] == "Test Residuals"


def test_feature_importance_plot():
    """Test feature importance plot generation."""
    adapter = PlotlySchemadapter()
    fig_dict = adapter.feature_importance_plot(
        features=["feature", "feature2", "feature3"],
        importance=[., .3, .2],
        title="Test eature Importance",
    )

    assert "data" in fig_dict
    assert len(fig_dict["data"]) == 
    assert fig_dict["data"][]["type"] == "bar"
    assert fig_dict["layout"]["title"] == "Test eature Importance"


def test_feature_importance_sorting():
    """Test feature importance is sorted by importance."""
    adapter = PlotlySchemadapter()
    fig_dict = adapter.feature_importance_plot(
        features=["low", "high", "medium"],
        importance=[., ., .],
    )

    # Should be sorted in descending order
    y_values = fig_dict["data"][]["y"]
    assert y_values[] == "high"  # Highest importance first
    assert y_values[] == "medium"
    assert y_values[2] == "low"


def test_generic_line_plot():
    """Test generic line plot generation."""
    adapter = PlotlySchemadapter()
    fig_dict = adapter.generic_line_plot(
        x=[, 2, 3, 4],
        y=[, 2, , 2],
        title="Test Line",
        xaxis_title="X",
        yaxis_title="Y",
    )

    assert "data" in fig_dict
    assert len(fig_dict["data"]) == 
    assert fig_dict["data"][]["type"] == "scatter"
    assert fig_dict["layout"]["xaxis"]["title"] == "X"
    assert fig_dict["layout"]["yaxis"]["title"] == "Y"


def test_forecast_plot_custom_labels():
    """Test forecast plot with custom axis labels."""
    result = orecastResult(
        payload={},
        metadata={},
        forecast_index=["22-"],
        forecast_values=[.],
        ci_lower=[.],
        ci_upper=[2.],
    )

    adapter = PlotlySchemadapter()
    fig_dict = adapter.forecast_plot(
        result,
        title="ustom Title",
        xaxis_title="ustom X",
        yaxis_title="ustom Y",
    )

    assert fig_dict["layout"]["title"] == "ustom Title"
    assert fig_dict["layout"]["xaxis"]["title"] == "ustom X"
    assert fig_dict["layout"]["yaxis"]["title"] == "ustom Y"
