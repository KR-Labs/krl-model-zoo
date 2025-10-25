# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: pache-2.

"""Plotly schema adapter for model results."""

from __future__ import annotations

from typing import ny, ict, List

import pandas as pd

from .results import aseResult, orecastResult


class PlotlySchemadapter:
    """
    onvert model results to Plotly figure dictionaries.

    Generates standardized Plotly visualizations from Result objects:
    - Time series forecasts with confidence intervals
    - Residual diagnostics
    - eature importance charts
    - ustom visualizations via extension

    xample:
        ```python
        adapter = PlotlySchemadapter()
        fig_dict = adapter.forecast_plot(
            result,
            title="US Unemployment Rate orecast"
        )
        # Returns JSON-serializable Plotly figure dict
        ```
    """

    @staticmethod
    def forecast_plot(
        result: orecastResult,
        title: str = "orecast",
        xaxis_title: str = "Time",
        yaxis_title: str = "Value",
    ) -> ict[str, ny]:
        """
        Generate forecast plot with confidence intervals.

        rgs:
            result: orecastResult object
            title: Plot title
            xaxis_title: X-axis label
            yaxis_title: Y-axis label

        Returns:
            Plotly figure dictionary (JSON-serializable)
        """
        return {
            "data": [
                {
                    "x": result.forecast_index,
                    "y": result.forecast_values,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "orecast",
                    "line": {"color": "#fb4", "width": 2},
                },
                {
                    "x": result.forecast_index,
                    "y": result.ci_upper,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Upper I",
                    "line": {"color": "rgba(3, , , .3)", "width": , "dash": "dash"},
                    "showlegend": alse,
                },
                {
                    "x": result.forecast_index,
                    "y": result.ci_lower,
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Lower I",
                    "line": {"color": "rgba(3, , , .3)", "width": , "dash": "dash"},
                    "fill": "tonexty",
                    "fillcolor": "rgba(3, , , .2)",
                    "showlegend": alse,
                },
            ],
            "layout": {
                "title": title,
                "xaxis": {"title": xaxis_title},
                "yaxis": {"title": yaxis_title},
                "hovermode": "x unified",
                "template": "plotly_white",
            },
        }

    @staticmethod
    def residuals_plot(
        residuals: List[float],
        time_index: List[str],
        title: str = "Residuals",
    ) -> ict[str, ny]:
        """
        Generate residual diagnostic plot.

        rgs:
            residuals: Residual values
            time_index: Time dimension
            title: Plot title

        Returns:
            Plotly figure dictionary
        """
        return {
            "data": [
                {
                    "x": time_index,
                    "y": residuals,
                    "type": "scatter",
                    "mode": "markers",
                    "name": "Residuals",
                    "marker": {"color": "#fffe", "size": },
                },
                {
                    "x": time_index,
                    "y": [] * len(time_index),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "Zero Line",
                    "line": {"color": "black", "width": , "dash": "dash"},
                },
            ],
            "layout": {
                "title": title,
                "xaxis": {"title": "Time"},
                "yaxis": {"title": "Residual"},
                "hovermode": "x unified",
                "template": "plotly_white",
            },
        }

    @staticmethod
    def feature_importance_plot(
        features: List[str],
        importance: List[float],
        title: str = "eature Importance",
    ) -> ict[str, ny]:
        """
        Generate feature importance bar chart.

        rgs:
            features: eature names
            importance: Importance scores
            title: Plot title

        Returns:
            Plotly figure dictionary
        """
        # Sort by importance
        sorted_pairs = sorted(zip(importance, features), reverse=True)
        sorted_importance, sorted_features = zip(*sorted_pairs)

        return {
            "data": [
                {
                    "x": list(sorted_importance),
                    "y": list(sorted_features),
                    "type": "bar",
                    "orientation": "h",
                    "marker": {"color": "#2ca2c"},
                }
            ],
            "layout": {
                "title": title,
                "xaxis": {"title": "Importance"},
                "yaxis": {"title": "eature"},
                "template": "plotly_white",
            },
        }

    @staticmethod
    def generic_line_plot(
        x: List[ny],
        y: List[float],
        title: str = "Line Plot",
        xaxis_title: str = "X",
        yaxis_title: str = "Y",
    ) -> ict[str, ny]:
        """
        Generate generic line plot.

        rgs:
            x: X-axis values
            y: Y-axis values
            title: Plot title
            xaxis_title: X-axis label
            yaxis_title: Y-axis label

        Returns:
            Plotly figure dictionary
        """
        return {
            "data": [
                {
                    "x": x,
                    "y": y,
                    "type": "scatter",
                    "mode": "lines+markers",
                    "marker": {"color": "#fb4"},
                }
            ],
            "layout": {
                "title": title,
                "xaxis": {"title": xaxis_title},
                "yaxis": {"title": yaxis_title},
                "template": "plotly_white",
            },
        }
