# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Plotly Adapter - Result Visualization Integration
==================================================

Apache 2.0 License - Gate 1 Foundation
Author: KR Labs

Converts result classes to Plotly-compatible JSON schemas.
Enables interactive visualizations for forecasts, causal effects, and anomalies.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime, timedelta


class PlotlyAdapter:
    """
    Adapter for converting result objects to Plotly figure specs.
    
    Converts ForecastResult, CausalResult, AnomalyResult, etc.
    into JSON-compatible dictionaries for Plotly visualization.
    
    Examples
    --------
    >>> from krl_core.results import ForecastResult
    >>> result = ForecastResult(
    ...     model_name='KalmanFilter',
    ...     point_forecast=np.array([1, 2, 3]),
    ...     lower=np.array([0.5, 1.5, 2.5]),
    ...     upper=np.array([1.5, 2.5, 3.5])
    ... )
    >>> adapter = PlotlyAdapter()
    >>> fig_spec = adapter.forecast_to_plotly(result)
    """
    
    def __init__(self):
        pass
    
    def forecast_to_plotly(
        self,
        result,
        dates: Optional[List] = None,
        title: str = "Forecast"
    ) -> Dict[str, Any]:
        """
        Convert ForecastResult to Plotly figure spec.
        
        Parameters
        ----------
        result : ForecastResult
            Forecast result object
        dates : list, optional
            Date labels for x-axis
        title : str, default='Forecast'
            Figure title
        
        Returns
        -------
        fig_spec : dict
            Plotly figure specification
        """
        n = len(result.point_forecast)
        if dates is None:
            dates = list(range(n))
        
        data = [
            {
                'x': dates,
                'y': result.point_forecast.tolist(),
                'type': 'scatter',
                'mode': 'lines',
                'name': 'Forecast',
                'line': {'color': '#1f77b4', 'width': 2}
            }
        ]
        
        # Add confidence intervals
        if result.lower is not None and result.upper is not None:
            data.extend([
                {
                    'x': dates,
                    'y': result.upper.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': f'{int((1-result.alpha)*100)}% Upper',
                    'line': {'color': 'rgba(31, 119, 180, 0.3)', 'dash': 'dash'},
                    'showlegend': True
                },
                {
                    'x': dates,
                    'y': result.lower.tolist(),
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': f'{int((1-result.alpha)*100)}% Lower',
                    'line': {'color': 'rgba(31, 119, 180, 0.3)', 'dash': 'dash'},
                    'fill': 'tonexty',
                    'fillcolor': 'rgba(31, 119, 180, 0.2)'
                }
            ])
        
        layout = {
            'title': title,
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Value'},
            'hovermode': 'x unified',
            'template': 'plotly_white'
        }
        
        return {'data': data, 'layout': layout}
    
    def causal_to_plotly(
        self,
        result,
        title: str = "Causal Effect"
    ) -> Dict[str, Any]:
        """
        Convert CausalResult to Plotly figure spec.
        
        Parameters
        ----------
        result : CausalResult
            Causal result object
        title : str, default='Causal Effect'
            Figure title
        
        Returns
        -------
        fig_spec : dict
            Plotly figure specification with effect size and CI
        """
        # Bar plot for treatment effect
        data = [
            {
                'x': ['Treatment Effect'],
                'y': [result.treatment_effect],
                'type': 'bar',
                'name': 'Effect',
                'marker': {'color': '#2ca02c'},
                'error_y': {
                    'type': 'data',
                    'array': [result.confidence_interval[1] - result.treatment_effect],
                    'arrayminus': [result.treatment_effect - result.confidence_interval[0]]
                }
            }
        ]
        
        layout = {
            'title': f"{title}<br><sub>p-value: {result.p_value:.4f}</sub>",
            'yaxis': {'title': 'Effect Size'},
            'showlegend': False,
            'template': 'plotly_white'
        }
        
        return {'data': data, 'layout': layout}
    
    def anomaly_to_plotly(
        self,
        result,
        original_series: np.ndarray,
        dates: Optional[List] = None,
        title: str = "Anomaly Detection"
    ) -> Dict[str, Any]:
        """
        Convert AnomalyResult to Plotly figure spec.
        
        Parameters
        ----------
        result : AnomalyResult
            Anomaly result object
        original_series : array-like
            Original time series
        dates : list, optional
            Date labels
        title : str, default='Anomaly Detection'
            Figure title
        
        Returns
        -------
        fig_spec : dict
            Plotly figure with anomalies highlighted
        """
        n = len(original_series)
        if dates is None:
            dates = list(range(n))
        
        # Normal points
        normal_mask = ~result.anomaly_labels
        data = [
            {
                'x': [dates[i] for i in range(n) if normal_mask[i]],
                'y': [original_series[i] for i in range(n) if normal_mask[i]],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Normal',
                'marker': {'color': '#1f77b4', 'size': 5}
            }
        ]
        
        # Anomaly points
        if result.anomaly_indices.size > 0:
            data.append({
                'x': [dates[i] for i in result.anomaly_indices],
                'y': [original_series[i] for i in result.anomaly_indices],
                'type': 'scatter',
                'mode': 'markers',
                'name': 'Anomaly',
                'marker': {'color': '#d62728', 'size': 10, 'symbol': 'x'}
            })
        
        layout = {
            'title': f"{title}<br><sub>Detected {len(result.anomaly_indices)} anomalies</sub>",
            'xaxis': {'title': 'Time'},
            'yaxis': {'title': 'Value'},
            'hovermode': 'closest',
            'template': 'plotly_white'
        }
        
        return {'data': data, 'layout': layout}
    
    def classification_to_plotly(
        self,
        result,
        title: str = "Classification Results"
    ) -> Dict[str, Any]:
        """
        Convert ClassificationResult to Plotly confusion matrix.
        
        Parameters
        ----------
        result : ClassificationResult
            Classification result object
        title : str, default='Classification Results'
            Figure title
        
        Returns
        -------
        fig_spec : dict
            Plotly heatmap of predictions
        """
        # Create confusion matrix if true labels available
        # For now, show prediction distribution
        unique, counts = np.unique(result.predictions, return_counts=True)
        
        data = [{
            'x': [str(c) for c in unique],
            'y': counts.tolist(),
            'type': 'bar',
            'marker': {'color': '#ff7f0e'}
        }]
        
        layout = {
            'title': title,
            'xaxis': {'title': 'Class'},
            'yaxis': {'title': 'Count'},
            'template': 'plotly_white'
        }
        
        return {'data': data, 'layout': layout}
    
    def specialization_to_plotly(
        self,
        result,
        title: str = "Specialization Analysis"
    ) -> Dict[str, Any]:
        """
        Convert SpecializationResult to Plotly figure.
        
        Parameters
        ----------
        result : SpecializationResult
            Specialization result object
        title : str, default='Specialization Analysis'
            Figure title
        
        Returns
        -------
        fig_spec : dict
            Plotly bar chart of location quotients
        """
        industries = list(result.location_quotients.keys())
        lqs = list(result.location_quotients.values())
        
        # Color by specialization (LQ > 1)
        colors = ['#2ca02c' if lq > 1 else '#d62728' for lq in lqs]
        
        data = [{
            'x': industries,
            'y': lqs,
            'type': 'bar',
            'marker': {'color': colors},
            'text': [f'{lq:.2f}' for lq in lqs],
            'textposition': 'outside'
        }]
        
        layout = {
            'title': title,
            'xaxis': {'title': 'Industry'},
            'yaxis': {'title': 'Location Quotient', 'zeroline': True},
            'shapes': [{
                'type': 'line',
                'x0': -0.5,
                'x1': len(industries) - 0.5,
                'y0': 1,
                'y1': 1,
                'line': {'color': 'black', 'width': 2, 'dash': 'dash'}
            }],
            'template': 'plotly_white'
        }
        
        return {'data': data, 'layout': layout}
