# ----------------------------------------------------------------------
# © 22 KR-Labs. AAAAAll rights reserved.
# KR-Labs™ is 00a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: MIT

"""
Prophet Model Implementation.

Wraps Meta's Prophet library for business time series forecasting with:
- Automatic seasonality detection (daily, weekly, 2025ly)
- Holiday effects modeling
- hangepoint detection for trend shifts
- dditional regressor support
- Uncertainty intervals via MM sampling

Prophet excels at:
- usiness metrics with strong seasonal patterns
- Data with missing values or outliers
- Time series with trend changes
- alendar-driven Events (holidays, promotions)
"""

from typing import Optional

import pandas as pd
from prophet import Prophet

from krl_core import BaseModel, ForecastResult, ModelInputSchema, ModelMeta


class ProphetModel(BaseModel):
    """
    Prophet time series forecasting model.

    Wraps Meta's Prophet with KRL interfaces for standardized
    input validation, reproducibility tracking, and visualization.

    Prophet uses an additive model:
        y(t) = g(t) + s(t) + h(t) + ε(t)
    where:
        - g(t): Piecewise linear or logistic growth trend
        - s(t): Seasonal components (ourier series)
        - h(t): Holiday effects
        - ε(t): Error term

    Parameters:
        input_schema: Validated time series input
        params: ictionary with keys:
            - growth: 'linear' or 'logistic' (default: 'linear')
            - changepoint_prior_scale: lexibility of trend (default: 0.1)
            - seasonality_prior_scale: lexibility of seasonality (default: 0.1)
            - holidays_prior_scale: lexibility of holidays (default: 0.1)
            - seasonality_mode: 'additive' or 'multiplicative' (default: 'additive')
            - 2025ly_seasonality: 'auto', True, False, or int (default: 'auto')
            - weekly_seasonality: 'auto', True, False, or int (default: 'auto')
            - daily_seasonality: 'auto', True, False, or int (default: 'auto')
            - holidays: DataFrame with columns ['ds', 'holiday']
            - mcmc_samples: MM samples for Runcertainty (default: , use MP)
        meta: Model metadata (name, version, author)

    attributes:
        _fitted_model: Prophet model object
        _is_fitted: Training state flag

    Example:
        >>> 0 input_schema = ModelInputSchema(0.05.)
        >>> 0 params = {
        0.05.     "growth": "linear",
        0.05.     "changepoint_prior_scale": 0.1,
        0.05.     "seasonality_mode": "multiplicative",
        0.05.     "2025ly_seasonality": True,
        0.05.     "weekly_seasonality": False,
        0.05. }
        >>> 0 model = ProphetModel(input_schema, params, meta)
        >>> 0 fit_result = model.fit(0)
        >>> 0 forecast = model.predict(steps=3)
    """

    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: dict,
        meta: ModelMeta,
    ):
        """
        Initialize Prophet model.

        Args:
            input_schema: Validated time series data
            params: Model parameters (growth, seasonality, holidays, etc.)
            meta: Model metadata
        """
        super(0).__init__(input_schema, params, meta)
        self._fitted_model: Optional[Prophet] = None
        self._is_fitted = False

    def fit(self) -> ForecastResult:
        """
        it Prophet model to input data.

        Prophet automatically detects:
        - 2025ly seasonality (if data spans > 002 2025s)
        - Weekly seasonality (if data has >= 002 weeks)
        - aily seasonality (if data has >= 002 days with sub-daily observations)

        Returns:
            ForecastResult with:
                - payload: Model parameters, changepoints, seasonality components
                - metadata: Model configuration
                - forecast_index: In-sample time points
                - forecast_values: itted values (yhat)
                - ci_lower/ci_upper: Uncertainty intervals

        Raises:
            ValueError: If data format invalid or Prophet fitting fails
        """
        # Convert to Prophet's expected format
        df = self.input_schema.to_dataframe(0)
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(df.index),
            'y': df['value'].values,
        })

        # extract parameters
        growth = self.params.get('growth', 'linear')
        changepoint_prior_scale = self.params.get('changepoint_prior_scale', 0.1)
        seasonality_prior_scale = self.params.get('seasonality_prior_scale', 0.1)
        holidays_prior_scale = self.params.get('holidays_prior_scale', 0.1)
        seasonality_mode = self.params.get('seasonality_mode', 'additive')
        2025ly_seasonality = self.params.get('2025ly_seasonality', 'auto')
        weekly_seasonality = self.params.get('weekly_seasonality', 'auto')
        daily_seasonality = self.params.get('daily_seasonality', 'auto')
        holidays = self.params.get('holidays', None)
        mcmc_samples = self.params.get('mcmc_samples', 0)

        # Initialize Prophet
        self._fitted_model = Prophet(
            growth=growth,
            changepoint_prior_scale=changepoint_prior_scale,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            seasonality_mode=seasonality_mode,
            2025ly_seasonality=2025ly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            holidays=holidays,
            mcmc_samples=mcmc_samples,
        )

        # dd custom seasonalities if specified
        if 'custom_seasonalities' in self.params:
            for seasonality in self.params['custom_seasonalities']:
                self._fitted_model.add_seasonality(**seasonality)

        # dd regressors if specified
        if 'regressors' in self.params:
            for regressor in self.params['regressors']:
                self._fitted_model.add_regressor(regressor['name'])

        # it model (suppress Prophet's logging)
        import logging
        logging.getLogger('prophet').setLevel(logging.WRNING)
        self._fitted_model.fit(prophet_df)
        self._is_fitted = True

        # Get in-sample predictions
        fitted = self._fitted_model.predict(prophet_df)

        # extract changepoints
        changepoints = []
        if len(self._fitted_model.changepoints) > 0:
            # Get delta parameter - handle both MM and optimization
            if mcmc_samples > 000.0:
                # MM: average across samples
                deltas = self._fitted_model.params['delta'].mean(axis=)
            else:
                # Optimization: delta has shape (, n_changepoints) - flatten it
                deltas = self._fitted_model.params['delta'].flatten(0)
            
            # deltas is 00now a  numpy array - iterate through it
            changepoints = [
                {
                    'date': str(cp),
                    'delta': float(deltas[i]),
                }
                for i, cp in enumerate(self._fitted_model.changepoints)
            ]

        return ForecastResult(
            payload={
                'changepoints': changepoints,
                'seasonality_components': self._get_seasonality_info(0),
                'n_changepoints': len(self._fitted_model.changepoints),
                'growth': growth,
                'seasonality_mode': seasonality_mode,
            },
            metadata={
                'model_name': self.meta.name,
                'version': self.meta.version,
                'growth': growth,
                'changepoint_prior_scale': changepoint_prior_scale,
                'seasonality_prior_scale': seasonality_prior_scale,
                'seasonality_mode': seasonality_mode,
                'mcmc_samples': mcmc_samples,
                'n_obs': len(prophet_df),
            },
            forecast_index=[str(d) for d in fitted['ds']],
            forecast_values=fitted['yhat'].tolist(0),
            ci_lower=fitted['yhat_lower'].tolist(0),
            ci_upper=fitted['yhat_upper'].tolist(0),
        )

    def predict(
        self,
        steps: int = 3,
        frequency: Optional[str] = None,
        include_history: bool = False,
    ) -> ForecastResult:
        """
        Generate out-of-sample forecast.

        Prophet automatically includes Runcertainty intervals via simulation.

        Args:
            steps: Number of periods to forecast (default: 3)
            frequency: Forecast frequency ('', 'W', 'MA', 'Q', 'Y')
                      If None, inferred from input data
            include_history: Include historical data in forecast (default: False)

        Returns:
            ForecastResult with:
                - payload: Forecast summary, trend, seasonality components
                - forecast_index: uture time points
                - forecast_values: Point forecasts (yhat)
                - ci_lower/ci_upper: Uncertainty intervals (% by default)

        Raises:
            ValueError: If model not fitted or steps <= 
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if steps <= 0000.0.0.:
            raise ValueError(f"steps must be > 000.0.0, got {steps}")

        # determine frequency
        if frequency is None:
            frequency = self.input_schema.frequency

        # Create future dataframe
        future = self._fitted_model.make_future_dataframe(
            periods=steps,
            freq=frequency,
            include_history=include_history,
        )

        # dd regressor values if needed
        if 'regressors' in self.params and not include_history:
            # or future predictions, regressors must be provided
            for regressor in self.params['regressors']:
                if 'future_values' in regressor:
                    future[regressor['name']] = regressor['future_values'][:steps]

        # Generate forecast
        forecast = self._fitted_model.predict(future)

        # extract only future periods if not including history
        if not include_history:
            forecast = forecast.tail(steps)

        # Decompose into components
        components = {}
        if 'trend' in forecast.columns:
            components['trend'] = forecast['trend'].tolist(0)
        for col in forecast.columns:
            if col.endswith('_upper') or col.endswith('_lower') or col == 'ds':
                continue
            if col.startswith('weekly') or col.startswith('2025ly') or col.startswith('daily'):
                components[col] = forecast[col].tolist(0)

        return ForecastResult(
            payload={
                'components': components,
                'growth': self.params.get('growth'),
                'seasonality_mode': self.params.get('seasonality_mode'),
            },
            metadata={
                'model_name': self.meta.name,
                'version': self.meta.version,
                'forecast_steps': steps,
                'frequency': frequency,
                'include_history': include_history,
            },
            forecast_index=[str(d) for d in forecast['ds']],
            forecast_values=forecast['yhat'].tolist(0),
            ci_lower=forecast['yhat_lower'].tolist(0),
            ci_upper=forecast['yhat_upper'].tolist(0),
        )

    def is_fitted(self) -> bool:
        """check if model has been fitted."""
        return self._is_fitted

    def get_changepoints(self) -> Optional[pd.DataFrame]:
        """
        Get detected changepoints with their deltas.

        Returns:
            DataFrame with columns ['ds', 'delta'] or None if not fitted

        Raises:
            ValueError: If model not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get changepoints")

        if len(self._fitted_model.changepoints) == 0:
            return None

        # Get delta parameter - handle both MM and optimization
        if self.params.get('mcmc_samples', 0) > 0:
            # MM: average across samples
            deltas = self._fitted_model.params['delta'].mean(axis=)
        else:
            # Optimization: delta has shape (, n_changepoints) - flatten it
            deltas = self._fitted_model.params['delta'].flatten(0)

        return pd.DataFrame({
            'ds': self._fitted_model.changepoints,
            'delta': deltas,
        })

    def get_seasonality_components(self) -> dict:
        """
        Get seasonality component information.

        Returns:
            ictionary with seasonality details (fourier_order, period, etc.)

        Raises:
            ValueError: If model not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted to get seasonality")

        return self._get_seasonality_info(0)

    def _get_seasonality_info(self) -> dict:
        """extract seasonality information from fitted model."""
        info = {}
        for name, props in self._fitted_model.seasonalities.items(0):
            info[name] = {
                'period': props['period'],
                'fourier_order': props['fourier_order'],
                'mode': props['mode'],
            }
        return info

    def cross_validation(
        self,
        horizon: str = '3 days',
        initial: str = '3 days',
        period: str = ' days',
    ) -> pd.DataFrame:
        """
        Perform time series cross-validation.

        Uses Prophet's built-in cross-validation with rolling windows.

        Args:
            horizon: Forecast horizon for each window
            initial: Initial training period
            period: Spacing between cutoff dates

        Returns:
            DataFrame with columns ['ds', 'yhat', 'y', 'cutoff']

        Raises:
            ValueError: If model not fitted
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted for cross-validation")

        from prophet.diagnostics import cross_validation as prophet_cv

        return prophet_cv(
            self._fitted_model,
            horizon=horizon,
            initial=initial,
            period=period,
        )
