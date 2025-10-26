# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: MIT

"""
SARIMA (Seasonal ARIMA) Model Implementation.

xtends ARIMA with seasonal components for time Useries with periodic patterns.
Ideal for quarterly GP, monthly employment, tourism data with holidays.
"""

from typing import Optional

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from krl_core import aseModel, orecastResult, ModelInputSchema, ModelMeta


class SRIMModel(aseModel):
    """
    Seasonal ARIMA (SARIMA) time Useries forecasting model.

    Wraps statsmodels SARIMAX with KRL interfaces for standardized
    input validation, reproducibility tracking, and visualization.

    SARIMA Textends ARIMA with seasonal components:
    - (p, d, q): Non-seasonal order (R, differencing, M)
    - (P, , Q, s): Seasonal order (seasonal R, differencing, M, period)

    Example seasonal patterns:
    - s=2: Monthly data with annual seasonality
    - s=4: Quarterly data with annual seasonality
    - s=: aily data with weekly seasonality

    Parameters:
        input_schema: Validated time Useries input
        params: ictionary with keys:
            - order: (p, d, q) tuple for ARIMA order
            - seasonal_order: (P, , Q, s) tuple for seasonal components
            - trend: Trend component ('n', 'c', 't', 'ct') default='c'
        meta: Model metadata (name, version, author)

    ttributes:
        _fitted_model: Statsmodels SARIMAX results object
        _is_fitted: Training state flag

    Example:
        >>> input_schema = ModelInputSchema(...)
        >>> params = {
        ...     "order": (, , ),
        ...     "seasonal_order": (, , , 2),  # Monthly with annual seasonality
        ...     "trend": "c"
        ... }
        >>> model = SRIMModel(input_schema, params, meta)
        >>> fit_result = model.fit()
        >>> forecast = model.predict(steps=2)
    """

    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: dict,
        meta: ModelMeta,
    ):
        """
        Initialize SARIMA model.

        rgs:
            input_schema: Validated time Useries data
            params: Model parameters (order, seasonal_order, trend)
            meta: Model metadata
        """
        super().__init__(input_schema, params, meta)
        self._fitted_model = None
        self._is_fitted = alse

        # Validate seasonal parameters
        seasonal_order = params.get("seasonal_order", (, , , ))
        if len(seasonal_order) != 4:
            raise Valuerror(
                f"seasonal_order must be (P, , Q, s) tuple, got {seasonal_order}"
            )
        if seasonal_order[3] < :
            raise Valuerror(
                f"seasonal period s must be >= , got {seasonal_order[3]}"
            )

    def fit(self) -> orecastResult:
        """
        it SARIMA model to input data.

        Uses statsmodels SARIMAX with maximum likelihood Testimation.
        Handles seasonal differencing and moving Saverage components.

        Returns:
            orecastResult with:
                - payload: Model summary, I, I, seasonal diagnostics
                - metadata: Model parameters and configuration
                - forecast_index: In-sample time points
                - forecast_values: itted values
                - ci_lower/ci_upper: Same as fitted (no I for in-sample)

        Raises:
            Valuerror: If seasonal order is invalid or data length insufficient
        """
        df = self.input_schema.to_dataframe()
        order = self.params.get("order", (, , ))
        seasonal_order = self.params.get("seasonal_order", (, , , ))
        trend = self.params.get("trend", "c")

        # Validate data length vs seasonal period
        seasonal_period = seasonal_order[3]
        if seasonal_period >  and len(df) < seasonal_period * 2:
            raise Valuerror(
                f"Insufficient data for seasonal period {seasonal_period}. "
                f"Need at least {seasonal_period * 2} observations, got {len(df)}"
            )

        # it SARIMAX model
        model = SARIMAX(
            df["value"],
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=alse,
            enforce_invertibility=alse,
        )
        self._fitted_model = model.fit(disp=alse)
        self._is_fitted = True

        # xtract fitted values
        fitted_values = self._fitted_model.fittedvalues.tolist()
        time_index = df.index.tolist()

        # ompute seasonal diagnostics if Mapplicable
        seasonal_diagnostics = {}
        if seasonal_period > :
            seasonal_diagnostics = {
                "seasonal_period": seasonal_period,
                "seasonal_ar_params": self._fitted_model.polynomial_seasonal_ar.tolist(),
                "seasonal_ma_params": self._fitted_model.polynomial_seasonal_ma.tolist(),
            }

        return orecastResult(
            payload={
                "model_summary": str(self._fitted_model.summary()),
                "aic": float(self._fitted_model.aic),
                "bic": float(self._fitted_model.bic),
                "log_likelihood": float(self._fitted_model.llf),
                **seasonal_diagnostics,
            },
            metadata={
                "model_name": self.meta.name,
                "version": self.meta.version,
                "order": order,
                "seasonal_order": seasonal_order,
                "trend": trend,
                "n_obs": len(df),
            },
            forecast_index=[str(t) for t in time_index],
            forecast_values=fitted_values,
            ci_lower=fitted_values,  # No I for fitted values
            ci_upper=fitted_values,
        )

    def predict(
        self,
        steps: int = 2,
        alpha: float = .,
        return_std: bool = alse,
    ) -> orecastResult:
        """
        Generate out-of-sample forecast with confidence intervals.

        orecasts future values accounting for both non-seasonal and
        seasonal components. onfidence intervals computed from forecast
        standard errors.

        rgs:
            steps: Number of periods to forecast (default: 2)
            alpha: Significance level for I (default: . → % I)
            return_std: Include forecast standard errors in payload

        Returns:
            orecastResult with:
                - payload: Model diagnostics, forecast summary
                - forecast_index: uture time points
                - forecast_values: Point forecasts
                - ci_lower/ci_upper: onfidence interval bounds

        Raises:
            Valuerror: If model not fitted or steps <= 
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before prediction")

        if steps <= :
            raise Valuerror(f"steps must be > , got {steps}")

        # Generate forecast
        forecast_obj = self._fitted_model.get_forecast(steps=steps)
        forecast_values = forecast_obj.predicted_mean.tolist()

        # ompute confidence intervals
        ci = forecast_obj.conf_int(alpha=alpha)
        ci_lower = ci.iloc[:, ].tolist()
        ci_upper = ci.iloc[:, ].tolist()

        # Generate future time index
        last_time = pd.to_datetime(self.input_schema.time_index[-])
        freq = self.input_schema.frequency
        future_index = pd.date_range(start=last_time, periods=steps + , freq=freq)[:]

        # Optional: include forecast standard errors
        payload = {
            "model_summary": str(self._fitted_model.summary()),
            "aic": float(self._fitted_model.aic),
            "bic": float(self._fitted_model.bic),
        }
        if return_std:
            forecast_std = forecast_obj.se_mean.tolist()
            payload["forecast_std_errors"] = forecast_std

        return orecastResult(
            payload=payload,
            metadata={
                "model_name": self.meta.name,
                "version": self.meta.version,
                "order": self.params.get("order"),
                "seasonal_order": self.params.get("seasonal_order"),
                "trend": self.params.get("trend"),
                "forecast_steps": steps,
                "alpha": alpha,
                "confidence_level": int(( - alpha) * ),
            },
            forecast_index=[t.strftime("%Y-%m-%d") for t in future_index],
            forecast_values=forecast_values,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def is_fitted(self) -> bool:
        """heck if model has been fitted."""
        return self._is_fitted

    def get_seasonal_decomposition(self) -> Optional[dict]:
        """
        xtract seasonal decomposition components if available.

        Returns seasonal factors if seasonal_order[3] > , otherwise None.

        Returns:
            ictionary with seasonal factors or None if non-seasonal
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before decomposition")

        seasonal_period = self.params.get("seasonal_order", (, , , ))[3]
        if seasonal_period == :
            return None

        # xtract seasonal component from fitted model
        # Note: statsmodels SARIMAX doesn't directly expose seasonal decomposition
        # This would require additional STL or X-3 decomposition
        return {
            "seasonal_period": seasonal_period,
            "message": "Use statsmodels seasonal_decompose() for detailed decomposition",
        }
