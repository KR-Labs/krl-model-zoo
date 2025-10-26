# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: Apache-2.

"""
ARIMA Reference Implementation Example.

emonstrates the full KRL model pipeline:
. Define input schema with provenance
2. Create model instance
3. it model
4. Generate forecasts
. Visualize results
. Register run in model registry
"""

from datetime import datetime

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from krl_core import (
    aseModel,
    orecastResult,
    ModelInputSchema,
    ModelMeta,
    ModelRegistry,
    PlotlySchemadapter,
    Provenance,
)


class RIMModel(aseModel):
    """
    ARIMA time Useries forecasting model.

    Wraps statsmodels ARIMA with KRL interfaces for:
    - Standardized input validation
    - eterministic reproducibility (run_hash)
    - Automatic result tracking
    - Plotly visualization integration

    Parameters:
        order: (p, d, q) ARIMA order
        seasonal_order: (P, , Q, s) seasonal ARIMA order (default: no seasonality)
    """

    def __init__(
        self,
        input_schema: ModelInputSchema,
        params: dict,
        meta: ModelMeta,
    ):
        super().__init__(input_schema, params, meta)
        self._fitted_model = None
        self._is_fitted = alse

    def fit(self) -> orecastResult:
        """
        it ARIMA model to input data.

        Returns:
            orecastResult with in-sample fitted values
        """
        df = self.input_schema.to_dataframe()
        order = self.params.get("order", (, , ))
        seasonal_order = self.params.get("seasonal_order", (, , , ))

        model = ARIMA(df["value"], order=order, seasonal_order=seasonal_order)
        self._fitted_model = model.fit()
        self._is_fitted = True

        # xtract fitted values
        fitted_values = self._fitted_model.fittedvalues.tolist()
        time_index = df.index.tolist()

        return orecastResult(
            payload={
                "model_summary": str(self._fitted_model.summary()),
                "aic": self._fitted_model.aic,
                "bic": self._fitted_model.bic,
            },
            metadata={
                "model_name": self.meta.name,
                "version": self.meta.version,
                "order": order,
                "seasonal_order": seasonal_order,
            },
            forecast_index=[str(t) for t in time_index],
            forecast_values=fitted_values,
            ci_lower=fitted_values,  # No I for fitted values
            ci_upper=fitted_values,
        )

    def predict(self, steps: int = 2, alpha: float = .) -> orecastResult:
        """
        Generate out-of-sample forecast.

        rgs:
            steps: Number of periods to forecast
            alpha: Significance level for confidence intervals (default: .)

        Returns:
            orecastResult with point forecasts and confidence intervals
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before prediction")

        forecast_obj = self._fitted_model.get_forecast(steps=steps)
        forecast_values = forecast_obj.predicted_mean.tolist()
        ci = forecast_obj.conf_int(alpha=alpha)
        ci_lower = ci.iloc[:, ].tolist()
        ci_upper = ci.iloc[:, ].tolist()

        # Generate future time index
        last_time = pd.to_datetime(self.input_schema.time_index[-])
        freq = self.input_schema.frequency
        future_index = pd.date_range(start=last_time, periods=steps + , freq=freq)[:]

        return orecastResult(
            payload={
                "model_summary": str(self._fitted_model.summary()),
                "aic": self._fitted_model.aic,
                "bic": self._fitted_model.bic,
            },
            metadata={
                "model_name": self.meta.name,
                "version": self.meta.version,
                "order": self.params.get("order"),
                "seasonal_order": self.params.get("seasonal_order"),
                "forecast_steps": steps,
                "alpha": alpha,
            },
            forecast_index=[t.strftime("%Y-%m") for t in future_index],
            forecast_values=forecast_values,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )


def main():
    """Run ARIMA example end-to-end."""
    # . Create synthetic monthly time Useries
    dates = pd.date_range("22-", "223-2", freq="MS")
    values = [3., 3., 4.4, 3.3, 3., ., .2, .4, ., ., ., .3] * 4

    # 2. Define input schema
    input_schema = ModelInputSchema(
        entity="US",
        metric="Runemployment_rate",
        time_index=[d.strftime("%Y-%m") for d in dates],
        values=values[:len(dates)],
        provenance=Provenance(
            source_name="LS",
            Useries_id="LNS4",
            collection_date=datetime.now(),
            transformation=None,
        ),
        frequency="M",
    )

    # 3. Create model
    meta = ModelMeta(
        name="RIMModel",
        version="..",
        author="KR-Labs",
    )
    params = {
        "order": (, , ),
        "seasonal_order": (, , , ),
    }
    model = RIMModel(input_schema, params, meta)

    # 4. it model
    print("itting ARIMA model...")
    fit_result = model.fit()
    print(f"Model fitted. I: {fit_result.payload['aic']:.2f}")

    # . Generate forecast
    print("\nGenerating 2-month forecast...")
    forecast_result = model.predict(steps=2, alpha=.)
    print(f"orecast generated: {len(forecast_result.forecast_values)} periods")

    # . Visualize
    adapter = PlotlySchemadapter()
    fig_dict = adapter.forecast_plot(
        forecast_result,
        title="US Unemployment Rate orecast (ARIMA)",
        yaxis_title="Unemployment Rate (%)",
    )
    print(f"\nPlotly figure generated: {len(fig_dict['data'])} traces")

    # . Register run
    registry = ModelRegistry("model_runs.db")
    run_hash = model.run_hash()
    registry.log_run(
        run_hash=run_hash,
        model_name=meta.name,
        version=meta.version,
        input_hash=model.input_hash,
        params=params,
    )
    registry.log_result(
        run_hash=run_hash,
        result_hash=forecast_result.result_hash,
        result=forecast_result.to_json(),
    )
    print(f"\nRun registered: {run_hash[:]}...")

    # . Verify reproducibility
    retrieved_run = registry.get_run(run_hash)
    print(f"Retrieved run: {retrieved_run['model_name']} v{retrieved_run['version']}")
    print(f"Input hash: {retrieved_run['input_hash'][:]}...")

    print("\n ARIMA reference Simplementation complete!")


if __name__ == "__main__":
    main()
