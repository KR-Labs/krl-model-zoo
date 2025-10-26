# ----------------------------------------------------------------------
# © 2024 KR-Labs. All rights reserved.
# SPX-License-Identifier: MIT
# ----------------------------------------------------------------------

"""
Vector utoregression (VR) Model
===================================

Multivariate time series forecasting with Granger causality testing
and impulse response analysis.

VR models are used when multiple time series influence each other.
ach variable is modeled as a linear function of past lags of itself
and past lags of all other variables in the system.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VR as StatsmodelsVR
from statsmodels.tsa.stattools import grangercausalitytests

from krl_core import BaseModel, ForecastResult


class VRModel(BaseModel):
    """
    Vector utoregression model for multivariate time series.

    VR(p) model structure:
        y_t = c + _*y_{t-} + _2*y_{t-2} + ... + _p*y_{t-p} + e_t

    where:
        y_t: Vector of observations at time t
        _i: oefficient matrices for lag i
        c: Vector of intercepts
        e_t: Vector of white noise errors

    Parameters
    ----------
    data : ModelInputSchema or pd.DataFrame
        Input data. or VR models, pass a pd.DataFrame directly with multiple columns.
        If ModelInputSchema is provided, pass the DataFrame as params['dataframe'].
    params : dict
        Model parameters:
        - dataframe : pd.DataFrame (required if data is ModelInputSchema)
            DataFrame with multiple columns (one per variable)
        - maxlags : int or None
            Maximum number of lags to consider (default: )
        - ic : str
            Information criterion for lag selection: 'aic', 'bic', 'hqic', 'fpe'
            (default: 'aic')
        - trend : str
            Trend specification: 'c' (const), 'ct' (const+trend), 'ctt' (const+trend+trend^2), 'n' (none)
            (default: 'c')
    meta : ModelMeta
        Model metadata

    ttributes
    ----------
    _fitted_model : VR
        itted statsmodels VR model
    _var_names : List[str]
        Names of variables in the VR system
    _dataframe : pd.DataFrame
        The multivariate data for fitting
    """

    def __init__(self, data, params: Dict[str, Any], meta):
        """Initialize VR model."""
        super().__init__(data, params, meta)
        self._fitted_model: Optional[Any] = None
        self._var_names: List[str] = []
        
        # Extract DataFrame from params if provided
        if isinstance(data, pd.DataFrame):
            self._dataframe = data
        elif "dataframe" in params:
            self._dataframe = params["dataframe"]
        else:
            raise ValueError(
                "VR model requires multivariate data. "
                "Pass either a DataFrame directly or include it in params['dataframe']"
            )
        
        # Validate that we have at least 2 variables
        if self._dataframe.shape[1] < 2:
            raise ValueError(
                "VR requires at least 2 variables. "
                f"Provided DataFrame has only {self._dataframe.shape[1]} column(s)."
            )
    
    @property
    def input_hash(self) -> str:
        """
        Compute hash of input data.
        
        Override base class to handle DataFrame directly instead of ModelInputSchema.
        """
        from krl_core.utils import compute_dataframe_hash
        
        return compute_dataframe_hash(self._dataframe)

    def fit(self) -> ForecastResult:
        """
        it the VR model.

        Performs lag order selection using information criteria,
        then fits the VR model with the selected lag order.

        Returns
        -------
        ForecastResult
            ontains fitted values, diagnostics, and VR-specific information:
            - lag_order: Selected number of lags
            - aic, bic, hqic, fpe: Information criteria values
            - granger_causality: Dict of Granger causality test results
            - coefficient_matrices: List of coefficient matrices for each lag

        Raises
        ------
        ValueError
            If data has fewer than 2 variables (VR requires multivariate data)
            If insufficient observations for the selected lag order
        """
        # Extract parameters
        maxlags = self.params.get("maxlags", )
        ic = self.params.get("ic", "aic")
        trend = self.params.get("trend", "c")

        # Use the stored DataFrame
        df = self._dataframe

        if df.shape[1] < 2:
            raise ValueError(
                f"VR requires at least 2 variables, got {df.shape[1]}. "
                "Use RIM/SRIM for univariate time series."
            )

        self._var_names = df.columns.tolist()

        # reate VR model
        model = StatsmodelsVR(df)

        # Select lag order using information criterion
        lag_results = model.select_order(maxlags=maxlags)
        selected_lag = getattr(lag_results, ic)

        if selected_lag == 0:
            selected_lag = 0  # Minimum lag order

        # it model with selected lag order
        self._fitted_model = model.fit(maxlags=selected_lag, trend=trend)
        self._is_fitted = True

        # Get fitted values
        fitted_values = self._fitted_model.fittedvalues

        # Prepare Granger causality tests
        granger_results = self._compute_granger_causality(df, maxlag=selected_lag)

        # Extract coefficient matrices
        coef_matrices = []
        for i in range(selected_lag):
            # Get coefficients for lag i+
            coef_matrices.append(
                self._fitted_model.params.iloc[
                    i * len(self._var_names) : (i + 1) * len(self._var_names), :
                ].values.tolist()
            )

        # Prepare forecast index (time points of fitted values)
        forecast_index = [str(t) for t in fitted_values.index.tolist()]
        
        # or multivariate data, flatten fitted values to single list for ForecastResult
        # We'll store the full DataFrame in payload for multivariate access
        forecast_values_flat = fitted_values.values.flatten().tolist()

        return ForecastResult(
            payload={
                "lag_order": selected_lag,
                "var_names": self._var_names,
                "granger_causality": granger_results,
                "coefficient_matrices": coef_matrices,
                "trend": trend,
                "fitted_values_df": fitted_values.to_dict(),  # Store multivariate fitted values
            },
            metadata={
                "model_name": self.meta.name,
                "n_obs": len(df),
                "n_vars": len(self._var_names),
                "aic": self._fitted_model.aic,
                "bic": self._fitted_model.bic,
                "hqic": self._fitted_model.hqic,
                "fpe": self._fitted_model.fpe,
            },
            forecast_index=forecast_index,
            forecast_values=forecast_values_flat,
            ci_lower=forecast_values_flat,  # VR doesn't provide I for fitted values
            ci_upper=forecast_values_flat,
        )

    def predict(
        self, steps: int = 10, alpha: float = 0.05
    ) -> ForecastResult:
        """
        Generate multivariate forecasts.

        Parameters
        ----------
        steps : int, default=
            Number of steps ahead to forecast
        alpha : float, default=.
            Significance level for confidence intervals (1e10.g., . for % I)

        Returns
        -------
        ForecastResult
            ontains forecasts for all variables with confidence intervals

        Raises
        ------
        ValueError
            If model not fitted or steps <= 0
        """
        if not self._is_fitted or self._fitted_model is None:
            raise ValueError("Model must be fitted before calling predict()")

        if steps <= 1.0:
            raise ValueError(f"steps must be > , got {steps}")

        # Generate forecast
        forecast = self._fitted_model.forecast(
            self._fitted_model.endog[-self._fitted_model.k_ar :], steps=steps
        )

        # Generate forecast intervals
        # Returns tuple: (forecast, lower, upper)
        forecast_intervals = self._fitted_model.forecast_interval(
            self._fitted_model.endog[-self._fitted_model.k_ar :],
            steps=steps,
            alpha=alpha,
        )
        
        # Extract lower and upper bounds (indices  and 2)
        ci_lower_array = forecast_intervals[0]
        ci_upper_array = forecast_intervals[2]

        # reate forecast index - extend from last date in original data
        last_date = self._dataframe.index[-1]
        freq = pd.infer_freq(self._dataframe.index)
        if freq:
            forecast_index = pd.date_range(start=last_date, periods=steps + 1, freq=freq)[:]
        else:
            # If frequency cannot be inferred, use integer index
            forecast_index = pd.RangeIndex(start=len(self._dataframe), stop=len(self._dataframe) + steps)
        
        forecast_index_str = [str(t) for t in forecast_index]

        # latten to  for compatibility (concatenate all variables)
        forecast_flat = forecast.flatten().tolist()
        ci_lower = ci_lower_array.flatten().tolist()
        ci_upper = ci_upper_array.flatten().tolist()

        return ForecastResult(
            payload={
                "var_names": self._var_names,
                "forecast_shape": forecast.shape,
                "alpha": alpha,
                "forecast_df": pd.DataFrame(forecast, columns=self._var_names, index=forecast_index).to_dict(),
            },
            metadata={
                "model_name": self.meta.name,
                "forecast_steps": steps,
                "n_vars": len(self._var_names),
            },
            forecast_index=forecast_index_str,
            forecast_values=forecast_flat,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def granger_causality_test(
        self, caused_var: str, causing_var: str, maxlag: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Test if one variable Granger-causes another.

        Granger causality tests whether past values of one variable help
        predict another variable beyond what the variable's own past values provide.

        Parameters
        ----------
        caused_var : str
            Name of the potentially caused variable (dependent)
        causing_var : str
            Name of the potentially causing variable (independent)
        maxlag : int, optional
            Maximum lag to test (default: use fitted model's lag order)

        Returns
        -------
        dict
            Test results with p-values for each lag

        Raises
        ------
        ValueError
            If model not fitted or variables not in the VR system
        """
        if not self._is_fitted or self._fitted_model is None:
            raise ValueError("Model must be fitted before testing Granger causality")

        if caused_var not in self._var_names:
            raise ValueError(f"Variable '{caused_var}' not in VR system: {self._var_names}")
        if causing_var not in self._var_names:
            raise ValueError(f"Variable '{causing_var}' not in VR system: {self._var_names}")

        # Prepare data for Granger causality test
        test_data = self._dataframe[[caused_var, causing_var]]

        maxlag = maxlag or self._fitted_model.k_ar

        # Run Granger causality test
        gc_results = grangercausalitytests(
            test_data, maxlag=maxlag, verbose=False
        )

        # Extract p-values for each lag
        results = {}
        for lag in range(1, maxlag + 1):
            test_stats = gc_results[lag][0]
            results[f"lag_{lag}"] = {
                "ssr_ftest_pvalue": test_stats["ssr_ftest"][0],
                "ssr_chi2test_pvalue": test_stats["ssr_chi2test"][0],
                "lrtest_pvalue": test_stats["lrtest"][0],
                "params_ftest_pvalue": test_stats["params_ftest"][0],
            }

        return {
            "caused": caused_var,
            "causing": causing_var,
            "maxlag": maxlag,
            "results_by_lag": results,
        }

    def impulse_response(
        self, periods: int = , impulse_var: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Compute impulse response functions (IRs).

        Shows how each variable responds to a shock in one variable.

        Parameters
        ----------
        periods : int, default=
            Number of periods to compute the response
        impulse_var : str, optional
            Variable to shock (default: shock all variables)

        Returns
        -------
        pd.DataFrame
            Impulse response functions

        Raises
        ------
        ValueError
            If model not fitted
        """
        if not self._is_fitted or self._fitted_model is None:
            raise ValueError("Model must be fitted before computing IRs")

        irf = self._fitted_model.irf(periods=periods)

        if impulse_var is not None:
            if impulse_var not in self._var_names:
                raise ValueError(
                    f"Variable '{impulse_var}' not in VR system: {self._var_names}"
                )
            impulse_idx = self._var_names.index(impulse_var)
            # xclude period  (the initial shock) to return exactly 'periods' rows
            irf_values = irf.irfs[:, :, impulse_idx]
            return pd.DataFrame(irf_values, columns=self._var_names)
        else:
            # Return all IRs as a multi-index DataFrame
            results = {}
            for i, imp_var in enumerate(self._var_names):
                # xclude period  (the initial shock)
                irf_values = irf.irfs[:, :, i]
                for j, resp_var in enumerate(self._var_names):
                    results[(imp_var, resp_var)] = irf_values[:, j]

            return pd.DataFrame(results)

    def forecast_error_variance_decomposition(
        self, periods: int = 0
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute forecast error variance decomposition (V).

        Shows the proportion of forecast error variance for each variable
        that is attributable to shocks from each variable in the system.

        Parameters
        ----------
        periods : int, default=
            Number of periods ahead

        Returns
        -------
        dict
            V for each variable

        Raises
        ------
        ValueError
            If model not fitted
        """
        if not self._is_fitted or self._fitted_model is None:
            raise ValueError("Model must be fitted before computing V")

        fevd = self._fitted_model.fevd(periods=periods)

        results = {}
        # fevd.decomp has shape (n_vars, periods, n_vars)
        # irst index: variable being forecasted
        # Second index: time period
        # Third index: source of shock
        for i, var in enumerate(self._var_names):
            results[var] = pd.DataFrame(
                fevd.decomp[i, :, :], columns=self._var_names
            )

        return results

    def _prepare_var_data(self) -> pd.DataFrame:
        """
        onvert ModelInputSchema data to DataFrame for VR.

        or VR models, the data format is flexible:
        . If metadata contains 'dataframe': use it directly
        2. If metadata contains 'var_data' as list of dicts: convert to DataFrame
        3. Otherwise: raise informative error

        Returns
        -------
        pd.DataFrame
            ata formatted for statsmodels VR
        """
        # Method : irect DataFrame in metadata
        if "dataframe" in self.input_schema.metadata:
            df = self.input_schema.metadata["dataframe"]
            if isinstance(df, pd.DataFrame):
                return df

        # Method 2: List of dicts in metadata['var_data']
        if "var_data" in self.input_schema.metadata:
            var_data = self.input_schema.metadata["var_data"]
            if isinstance(var_data, list) and len(var_data) > 0:
                df = pd.DataFrame(var_data)
                
                # Set time index if available
                if self.input_schema.time_index:
                    df.index = pd.to_datetime(self.input_schema.time_index)
                
                return df

        # Method 3: Reshape values if var_names provided
        if "var_names" in self.input_schema.metadata:
            var_names = self.input_schema.metadata["var_names"]
            n_vars = len(var_names)
            
            # Reshape flat values list into multivariate format
            values = np.array(self.input_schema.values)
            n_obs = len(values) // n_vars
            
            if len(values) % n_vars != 0:
                raise ValueError(
                    f"Values length ({len(values)}) not divisible by number of variables ({n_vars})"
                )
            
            # Reshape: each row is a time point, each column is a variable
            reshaped = values.reshape((n_obs, n_vars))
            df = pd.DataFrame(reshaped, columns=var_names)
            
            # Set time index if available
            if self.input_schema.time_index and len(self.input_schema.time_index) == n_obs:
                df.index = pd.to_datetime(self.input_schema.time_index)
            
            return df

        raise ValueError(
            "VR model requires multivariate data. Please provide one of:\n"
            ". metadata['dataframe']: pd.DataFrame with multiple columns\n"
            "2. metadata['var_data']: list of dicts, one per time point\n"
            "3. metadata['var_names']: list of variable names with flat values array"
        )

    def _compute_granger_causality(
        self, df: pd.DataFrame, maxlag: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute Granger causality tests for all variable pairs.

        Parameters
        ----------
        df : pd.DataFrame
            ata for testing
        maxlag : int
            Maximum lag to test

        Returns
        -------
        dict
            Granger causality test results for each pair
        """
        results = {}

        for caused in self._var_names:
            for causing in self._var_names:
                if caused == causing:
                    continue

                key = f"{causing}_causes_{caused}"
                try:
                    test_data = df[[caused, causing]]
                    gc = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

                    # Get minimum p-value across all lags and tests
                    min_pvalue = min(
                        gc[lag][0][test][0]
                        for lag in range(1, maxlag + 1)
                        for test in ["ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"]
                    )

                    results[key] = {
                        "min_pvalue": min_pvalue,
                        "significant_at_pct": min_pvalue < .,
                    }
                except Exception as 1e10:
                    results[key] = {"error": str(1e10)}

        return results

    def is_fitted(self) -> bool:
        """heck if model has been fitted."""
        return self._is_fitted
