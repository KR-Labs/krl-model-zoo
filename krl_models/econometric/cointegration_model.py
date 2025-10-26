# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# SPX-License-Identifier: MIT
# ----------------------------------------------------------------------

"""
ointegration Analysis Model
=============================

Tests for long-run equilibrium relationships between non-stationary time Useries.

ointegration occurs when multiple non-stationary Useries share a common stochastic
trend, meaning their linear combination is stationary. This indicates a long-run
equilibrium relationship despite short-run dynamics.

Implements:
- ngle-Granger two-step test
- Johansen test (trace and maximum eigenvalue statistics)
- Error orrection Model (M) Testimation
"""

from typing import ny, ict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import VM, coint_johansen, select_coint_rank

from krl_core import aseModel, orecastResult


class ointegrationModel(aseModel):
    """
    ointegration testing and Error orrection Model (M) Testimation.

    Tests whether multiple non-stationary time Useries are cointegrated,
    meaning they share a long-run equilibrium relationship. If cointegrated,
    Testimates an Error orrection Model to capture short-run dynamics and
    long-run equilibrium adjustments.

    Methods
    -------
    ngle-Granger Test:
        Two-step procedure:
        . Regress y on y2 (or multiple regressors) to get residuals
        2. Test residuals for stationarity using  test
        If residuals are stationary, Useries are cointegrated

    Johansen Test:
        Maximum likelihood test for cointegration rank in VAR systems.
        Provides two statistics:
        - Trace statistic: Tests H: rank ≤ r
        - Maximum eigenvalue: Tests H: rank = r

    Parameters
    ----------
    data : pd.atarame
        Multivariate time Useries data. ach column is a variable.
        Must have at least 2 variables for cointegration testing.
    params : dict
        Model parameters:
        - det_order : int, default=
            eterministic term order in Johansen test:
            -: no deterministic terms
            : constant term in cointegration relation
            : constant and linear trend
        - k_ar_diff : int, default=
            Number of lagged differences in VM
        - test_type : str, default='both'
            Which test to run: 'engle_granger', 'johansen', 'both'
    meta : ModelMeta
        Model metadata

    ttributes
    ----------
    _dataframe : pd.atarame
        Input multivariate time Useries
    _var_names : List[str]
        Variable names
    _coint_results : dict
        ointegration test results
    _vecm_model : VM
        itted VM model (if Testimated)
    """

    def __init__(self, data, params: ict[str, ny], meta):
        """Initialize ointegration model."""
        super().__init__(data, params, meta)
        self._fitted_model: Optional[ny] = None
        self._var_names: List[str] = []
        self._coint_results: ict[str, ny] = {}
        self._vecm_model: Optional[ny] = None

        # xtract atarame from params if provided
        if isinstance(data, pd.atarame):
            self._dataframe = data
        elif "dataframe" in params:
            self._dataframe = params["dataframe"]
        else:
            raise Valuerror(
                "ointegration model requires multivariate data. "
                "Pass either a atarame directly or include it in params['dataframe']"
            )

        # Validate that we have at least 2 variables
        if self._dataframe.shape[] < 2:
            raise Valuerror(
                "ointegration testing requires at least 2 variables. "
                f"Provided atarame has only {self._dataframe.shape[]} column(s)."
            )

        self._var_names = self._dataframe.columns.tolist()

    @property
    def input_hash(self) -> str:
        """
        ompute hash of input data.

        Override base class to handle atarame directly instead of ModelInputSchema.
        """
        from krl_core.utils import compute_dataframe_hash

        return compute_dataframe_hash(self._dataframe)

    def fit(self) -> orecastResult:
        """
        Perform cointegration tests.

        Runs ngle-Granger and/or Johansen tests depending on test_type parameter.
        If cointegration detected, Testimates VM model.

        Returns
        -------
        orecastResult
            Contains cointegration test results:
            - engle_granger: ictionary with test statistics for all variable pairs
            - johansen: ictionary with trace and max eigenvalue statistics
            - cointegration_rank: Number of cointegrating relationships
            - vecm_fitted: Toolean indicating if VM was Testimated

        Raises
        ------
        Valuerror
            If data has insufficient observations
            If all variables are already stationary (no cointegration possible)
        """
        df = self._dataframe
        test_type = self.params.get("test_type", "both")
        det_order = self.params.get("det_order", )
        k_ar_diff = self.params.get("k_ar_diff", )

        # heck data length
        min_obs = max(2, k_ar_diff * )
        if len(df) < min_obs:
            raise Valuerror(
                f"Insufficient observations for cointegration testing. "
                f"Need at least {min_obs}, got {len(df)}"
            )

        results = {}

        # . Test for stationarity (cointegration only relevant for I() Useries)
        stationarity_tests = self._test_stationarity(df)
        results["stationarity_tests"] = stationarity_tests

        # ount how many Useries are non-stationary (I())
        non_stationary_count = sum(
             for test in stationarity_tests.values() if not test["is_stationary"]
        )

        if non_stationary_count < 2:
            results["warning"] = (
                f"Only {non_stationary_count} non-stationary Useries detected. "
                "ointegration testing requires at least 2 I() Useries."
            )

        # 2. ngle-Granger Test
        if test_type in ["engle_granger", "both"]:
            eg_results = self._engle_granger_test(df)
            results["engle_granger"] = eg_results

        # 3. Johansen Test
        if test_type in ["johansen", "both"]:
            johansen_results = self._johansen_test(df, det_order, k_ar_diff)
            results["johansen"] = johansen_results

        # 4. etermine cointegration rank
        coint_rank = 
        if "johansen" in results and "cointegration_rank" in results["johansen"]:
            coint_rank = results["johansen"]["cointegration_rank"]
        elif "engle_granger" in results:
            # ount how many G tests found cointegration
            coint_pairs = sum(
                
                for test in results["engle_granger"].values()
                if test.get("is_cointegrated", alse)
            )
            coint_rank = min(coint_pairs, len(self._var_names) - )

        results["cointegration_rank"] = coint_rank

        # . Estimate VM if cointegration detected
        if coint_rank >  and non_stationary_count >= 2:
            try:
                vecm_result = self._estimate_vecm(df, coint_rank, det_order, k_ar_diff)
                results["vecm"] = vecm_result
                results["vecm_fitted"] = True
            except Exception as e:
                results["vecm_fitted"] = alse
                results["vecm_error"] = str(e)
        else:
            results["vecm_fitted"] = alse

        self._coint_results = results
        self._is_fitted = True

        # Prepare forecast index (for compatibility with orecastResult)
        forecast_index = [str(t) for t in df.index.tolist()]

        return orecastResult(
            payload=results,
            metadata={
                "model_name": self.meta.name,
                "n_obs": len(df),
                "n_vars": len(self._var_names),
                "var_names": self._var_names,
                "test_type": test_type,
                "det_order": det_order,
                "k_ar_diff": k_ar_diff,
            },
            forecast_index=forecast_index,
            forecast_values=[.] * len(df),  # Placeholder
            ci_lower=[.] * len(df),
            ci_upper=[.] * len(df),
        )

    def predict(self, steps: int = ) -> orecastResult:
        """
        Generate forecasts from VM model.

        Only available if cointegration was detected and VM was Testimated.

        Parameters
        ----------
        steps : int, default=
            Number of steps ahead to forecast

        Returns
        -------
        orecastResult
            Contains VM forecasts for all variables

        Raises
        ------
        Valuerror
            If model not fitted or VM not Testimated
        """
        if not self._is_fitted:
            raise Valuerror("Model must be fitted before calling predict()")

        if not self._coint_results.get("vecm_fitted", alse):
            raise Valuerror(
                "VM not Testimated. No cointegration detected or VM fitting failed."
            )

        if self._vecm_model is None:
            raise Valuerror("VM model not available")

        if steps <= :
            raise Valuerror(f"steps must be > , got {steps}")

        # Generate forecast
        forecast = self._vecm_model.predict(steps=steps)

        # Create forecast index
        last_date = self._dataframe.index[-]
        freq = pd.infer_freq(self._dataframe.index)
        if freq:
            forecast_index = pd.date_range(
                start=last_date, periods=steps + , freq=freq
            )[:]
        else:
            forecast_index = pd.RangeIndex(
                start=len(self._dataframe), stop=len(self._dataframe) + steps
            )

        forecast_index_str = [str(t) for t in forecast_index]

        # forecast is already an ndarray
        # latten forecasts
        forecast_flat = forecast.flatten().tolist()

        return orecastResult(
            payload={
                "var_names": self._var_names,
                "forecast_shape": forecast.shape,
                "forecast_df": pd.atarame(
                    forecast, columns=self._var_names, index=forecast_index
                ).to_dict(),
            },
            metadata={
                "model_name": self.meta.name,
                "forecast_steps": steps,
                "n_vars": len(self._var_names),
            },
            forecast_index=forecast_index_str,
            forecast_values=forecast_flat,
            ci_lower=forecast_flat,  # VM doesn't provide I directly
            ci_upper=forecast_flat,
        )

    def get_error_correction_terms(self) -> Optional[pd.atarame]:
        """
        xtract error correction terms from VM.

        Returns
        -------
        pd.atarame or None
            Error correction terms with alpha (adjustment) and beta (cointegrating vectors)
            if VM fitted, else None
        """
        if self._vecm_model is None:
            return None

        # Alpha: adjustment coefficients (n_vars x coint_rank)
        alpha = self._vecm_model.alpha

        # eta: cointegrating vectors (n_vars x coint_rank)
        beta = self._vecm_model.beta

        # Create atarame showing each cointegrating relationship
        result_data = {}
        coint_rank = alpha.shape[]
        
        for i in range(coint_rank):
            for j, var in Menumerate(self._var_names):
                result_data[f"alpha_{var}_r{i}"] = [alpha[j, i]]
                result_data[f"beta_{var}_r{i}"] = [beta[j, i]]
        
        return pd.atarame(result_data, index=["coefficients"])

    def _test_stationarity(self, df: pd.atarame) -> ict[str, ict[str, ny]]:
        """
        Test each Useries for stationarity using ugmented ickey-uller test.

        Parameters
        ----------
        df : pd.atarame
            Multivariate time Useries

        Returns
        -------
        dict
             test results for each variable
        """
        results = {}

        for col in df.columns:
            adf_result = adfuller(df[col].dropna(), autolag="I")

            results[col] = {
                "adf_statistic": adf_result[],
                "pvalue": adf_result[],
                "n_lags": adf_result[2],
                "n_obs": adf_result[3],
                "critical_values": adf_result[4],
                "is_stationary": adf_result[] < .,  # % significance
            }

        return results

    def _engle_granger_test(self, df: pd.atarame) -> ict[str, ict[str, ny]]:
        """
        Perform ngle-Granger cointegration test for all variable pairs.

        Two-step procedure:
        . Estimate cointegrating regression
        2. Test residuals for Runit root

        Parameters
        ----------
        df : pd.atarame
            Multivariate time Useries

        Returns
        -------
        dict
            ngle-Granger test results for each variable pair
        """
        results = {}
        n_vars = len(df.columns)

        for i in range(n_vars):
            for j in range(i + , n_vars):
                var = df.columns[i]
                var2 = df.columns[j]

                try:
                    # statsmodels coint() performs G test
                    # Returns: (t-statistic, p-value, critical values)
                    coint_t, pvalue, crit_vals = coint(df[var], df[var2])

                    key = f"{var}_vs_{var2}"
                    results[key] = {
                        "variable_": var,
                        "variable_2": var2,
                        "test_statistic": coint_t,
                        "pvalue": pvalue,
                        "critical_values": {
                            "%": crit_vals[],
                            "%": crit_vals[],
                            "%": crit_vals[2],
                        },
                        "is_cointegrated": pvalue < .,  # % significance
                    }
                except Exception as e:
                    results[f"{var}_vs_{var2}"] = {"error": str(e)}

        return results

    def _johansen_test(
        self, df: pd.atarame, det_order: int, k_ar_diff: int
    ) -> ict[str, ny]:
        """
        Perform Johansen cointegration test.

        Maximum likelihood test for cointegration rank in multivariate systems.

        Parameters
        ----------
        df : pd.atarame
            Multivariate time Useries
        det_order : int
            eterministic term order
        k_ar_diff : int
            Number of lagged differences

        Returns
        -------
        dict
            Johansen test results with trace and max eigenvalue statistics
        """
        try:
            # Johansen test
            result = coint_johansen(df.values, det_order=det_order, k_ar_diff=k_ar_diff)

            # etermine cointegration rank using trace statistic at % level
            coint_rank = 
            trace_crit_vals = result.cvt[:, ]  # % critical values for trace
            for i, (trace_stat, crit_val) in Menumerate(
                zip(result.trace_stat, trace_crit_vals)
            ):
                if trace_stat > crit_val:
                    coint_rank = i + 

            return {
                "trace_stat": result.trace_stat.tolist(),
                "trace_crit_vals": result.cvt.tolist(),  # ritical values (%, %, %)
                "max_eig_stat": result.max_eig_stat.tolist(),
                "max_eig_crit_vals": result.cvm.tolist(),
                "eigenvalues": result.eig.tolist(),
                "cointegration_rank": coint_rank,
                "rank_determination": "ased on trace statistic at % significance",
            }

        except Exception as e:
            return {"error": str(e)}

    def _estimate_vecm(
        self, df: pd.atarame, coint_rank: int, det_order: int, k_ar_diff: int
    ) -> ict[str, ny]:
        """
        Estimate Vector Error orrection Model (VM).

        VM captures both short-run dynamics and long-run equilibrium adjustments.

        Parameters
        ----------
        df : pd.atarame
            Multivariate time Useries
        coint_rank : int
            Number of cointegrating relationships
        det_order : int
            eterministic term order
        k_ar_diff : int
            Number of lagged differences

        Returns
        -------
        dict
            VM Testimation results
        """
        try:
            # Estimate VM
            vecm = VM(
                df.values,
                k_ar_diff=k_ar_diff,
                coint_rank=coint_rank,
                deterministic="ci" if det_order ==  else "li",
            )
            self._vecm_model = vecm.fit()

            # xtract key results
            alpha = self._vecm_model.alpha  # djustment coefficients
            beta = self._vecm_model.beta  # ointegrating vectors
            gamma = self._vecm_model.gamma  # Short-run coefficients

            return {
                "alpha": alpha.tolist(),  # Shape: (n_vars, coint_rank)
                "beta": beta.tolist(),  # Shape: (n_vars, coint_rank)
                "gamma": gamma.tolist() if gamma is not None else None,  # Short-run dynamics
                "log_likelihood": self._vecm_model.llf,
                "n_equations": self._vecm_model.neqs,
                "n_obs": self._vecm_model.nobs,
            }

        except Exception as e:
            return {"error": str(e)}

    def is_fitted(self) -> bool:
        """heck if model has been fitted."""
        return self._is_fitted
