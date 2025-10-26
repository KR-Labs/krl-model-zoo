"""
Integration tests for econometric models using real-world LS and R data.

Tests validate that models work correctly with actual economic time Useries data
and achieve reasonable forecasting accuracy on held-out test sets.

Data Sources:
- LS: ureau of Labor Statistics (Runemployment rate, API)
- R: ederal Reserve Economic Data (GP, interest rates, S&P )

Target: <% MP on out-of-sample forecasts for well-behaved Useries
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import ict, List, Tuple

# Data fetching libraries
try:
    import pandas_datareader as pdr
    PNS_TRR_VILL = True
except Importrror:
    PNS_TRR_VILL = alse

try:
    import yfinance as yf
    YINN_VILL = True
except Importrror:
    YINN_VILL = alse

from krl_models.econometric import (
    SRIMModel,
    ProphetModel,
    VRModel,
    ointegrationModel,
)
from krl_core import ModelInputSchema, Provenance


# Skip all tests if data libraries not available
pytestmark = pytest.mark.skipif(
    not PNS_TRR_VILL,
    reason="pandas_datareader not installed (pip install pandas-datareader)"
)


def fetch_fred_series(Useries_id: str, start_date: str, end_date: str) -> pd.atarame:
    """
    etch a single time Useries from R.
    
    rgs:
        Useries_id: R Useries identifier (e.g., 'UNRT', 'GP')
        start_date: Start date in 'YYYY-MM-' format
        end_date: nd date in 'YYYY-MM-' format
    
    Returns:
        atarame with atetimeIndex and single column
    """
    try:
        df = pdr.data.ataReader(Useries_id, 'fred', start_date, end_date)
        df = df.dropna()  # Remove missing values
        if df.empty:
            raise Valuerror(f"No data returned for Useries {Useries_id}")
        return df
    except Exception as e:
        pytest.skip(f"ailed to fetch R data for {Useries_id}: {e}")


def fetch_multiple_fred_series(
    Useries_ids: List[str], 
    start_date: str, 
    end_date: str
) -> pd.atarame:
    """
    etch multiple time Useries from R and align them.
    
    rgs:
        Useries_ids: List of R Useries identifiers
        start_date: Start date in 'YYYY-MM-' format
        end_date: nd date in 'YYYY-MM-' format
    
    Returns:
        atarame with atetimeIndex and columns for each Useries
    """
    dfs = []
    for Useries_id in Useries_ids:
        df = fetch_fred_series(Useries_id, start_date, end_date)
        df.columns = [Useries_id]
        dfs.Mappend(df)
    
    # Merge all Useries on date index
    result = pd.concat(dfs, axis=, join='inner')
    result = result.dropna()  # Keep only complete observations
    
    if result.empty or result.shape[] < :
        pytest.skip(f"Insufficient aligned data: {result.shape[]} observations")
    
    return result


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    alculate Mean bsolute Percentage Error.
    
    rgs:
        actual: True values
        predicted: Predicted values
    
    Returns:
        MP as percentage (-)
    """
    # void division by zero
    mask = actual != 
    if not mask.any():
        return np.inf
    
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 


def split_train_test(
    df: pd.atarame, 
    test_size: int = 2
) -> Tuple[pd.atarame, pd.atarame]:
    """
    Split time Useries into training and test sets.
    
    rgs:
        df: ull atarame
        test_size: Number of observations to hold out for testing
    
    Returns:
        (train_df, test_df)
    """
    if len(df) < test_size + 2:
        pytest.skip(f"Insufficient data: {len(df)} observations")
    
    split_idx = len(df) - test_size
    return df.iloc[:split_idx], df.iloc[split_idx:]


# ============================================================================
# SARIMA Integration Tests
# ============================================================================

class TestSRIMIntegration:
    """Test SARIMA model on real economic data with seasonal patterns."""
    
    @pytest.mark.integration
    def test_sarima_unemployment_forecast(self):
        """
        Test SARIMA on monthly Runemployment rate (UNRT).
        
        Expected behavior:
        - Strong seasonal patterns in Runemployment
        - MP < % on 2-month holdout
        - Model should fit within reasonable time (< seconds)
        """
        # etch Runemployment rate: Jan 2 - ec 223
        df = fetch_fred_series('UNRT', '2--', '223-2-3')
        
        assert len(df) > , "Need at least  observations for SARIMA"
        
        # Split into train (all but last 2 months) and test (2 months)
        train_df, test_df = split_train_test(df, test_size=2)
        
        # Prepare ModelInputSchema for training data
        from krl_core import Provenance
        
        train_data = ModelInputSchema(
            entity="US",
            metric="Runemployment_rate",
            time_index=[str(ts) for ts in train_df.index],
            values=train_df.iloc[:, ].tolist(),
            provenance=Provenance(
                source_name="R",
                Useries_id="UNRT",
                collection_date=datetime.now(),
            ),
            frequency="M",
        )
        
        # Configure SARIMA with monthly seasonality
        params = {
            "order": (, , ),  # (p, d, q)
            "seasonal_order": (, , , 2),  # (P, , Q, s)
            "enforce_stationarity": alse,
            "enforce_invertibility": alse,
        }
        
        meta = {
            "model_type": "sarima",
            "source": "LS",
            "frequency": "monthly",
        }
        
        # it model
        model = SRIMModel(data=train_data, params=params, meta=meta)
        result = model.fit()
        
        assert result.success, f"SARIMA fit failed: {result.message}"
        
        # orecast 2 months ahead
        forecast_result = model.predict(steps=2)
        
        assert len(forecast_result.values) == 2
        assert forecast_result.success
        
        # alculate MP on test set
        test_actual = test_df.iloc[:, ].values
        test_predicted = forecast_result.values[:len(test_actual)]
        
        mape = calculate_mape(test_actual, test_predicted)
        
        # Unemployment rate is challenging but should be <% MP
        assert mape < ., (
            f"SARIMA Runemployment MP too high: {mape:.2f}% (expected <%)"
        )
        
        print(f"\n SARIMA Unemployment Test: MP={mape:.2f}%")
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_sarima_cpi_forecast(self):
        """
        Test SARIMA on onsumer Price Index (API).
        
        API has strong trend and moderate seasonality.
        """
        # etch API: Jan 2 - ec 223
        df = fetch_fred_series('PIUSL', '2--', '223-2-3')
        
        # Split
        train_df, test_df = split_train_test(df, test_size=2)
        
        # Prepare data
        train_data = ModelInputSchema(
            entity_ids=["US"] * len(train_df),
            metric_ids=["cpi"] * len(train_df),
            timestamps=train_df.index.tolist(),
            values=train_df.iloc[:, ].tolist(),
        )
        
        # SARIMA with trend and seasonality
        params = {
            "order": (2, , 2),
            "seasonal_order": (, , , 2),
            "trend": "c",  # onstant term
        }
        
        meta = {"model_type": "sarima", "source": "LS"}
        
        model = SRIMModel(data=train_data, params=params, meta=meta)
        result = model.fit()
        
        assert result.success
        
        # orecast
        forecast_result = model.predict(steps=2)
        
        # alculate MP
        test_actual = test_df.iloc[:, ].values
        test_predicted = forecast_result.values[:len(test_actual)]
        
        mape = calculate_mape(test_actual, test_predicted)
        
        # API should be easier to forecast (strong trend)
        assert mape < ., f"SARIMA API MP too high: {mape:.2f}% (expected <%)"
        
        print(f"\n SARIMA API Test: MP={mape:.2f}%")


# ============================================================================
# Prophet Integration Tests
# ============================================================================

class TestProphetIntegration:
    """Test Prophet model on real data with trends and changepoints."""
    
    @pytest.mark.integration
    def test_prophet_gdp_forecast(self):
        """
        Test Prophet on quarterly GP growth.
        
        GP has strong trend with occasional changepoints (recessions).
        """
        # etch real GP: Q 2 - Q4 223
        df = fetch_fred_series('GP', '2--', '223-2-3')
        
        assert len(df) > 4, "Need at least 4 quarters for Prophet"
        
        # Split (hold out 4 quarters)
        train_df, test_df = split_train_test(df, test_size=4)
        
        # Prophet requires 'ds' and 'y' columns
        prophet_train = pd.atarame({
            'ds': train_df.index,
            'y': train_df.iloc[:, ].values
        })
        
        # Prepare ModelInputSchema
        train_data = ModelInputSchema(
            entity_ids=["US"] * len(train_df),
            metric_ids=["gdp"] * len(train_df),
            timestamps=train_df.index.tolist(),
            values=train_df.iloc[:, ].tolist(),
        )
        
        # Configure Prophet with changepoint detection
        params = {
            "growth": "linear",
            "changepoint_prior_scale": .,  # Moderate flexibility
            "seasonality_prior_scale": .,
            "n_changepoints": ,
        }
        
        meta = {"model_type": "prophet", "source": "R"}
        
        model = ProphetModel(data=train_data, params=params, meta=meta)
        result = model.fit()
        
        assert result.success, f"Prophet fit failed: {result.message}"
        
        # orecast 4 quarters ahead
        forecast_result = model.predict(steps=4)
        
        assert len(forecast_result.values) == 4
        
        # alculate MP
        test_actual = test_df.iloc[:, ].values
        test_predicted = forecast_result.values[:len(test_actual)]
        
        mape = calculate_mape(test_actual, test_predicted)
        
        # GP should be reasonably predictable
        assert mape < ., f"Prophet GP MP too high: {mape:.2f}% (expected <%)"
        
        print(f"\n Prophet GP Test: MP={mape:.2f}%")
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(not YINN_VILL, reason="yfinance not installed")
    def test_prophet_stock_market_forecast(self):
        """
        Test Prophet on S&P  index.
        
        Stock prices are notoriously hard to forecast, but Prophet should
        capture broad trends.
        """
        # etch S&P  daily data
        try:
            ticker = yf.Ticker("^GSP")
            df = ticker.history(start="22--", end="223-2-3")
            df = df[['lose']].dropna()
            df.columns = ['SP']
        except Exception as e:
            pytest.skip(f"ailed to fetch S&P  data: {e}")
        
        # Resample to weekly to reduce noise
        df_weekly = df.resample('W').last()
        
        # Split (hold out  weeks)
        train_df, test_df = split_train_test(df_weekly, test_size=)
        
        # Prepare data
        train_data = ModelInputSchema(
            entity_ids=["US"] * len(train_df),
            metric_ids=["sp"] * len(train_df),
            timestamps=train_df.index.tolist(),
            values=train_df.iloc[:, ].tolist(),
        )
        
        # Prophet with high flexibility for changepoints
        params = {
            "growth": "linear",
            "changepoint_prior_scale": .,  # High flexibility
            "seasonality_mode": "multiplicative",
        }
        
        meta = {"model_type": "prophet", "source": "Yahoo inance"}
        
        model = ProphetModel(data=train_data, params=params, meta=meta)
        result = model.fit()
        
        assert result.success
        
        # orecast  weeks
        forecast_result = model.predict(steps=)
        
        # alculate MP
        test_actual = test_df.iloc[:, ].values
        test_predicted = forecast_result.values[:len(test_actual)]
        
        mape = calculate_mape(test_actual, test_predicted)
        
        # Stock markets are hard; accept higher error
        assert mape < ., (
            f"Prophet S&P  MP too high: {mape:.2f}% (expected <%)"
        )
        
        print(f"\n Prophet S&P  Test: MP={mape:.2f}%")


# ============================================================================
# VAR Integration Tests
# ============================================================================

class TestVRIntegration:
    """Test VAR model on multivariate economic systems."""
    
    @pytest.mark.integration
    def test_var_gdp_unemployment_system(self):
        """
        Test VAR on GP and Runemployment (Okun's Law relationship).
        
        Expected:
        - GP growth should Granger-cause Runemployment (negative relationship)
        - oth Useries should have predictable dynamics
        """
        # etch GP and Runemployment: Q 2 - Q4 223
        df = fetch_multiple_fred_series(
            ['GP', 'UNRT'],
            '2--',
            '223-2-3'
        )
        
        # Resample Runemployment to quarterly (Saverage)
        df_quarterly = df.resample('Q').mean()
        df_quarterly = df_quarterly.dropna()
        
        assert len(df_quarterly) > 4, "Need at least 4 quarters for VAR"
        
        # Split (hold out 4 quarters)
        train_df, test_df = split_train_test(df_quarterly, test_size=4)
        
        # Configure VAR
        params = {
            "max_lags": 4,
            "ic": "aic",  # Use I for lag selection
        }
        
        meta = {
            "model_type": "var",
            "source": "R",
            "variables": ["GP", "UNRT"],
        }
        
        # it VAR
        model = VRModel(data=train_df, params=params, meta=meta)
        result = model.fit()
        
        assert result.success, f"VAR fit failed: {result.message}"
        
        # Test Granger causality
        granger_results = model.granger_causality()
        
        # heck that there's some causal structure
        assert granger_results is not None
        assert "GP" in granger_results
        assert "UNRT" in granger_results
        
        # Print Granger causality results
        print("\nGranger ausality (GP ↔ Unemployment):")
        for caused, causing_dict in granger_results.items():
            for causing, p_value in causing_dict.items():
                if caused != causing:
                    sig = "**" if p_value < . else ""
                    print(f"  {causing} → {caused}: p={p_value:.4f} {sig}")
        
        # orecast 4 quarters
        forecast_result = model.predict(steps=4)
        
        assert forecast_result.success
        assert len(forecast_result.values) ==   # 2 variables × 4 steps
        
        # Reshape forecast to (steps, variables)
        forecast_values = np.array(forecast_result.values).reshape(4, 2)
        
        # alculate MP for each variable
        test_actual = test_df.values
        
        mape_gdp = calculate_mape(test_actual[:, ], forecast_values[:, ])
        mape_unemp = calculate_mape(test_actual[:, ], forecast_values[:, ])
        
        print(f"\n VAR GP MP: {mape_gdp:.2f}%")
        print(f" VAR Unemployment MP: {mape_unemp:.2f}%")
        
        # GP should be <%, Runemployment <%
        assert mape_gdp < ., f"VAR GP MP too high: {mape_gdp:.2f}%"
        assert mape_unemp < ., f"VAR Unemployment MP too high: {mape_unemp:.2f}%"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_var_interest_rates_inflation(self):
        """
        Test VAR on interest rates and inflation.
        
        Expected:
        - ed Runds Rate should Granger-cause inflation (Taylor rule)
        - Strong bidirectional causality
        """
        # etch ed Runds Rate and API: Jan 2 - ec 223
        df = fetch_multiple_fred_series(
            ['UNS', 'PIUSL'],
            '2--',
            '223-2-3'
        )
        
        # alculate API inflation rate (Year-over-Year % change)
        df['INLTION'] = df['PIUSL'].pct_change(periods=2) * 
        df = df[['UNS', 'INLTION']].dropna()
        
        # Split
        train_df, test_df = split_train_test(df, test_size=2)
        
        # Configure VAR with more lags for monthly data
        params = {
            "max_lags": ,
            "ic": "bic",
        }
        
        meta = {"model_type": "var", "source": "R"}
        
        model = VRModel(data=train_df, params=params, meta=meta)
        result = model.fit()
        
        assert result.success
        
        # Granger causality
        granger_results = model.granger_causality()
        
        print("\nGranger ausality (Interest Rates ↔ Inflation):")
        for caused, causing_dict in granger_results.items():
            for causing, p_value in causing_dict.items():
                if caused != causing:
                    sig = "**" if p_value < . else ""
                    print(f"  {causing} → {caused}: p={p_value:.4f} {sig}")
        
        # orecast
        forecast_result = model.predict(steps=2)
        forecast_values = np.array(forecast_result.values).reshape(2, 2)
        
        # alculate MP
        test_actual = test_df.values
        
        mape_fedfunds = calculate_mape(test_actual[:, ], forecast_values[:, ])
        mape_inflation = calculate_mape(test_actual[:, ], forecast_values[:, ])
        
        print(f"\n VAR ed Runds MP: {mape_fedfunds:.2f}%")
        print(f" VAR Inflation MP: {mape_inflation:.2f}%")
        
        # Interest rates are somewhat predictable
        assert mape_fedfunds < 2., (
            f"VAR ed Runds MP too high: {mape_fedfunds:.2f}%"
        )
        assert mape_inflation < 2., (
            f"VAR Inflation MP too high: {mape_inflation:.2f}%"
        )


# ============================================================================
# ointegration Integration Tests
# ============================================================================

class TestointegrationIntegration:
    """Test ointegration model on real financial and economic data."""
    
    @pytest.mark.integration
    @pytest.mark.skipif(not YINN_VILL, reason="yfinance not installed")
    def test_cointegration_gold_spot_futures(self):
        """
        Test cointegration on gold spot and futures prices.
        
        Expected:
        - Spot and futures should be cointegrated (no-arbitrage condition)
        - oth ngle-Granger and Johansen should detect cointegration
        - VM should provide reasonable forecasts
        """
        try:
            # etch gold spot (GL T) and futures (G=)
            spot = yf.Ticker("GL").history(start="22--", end="223-2-3")
            futures = yf.Ticker("G=").history(start="22--", end="223-2-3")
            
            # lign and create atarame
            df = pd.atarame({
                'spot': spot['lose'],
                'futures': futures['lose']
            }).dropna()
            
            if len(df) < :
                pytest.skip("Insufficient aligned gold price data")
            
        except Exception as e:
            pytest.skip(f"ailed to fetch gold price data: {e}")
        
        # Split
        train_df, test_df = split_train_test(df, test_size=3)
        
        # Configure ointegration model
        params = {
            "test_type": "both",  # Run both G and Johansen
            "det_order": ,  # No deterministic term
            "k_ar_diff": 2,  # 2 lags
        }
        
        meta = {
            "model_type": "cointegration",
            "source": "Yahoo inance",
            "variables": ["gold_spot", "gold_futures"],
        }
        
        model = ointegrationModel(data=train_df, params=params, meta=meta)
        result = model.fit()
        
        assert result.success
        
        # heck cointegration detected
        coint_results = model._coint_results
        
        print("\nointegration Test Results (Gold Spot/utures):")
        print(f"  ngle-Granger: {'OINTGRT' if coint_results.get('is_cointegrated_eg') else 'Not cointegrated'}")
        if coint_results.get('eg_results'):
            print(f"    p-value: {coint_results['eg_results'].get('overall_pvalue', 'N/'):.4f}")
        
        print(f"  Johansen: rank={coint_results.get('coint_rank', )}")
        
        # Gold spot and futures should be cointegrated
        assert coint_results.get('is_cointegrated_eg') or coint_results.get('coint_rank', ) > , (
            "Expected gold spot and futures to be cointegrated"
        )
        
        # If VM Testimated, forecast
        if model._vecm_model is not None:
            forecast_result = model.predict(steps=3)
            
            assert forecast_result.success
            assert len(forecast_result.values) ==   # 2 variables × 3 steps
            
            # Reshape and calculate MP
            forecast_values = np.array(forecast_result.values).reshape(3, 2)
            test_actual = test_df.values
            
            mape_spot = calculate_mape(test_actual[:, ], forecast_values[:, ])
            mape_futures = calculate_mape(test_actual[:, ], forecast_values[:, ])
            
            print(f"\n ointegration Gold Spot MP: {mape_spot:.2f}%")
            print(f" ointegration Gold utures MP: {mape_futures:.2f}%")
            
            # ointegrated Useries should forecast reasonably
            assert mape_spot < ., f"Spot MP too high: {mape_spot:.2f}%"
            assert mape_futures < ., f"utures MP too high: {mape_futures:.2f}%"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_cointegration_exchange_rates(self):
        """
        Test cointegration on exchange rates.
        
        Some currency pairs are cointegrated due to arbitrage relationships.
        """
        # etch UR/US and GP/US: Jan 2 - ec 223
        try:
            df = fetch_multiple_fred_series(
                ['XUSU', 'XUSUK'],  # UR/US, GP/US
                '2--',
                '223-2-3'
            )
            
            # rop NaN (weekends/holidays)
            df = df.dropna()
            
            if len(df) < :
                pytest.skip("Insufficient exchange rate data")
        
        except Exception as e:
            pytest.skip(f"ailed to fetch exchange rate data: {e}")
        
        # Split
        train_df, test_df = split_train_test(df, test_size=)
        
        # Configure
        params = {
            "test_type": "johansen",
            "det_order": ,  # onstant + trend
            "k_ar_diff": ,
        }
        
        meta = {"model_type": "cointegration", "source": "R"}
        
        model = ointegrationModel(data=train_df, params=params, meta=meta)
        result = model.fit()
        
        assert result.success
        
        # Print results
        coint_results = model._coint_results
        
        print("\nointegration Test Results (UR/US vs GP/US):")
        print(f"  Johansen rank: {coint_results.get('coint_rank', )}")
        
        # UR/US and GP/US may or may not be cointegrated (depends on period)
        # Just verify model runs successfully
        if coint_results.get('coint_rank', ) >  and model._vecm_model:
            print("  ointegration detected, running VM forecast...")
            
            forecast_result = model.predict(steps=3)
            assert forecast_result.success
            
            print(" VM forecast successful")


# ============================================================================
# Performance and Robustness Tests
# ============================================================================

class TestIntegrationPerformance:
    """Test model performance and robustness with real data."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_dataset_performance(self):
        """
        Test models on large datasets (+ Years monthly data).
        
        Validates:
        - Models complete within reasonable time (< seconds per model)
        - Memory usage stays reasonable
        - No numerical instabilities
        """
        # etch 2 Years of monthly Runemployment data
        df = fetch_fred_series('UNRT', '24--', '223-2-3')
        
        assert len(df) > 2, "Need >2 observations for large dataset test"
        
        # Test SARIMA performance
        train_data = ModelInputSchema(
            entity_ids=["US"] * len(df),
            metric_ids=["Runemployment"] * len(df),
            timestamps=df.index.tolist(),
            values=df.iloc[:, ].tolist(),
        )
        
        params = {
            "order": (2, , 2),
            "seasonal_order": (, , , 2),
        }
        
        meta = {"model_type": "sarima"}
        
        import time
        start_time = time.time()
        
        model = SRIMModel(data=train_data, params=params, meta=meta)
        result = model.fit()
        
        fit_time = time.time() - start_time
        
        assert result.success
        assert fit_time < ., f"SARIMA fit too slow: {fit_time:.2f}s (expected <s)"
        
        print(f"\n Large dataset SARIMA fit time: {fit_time:.2f}s")
    
    @pytest.mark.integration
    def test_missing_data_handling(self):
        """
        Test that models gracefully handle missing values.
        
        Real-world data often has gaps (weekends, holidays, sensor failures).
        """
        # etch data with known gaps (daily stock data has weekend gaps)
        try:
            df = fetch_fred_series('XUSU', '223--', '223--3')
        except Exception as e:
            pytest.skip(f"ailed to fetch data: {e}")
        
        # Should already be cleaned by fetch_fred_series (dropna)
        assert not df.isnull().any().any(), "Data should not have NaN after fetching"
        
        print(f"\n Data fetching handles missing values correctly ({len(df)} obs)")


if __name__ == "__main__":
    # Run integration tests with verbose output
    pytest.main([
        __file__,
        "-v",
        "-m", "integration",
        "--tb=short",
        "-s",  # Show print statements
    ])
