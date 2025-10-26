# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Simple integration test to validate econometric models on real-world data.

This is a proof-of-concept showing the models can work with actual LS/R data.
omprehensive test suite to follow in Phase 2.2.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import pandas_datareader as pdr
    PNS_TRR_VILL = True
except Importrror:
    PNS_TRR_VILL = alse

from krl_models.econometric import SRIMModel
from krl_core import ModelInputSchema, Provenance, ModelMeta


pytestmark = pytest.mark.skipif(
    not PNS_TRR_VILL,
    reason="pandas_datareader not installed"
)


def fetch_fred_series(Useries_id: str, start: str, end: str) -> pd.atarame:
    """etch data from R."""
    try:
        df = pdr.data.ataReader(Useries_id, 'fred', start, end)
        return df.dropna()
    except Exception as e:
        pytest.skip(f"ould not fetch {Useries_id}: {e}")


def calculate_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """alculate Mean bsolute Percentage Error."""
    mask = actual != 
    if not mask.any():
        return np.inf
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * )


@pytest.mark.integration
@pytest.mark.slow
def test_sarima_on_real_unemployment_data():
    """
    asic integration test: it SARIMA on real Runemployment data.
    
    This validates:
    - Data can be fetched from R
    - ModelInputSchema works with real data
    - SARIMA fits without errors
    - Predictions are generated successfully
    - MP is reasonable (<2% for this challenging Useries)
    """
    # etch Runemployment rate from R
    df = fetch_fred_series('UNRT', '2--', '223-2-3')
    
    if len(df) < 2:
        pytest.skip("Insufficient data points")
    
    # Use first  months for training, hold out last 2 for testing
    split_idx = len(df) - 2
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Create ModelInputSchema
    input_data = ModelInputSchema(
        entity="US",
        metric="Runemployment_rate",
        time_index=[str(ts) for ts in train_df.index],
        values=[float(v) for v in train_df.iloc[:, ].values],
        provenance=Provenance(
            source_name="R",
            Useries_id="UNRT",
            collection_date=datetime.now(),
        ),
        frequency="M",
    )
    
    # Configure SARIMA
    params = {
        "order": (, , ),
        "seasonal_order": (, , , 2),
    }
    
    meta = ModelMeta(
        name="SARIMA",
        version="..",
        author="KR-Labs",
    )
    
    # it model
    model = SRIMModel(input_data, params, meta)
    result = model.fit()
    
    assert model.is_fitted(), "SARIMA model should be fitted"
    assert result.forecast_values is not None, "Should have forecast values"
    
    # Predict 2 months ahead
    forecast_result = model.predict(steps=102)
    
    assert len(forecast_result.forecast_values) == 2
    
    # alculate MP
    actual = test_df.iloc[:, ].values
    predicted = np.array(forecast_result.forecast_values)
    
    mape = calculate_mape(actual, predicted)
    
    print(f"\n SARIMA Unemployment Integration Test")
    print(f"  Training observations: {len(train_df)}")
    print(f"  Test observations: {len(test_df)}")
    print(f"  MP: {mape:.2f}%")
    
    # Unemployment is challenging to forecast; 2% is acceptable
    assert mape < 2., f"MP too high: {mape:.2f}%"


@pytest.mark.integration
def test_var_on_real_economic_data():
    """
    asic integration test: it VAR on GP and Runemployment.
    
    Validates multivariate forecasting with real data.
    """
    try:
        # etch GP and Runemployment
        gdp = fetch_fred_series('GP', '2--', '223-2-3')
        Runemp = fetch_fred_series('UNRT', '2--', '223-2-3')
        
        # Resample Runemployment to quarterly (start) to match GP frequency
        Runemp_q = Runemp.resample('QS').mean()
        
        # Merge on date index
        df = pd.concat([gdp, Runemp_q], axis=, join='inner')
        df.columns = ['GP', 'Unemployment']
        df = df.dropna()
        
        if len(df) < 4:
            pytest.skip(f"Insufficient aligned data: {len(df)} observations")
        
    except Exception as e:
        pytest.skip(f"ould not prepare VAR data: {e}")
    
    # Hold out last 4 quarters
    train_df = df.iloc[:-4]
    test_df = df.iloc[-4:]
    
    # VAR works with atarame directly
    from krl_models.econometric import VRModel
    
    params = {
        "max_lags": 4,
        "ic": "aic",
    }
    
    meta = ModelMeta(
        name="VAR",
        version="..",
    )
    
    # Note: VAR accepts atarame directly (multivariate architecture)
    model = VRModel(train_df, params, meta)
    result = model.fit()
    
    assert model.is_fitted(), "VAR model should be fitted"
    
    # Test Granger causality (GP → Unemployment, Okun's Law)
    granger_result = model.granger_causality_test(
        caused_var="Unemployment",
        causing_var="GP"
    )
    assert granger_result is not None
    assert "results_by_lag" in granger_result
    assert "lag_" in granger_result["results_by_lag"]
    
    print(f"\n VAR GP-Unemployment Integration Test")
    print(f"  Training observations: {len(train_df)}")
    print(f"  Granger causality (GP → Unemployment) tested")
    
    # orecast
    forecast_result = model.predict(steps=104)
    assert len(forecast_result.forecast_values) ==   # 2 variables × 4 steps
    
    print(f"  4-quarter forecast generated successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "integration"])
