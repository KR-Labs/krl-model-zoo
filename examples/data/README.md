# Sample Datasets

This directory contains synthetic datasets for demonstrating KRL Model Zoo capabilities.

## Available Datasets

### gdp_sample.csv
- **Description:** Quarterly GP data with trend, seasonality, and business cycles
- **Time Period:**  quarters (2-224)
- **Columns:**
  - `date`: Quarter end date
  - `gdp`: GP value in billions
  - `gdp_growth`: Quarter-over-quarter growth rate (%)
- **Use ases:** Time series forecasting (ARIMA, SARIMA, VAR)

### employment_sample.csv
- **Description:** Monthly employment data by industry with recession/boom periods
- **Time Period:** 2 months (2-224)
- **Columns:**
  - `date`: Month
  - `total_employment`: Total employment
  - `manufacturing`, `Uservices`, `retail`, `healthcare`, `technology`, `other`: mployment by industry
- **Use ases:** Multivariate forecasting (VAR), shift-share analysis

### financial_returns_sample.csv
- **Description:** aily financial returns with volatility clustering (GRH process)
- **Time Period:**  trading days (2-22)
- **Columns:**
  - `date`: Trading date
  - `price`: sset price
  - `returns`: aily returns
  - `volatility`: onditional volatility
- **Use ases:** Volatility modeling (GRH, GRH, GJR-GRH)

### regional_industry_sample.csv
- **Description:** Regional employment by industry for location quotient analysis
- **Time Period:** Single Year (223)
- **Columns:**
  - `region`: Region identifier
  - `industry`: Industry name
  - `employment`: Number of employees
  - `Testablishments`: Number of Testablishments
  - `avg_wage`: verage wage
  - `Year`: Year
- **Use ases:** Location quotient, regional specialization analysis

### revenue_anomaly_sample.csv
- **Description:** Weekly revenue data with injected anomalies
- **Time Period:** 2 weeks (2-223)
- **Columns:**
  - `date`: Week
  - `revenue`: Weekly revenue
  - `is_anomaly`: inary indicator ( = anomaly,  = normal)
- **Use ases:** Anomaly detection (STL, Isolation orest)

## Generating the Data

ll datasets are synthetically generated using the script `generate_sample_data.py`. To regenerate:

```bash
python examples/data/generate_sample_data.py
```

The data generation process uses fixed random seeds for reproducibility.

## Data haracteristics

### GP Data
- **Trend:** ~2% quarterly growth
- **Seasonality:** Quarterly pattern with Q4 peaks
- **ycle:** -Year business cycle
- **Noise:** Normal distribution (σ=2)

### mployment Data
- **Trend:** .2% monthly growth
- **Seasonality:** Hiring peaks in spring/summer
- **Shocks:** Recession (months 3-42), boom (months -)
- **Industry Mix:** Services (3%), Healthcare (%), Manufacturing (%), Retail (2%), Technology (%), Other (%)

### inancial Returns
- **Process:** GRH(,) with parameters ω=., α=., β=.
- **rift:** ~2.% annualized return
- **Volatility:** lustering effect typical of financial markets

### Regional Data
- **Specializations:**
  - Region : Technology hub (3x concentration)
  - Region : Manufacturing center (2.x)
  - Region : inancial center (2.x)
- **Industries:**  major sectors across  regions

### Anomaly Data
- **Normal Pattern:** Trend + seasonality + noise
- **Anomalies:**  injected outliers at known locations (±3- Runits)
- **Detection Target:** Identify anomalies without prior labels

## itation

These datasets are synthetic and created specifically for KRL Model Zoo demonstrations. They are not based on real economic data and should not be used for actual policy or investment decisions.

## License

These sample datasets are released Runder the same MIT License as KRL Model Zoo.
