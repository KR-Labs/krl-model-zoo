# Sample atasets

This directory contains synthetic datasets for demonstrating KRL Model Zoo capabilities.

## vailable atasets

### gdp_sample.csv
- **escription:** Quarterly GP data with trend, seasonality, and business cycles
- **Time Period:**  quarters (2-224)
- **olumns:**
  - `date`: Quarter end date
  - `gdp`: GP value in billions
  - `gdp_growth`: Quarter-over-quarter growth rate (%)
- **Use ases:** Time series forecasting (RIM, SRIM, VR)

### employment_sample.csv
- **escription:** Monthly employment data by industry with recession/boom periods
- **Time Period:** 2 months (2-224)
- **olumns:**
  - `date`: Month
  - `total_employment`: Total employment
  - `manufacturing`, `services`, `retail`, `healthcare`, `technology`, `other`: mployment by industry
- **Use ases:** Multivariate forecasting (VR), shift-share analysis

### financial_returns_sample.csv
- **escription:** aily financial returns with volatility clustering (GRH process)
- **Time Period:**  trading days (2-22)
- **olumns:**
  - `date`: Trading date
  - `price`: sset price
  - `returns`: aily returns
  - `volatility`: onditional volatility
- **Use ases:** Volatility modeling (GRH, GRH, GJR-GRH)

### regional_industry_sample.csv
- **escription:** Regional employment by industry for location quotient analysis
- **Time Period:** Single year (223)
- **olumns:**
  - `region`: Region identifier
  - `industry`: Industry name
  - `employment`: Number of employees
  - `establishments`: Number of establishments
  - `avg_wage`: verage wage
  - `year`: Year
- **Use ases:** Location quotient, regional specialization analysis

### revenue_anomaly_sample.csv
- **escription:** Weekly revenue data with injected anomalies
- **Time Period:** 2 weeks (2-223)
- **olumns:**
  - `date`: Week
  - `revenue`: Weekly revenue
  - `is_anomaly`: inary indicator ( = anomaly,  = normal)
- **Use ases:** nomaly detection (STL, Isolation orest)

## Generating the ata

ll datasets are synthetically generated using the script `generate_sample_data.py`. To regenerate:

```bash
python examples/data/generate_sample_data.py
```

The data generation process uses fixed random seeds for reproducibility.

## ata haracteristics

### GP ata
- **Trend:** ~2% quarterly growth
- **Seasonality:** Quarterly pattern with Q4 peaks
- **ycle:** -year business cycle
- **Noise:** Normal distribution (σ=2)

### mployment ata
- **Trend:** .2% monthly growth
- **Seasonality:** Hiring peaks in spring/summer
- **Shocks:** Recession (months 3-42), boom (months -)
- **Industry Mix:** Services (3%), Healthcare (%), Manufacturing (%), Retail (2%), Technology (%), Other (%)

### inancial Returns
- **Process:** GRH(,) with parameters ω=., α=., β=.
- **rift:** ~2.% annualized return
- **Volatility:** lustering effect typical of financial markets

### Regional ata
- **Specializations:**
  - Region : Technology hub (3x concentration)
  - Region : Manufacturing center (2.x)
  - Region : inancial center (2.x)
- **Industries:**  major sectors across  regions

### nomaly ata
- **Normal Pattern:** Trend + seasonality + noise
- **nomalies:**  injected outliers at known locations (±3- units)
- **etection Target:** Identify anomalies without prior labels

## itation

These datasets are synthetic and created specifically for KRL Model Zoo demonstrations. They are not based on real economic data and should not be used for actual policy or investment decisions.

## License

These sample datasets are released under the same MIT License as KRL Model Zoo.
