"""
Shift-Share Analysis for Regional Growth Decomposition
=======================================================

MIT License - Gate 2 Phase 2.4 Regional Specialization
author: KR Labs
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class ShiftShareModel(BaseModel):
    """
    Shift-Share Analysis for decomposing regional economic growth.
    
    Decomposes regional employment change into three components:
    1. National Share (NS): Growth if region matched national rate
    2. Industry Mix (IM): Effect of regional industry composition
    3. Competitive Share (CS): Regional competitive advantage
    
    Total Change = NS + IM + CS
    
    Parameters
    ----------
    industry_col : str, default='industry'
        Column name for industry identifier
    employment_col : str, default='employment'
        Column name for employment values
    region_col : str, default='region'
        Column name for region identifier
    time_col : str, default='year'
        Column name for time period
    """
    
    def __init__(
        self,
        industry_col: str = 'industry',
        employment_col: str = 'employment',
        region_col: str = 'region',
        time_col: str = 'year',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.industry_col = industry_col
        self.employment_col = employment_col
        self.region_col = region_col
        self.time_col = time_col
        
        self.results_ = None
        self.base_period_ = None
        self.end_period_ = None
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> 'ShiftShareModel':
        """
        Compute shift-share decomposition.
        
        Parameters
        ----------
        y : array-like or DataFrame
            Employment data across two time periods
        X : DataFrame, optional
            DataFrame with industry, region, and time columns
        **kwargs : dict
            Additional arguments (data_df if y is not DataFrame)
        
        Returns
        -------
        self : ShiftShareModel
            Fitted model with shift-share results
        """
        # Handle DataFrame input
        if isinstance(y, pd.DataFrame):
            df = y
        elif X is not None and isinstance(X, pd.DataFrame):
            df = X.copy()
            df[self.employment_col] = y
        else:
            raise ValueError("Must provide DataFrame with industry/region/time columns")
        
        # Get time periods
        periods = sorted(df[self.time_col].unique())
        if len(periods) != 2:
            raise ValueError("Shift-share requires exactly 2 time periods")
        
        self.base_period_, self.end_period_ = periods
        
        # Split data by period
        df_base = df[df[self.time_col] == self.base_period_]
        df_end = df[df[self.time_col] == self.end_period_]
        
        # Compute growth rates
        results_list = []
        
        for region in df_base[self.region_col].unique():
            region_base = df_base[df_base[self.region_col] == region]
            region_end = df_end[df_end[self.region_col] == region]
            
            # Merge base and end data
            merged = region_base.merge(
                region_end,
                on=[self.region_col, self.industry_col],
                suffixes=('_base', '_end')
            )
            
            # Regional employment
            E_ir_0 = merged[f'{self.employment_col}_base'].sum()
            E_ir_t = merged[f'{self.employment_col}_end'].sum()
            
            # National employment
            E_n_0 = df_base[self.employment_col].sum()
            E_n_t = df_end[self.employment_col].sum()
            
            # National growth rate
            g_n = (E_n_t - E_n_0) / E_n_0 if E_n_0 > 0 else 0
            
            # National Share (NS): If region grew at national rate
            NS = E_ir_0 * g_n
            
            # Industry Mix (IM) and Competitive Share (CS)
            IM = 0
            CS = 0
            
            for _, row in merged.iterrows():
                E_ir_i_0 = row[f'{self.employment_col}_base']
                E_ir_i_t = row[f'{self.employment_col}_end']
                
                industry = row[self.industry_col]
                
                # National industry employment
                E_n_i_0 = df_base[df_base[self.industry_col] == industry][self.employment_col].sum()
                E_n_i_t = df_end[df_end[self.industry_col] == industry][self.employment_col].sum()
                
                # Industry growth rate
                g_i = (E_n_i_t - E_n_i_0) / E_n_i_0 if E_n_i_0 > 0 else 0
                
                # Regional industry growth rate
                g_ir = (E_ir_i_t - E_ir_i_0) / E_ir_i_0 if E_ir_i_0 > 0 else 0
                
                # Industry Mix: Difference between industry and national growth
                IM += E_ir_i_0 * (g_i - g_n)
                
                # Competitive Share: Difference between regional and industry growth
                CS += E_ir_i_0 * (g_ir - g_i)
            
            # Total change
            actual_change = E_ir_t - E_ir_0
            
            results_list.append({
                'region': region,
                'base_employment': E_ir_0,
                'end_employment': E_ir_t,
                'actual_change': actual_change,
                'national_share': NS,
                'industry_mix': IM,
                'competitive_share': CS,
                'total_decomposed': NS + IM + CS
            })
        
        self.results_ = pd.DataFrame(results_list)
        
        return self
    
    def get_results(self) -> pd.DataFrame:
        """
        Get shift-share decomposition results.
        
        Returns
        -------
        results : DataFrame
            Decomposition for each region (NS, IM, CS)
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted to get results")
        
        return self.results_
    
    def get_region_summary(self, region: str) -> Dict[str, float]:
        """
        Get shift-share summary for specific region.
        
        Parameters
        ----------
        region : str
            Region identifier
        
        Returns
        -------
        summary : dict
            National share, industry mix, competitive share components
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted to get summary")
        
        region_data = self.results_[self.results_['region'] == region]
        
        if region_data.empty:
            raise ValueError(f"Region '{region}' not found in results")
        
        return region_data.iloc[0].to_dict()
    
    def forecast(self, steps: int = 1, X_future: Optional[np.ndarray] = None, **kwargs) -> ForecastResult:
        """
        Not directly applicable for shift-share (descriptive decomposition).
        
        Raises
        ------
        NotImplementedError
            Shift-share is a descriptive decomposition, not forecasting model
        """
        raise NotImplementedError("Shift-share is a descriptive decomposition, not a forecasting model")
