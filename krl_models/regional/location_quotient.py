# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Location Quotient for Regional Analysis
========================================

MIT License - Gate 2 Phase 2.4 Regional Specialization
author: KR Labs
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from krl_core.base_model import BaseModel
from krl_core.results import ForecastResult


class LocationQuotientModel(BaseModel):
    """
    Location Quotient (LQ) for measuring regional industry concentration.
    
    LQ = (Regional Industry Employment / Regional Total Employment) /
         (National Industry Employment / National Total Employment)
    
    Interpretation:
    - LQ > 1: Region is specialized in industry (exports)
    - LQ = 1: Region matches national average
    - LQ < 1: Region is under-concentrated (imports)
    
    Parameters
    ----------
    industry_col : str, default='industry'
        Column name for industry identifier
    employment_col : str, default='employment'
        Column name for employment values
    region_col : str, default='region'
        Column name for region identifier
    """
    
    def __init__(
        self,
        industry_col: str = 'industry',
        employment_col: str = 'employment',
        region_col: str = 'region',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.industry_col = industry_col
        self.employment_col = employment_col
        self.region_col = region_col
        
        self.lq_matrix_ = None
        self.regional_totals_ = None
        self.national_totals_ = None
        self.national_total_ = None
    
    def fit(self, y: np.ndarray, X: Optional[np.ndarray] = None, **kwargs) -> 'LocationQuotientModel':
        """
        Compute location quotients from regional employment data.
        
        Parameters
        ----------
        y : array-like or DataFrame
            Employment data (if array, must provide X with industry/region info)
        X : DataFrame, optional
            DataFrame with industry and region columns
        **kwargs : dict
            Additional arguments (data_df if y is not DataFrame)
        
        Returns
        -------
        self : LocationQuotientModel
            Fitted model with LQ matrix
        """
        # Handle DataFrame input
        if isinstance(y, pd.DataFrame):
            df = y
        elif X is not None and isinstance(X, pd.DataFrame):
            df = X.copy()
            df[self.employment_col] = y
        else:
            raise ValueError("Must provide DataFrame with industry/region columns")
        
        # Compute regional totals (total employment per region)
        self.regional_totals_ = df.groupby(self.region_col)[self.employment_col].sum()
        
        # Compute national totals (total employment per industry)
        self.national_totals_ = df.groupby(self.industry_col)[self.employment_col].sum()
        
        # Compute national total (total employment across all regions/industries)
        self.national_total_ = df[self.employment_col].sum()
        
        # Compute LQ for each region-industry pair
        def compute_lq(row):
            regional_share = row[self.employment_col] / self.regional_totals_[row[self.region_col]]
            national_share = self.national_totals_[row[self.industry_col]] / self.national_total_
            return regional_share / national_share if national_share > 0 else 0
        
        df['location_quotient'] = df.apply(compute_lq, axis=1)
        
        # Store as pivot table (regions × industries)
        self.lq_matrix_ = df.pivot_table(
            index=self.region_col,
            columns=self.industry_col,
            values='location_quotient',
            fill_value=0
        )
        
        return self
    
    def get_location_quotients(self) -> pd.DataFrame:
        """
        Get LQ matrix (regions × industries).
        
        Returns
        -------
        lq_matrix : DataFrame
            Location quotients for all region-industry combinations
        """
        if self.lq_matrix_ is None:
            raise ValueError("Model must be fitted to get location quotients")
        
        return self.lq_matrix_
    
    def get_specialized_industries(self, region: str, threshold: float = 1.0) -> Dict[str, float]:
        """
        Get industries where region is specialized (LQ > threshold).
        
        Parameters
        ----------
        region : str
            Region identifier
        threshold : float, default=1.0
            LQ threshold for specialization
        
        Returns
        -------
        industries : dict
            Specialized industries with their LQ values
        """
        if self.lq_matrix_ is None:
            raise ValueError("Model must be fitted to get specialized industries")
        
        if region not in self.lq_matrix_.index:
            raise ValueError(f"Region '{region}' not found in data")
        
        region_lqs = self.lq_matrix_.loc[region]
        specialized = region_lqs[region_lqs > threshold]
        
        return specialized.to_dict()
    
    def forecast(self, steps: int = 1, X_future: Optional[np.ndarray] = None, **kwargs) -> ForecastResult:
        """
        Not directly applicable for LQ (descriptive measure).
        
        Raises
        ------
        NotImplementedError
            Location Quotient is a descriptive measure, not forecasting model
        """
        raise NotImplementedError("Location Quotient is a descriptive measure, not a forecasting model")
