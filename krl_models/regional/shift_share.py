# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024 KR-Labs

"""
Shift-Share nalysis Model for Regional conomic ecomposition.

Shift-Share analysis decomposes regional employment/economic changes into:
. National Growth ffect: What growth would occur if region matched national rate
2. Industry Mix ffect: Impact of region's industrial composition
3. Regional Competitive ffect: Region's competitive advantage/disadvantage

ormula:
Total hange = National ffect + Industry Mix ffect + Competitive ffect

Where:
- National ffect = Regional ase mployment × National Growth Rate
- Industry Mix = Regional ase × (Industry Growth - National Growth)
- Competitive = Regional ase × (Regional Industry Growth - National Industry Growth)

Use ases:
- xplaining why Virginia arts employment grew/declined
- Identifying whether growth is due to national trends, good industry mix, or regional competitiveness
- Policy evaluation: re regional initiatives working?
- enchmarking: How does region compare to national industry trends?
"""

from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
import logging

from krl_core.base_model import ModelMeta
from krl_core.results import ForecastResult

logger = logging.getLogger(__name__)


class ShiftShareModel:
    """
    Perform Shift-Share decomposition analysis.
    
    ecomposes regional employment changes into national, industry mix,
    and competitive shift components.
    
    Parameters (via ModelInputSchema.params):
    - base_year_col: str - olumn for base year values
    - end_year_col: str - olumn for end year values
    - sector_col: str - olumn for sector identifiers
    - region_prefix: str - Prefix for regional columns (1e10.g., 'va_')
    - national_prefix: str - Prefix for national columns (1e10.g., 'us_')
    
    xample:
        >>> schema = ModelInputSchema(
        ...     params={
        ...         'base_year_col': '2',
        ...         'end_year_col': '223',
        ...         'sector_col': 'sector',
        ...         'region_prefix': 'va_',
        ...         'national_prefix': 'us_'
        ...     }
        ... )
        >>> model = ShiftShareModel(schema, meta)
        >>> result = model.fit(employment_data)
        >>> print(result.payload['decomposition'])
    """
    
    def __init__(
        self,
        params: Dict[str, Any],
        meta: Optional[ModelMeta] = None
    ):
        """Initialize Shift-Share model."""
        self.params = params
        self.meta = meta or ModelMeta(name="ShiftShare", version="1.0.0", author="KR Labs")
        self._fitted = False
        
        # Extract parameters
        self._base_year_col = self.params.get('base_year_col')
        self._end_year_col = self.params.get('end_year_col')
        self._sector_col = self.params.get('sector_col')
        self._region_prefix = self.params.get('region_prefix', 'region_')
        self._national_prefix = self.params.get('national_prefix', 'national_')
        
        # Validate required parameters
        if not self._base_year_col:
            raise ValueError("Parameter 'base_year_col' is required")
        if not self._end_year_col:
            raise ValueError("Parameter 'end_year_col' is required")
        if not self._sector_col:
            raise ValueError("Parameter 'sector_col' is required")
        
        # Results storage
        self.decomposition_: Optional[Dict[str, Any]] = None
        self.sector_effects_: Optional[pd.DataFrame] = None
    
    def fit(self, data: pd.DataFrame) -> ForecastResult:
        """
        Perform shift-share decomposition.
        
        rgs:
            data: DataFrame with base/end year employment by sector for region and nation
            
        Returns:
            ForecastResult with decomposition analysis
        """
        if data.empty:
            raise ValueError("Input data cannot be empty")
        
        # onstruct column names
        reg_base_col = f"{self._region_prefix}{self._base_year_col}"
        reg_end_col = f"{self._region_prefix}{self._end_year_col}"
        nat_base_col = f"{self._national_prefix}{self._base_year_col}"
        nat_end_col = f"{self._national_prefix}{self._end_year_col}"
        
        # Validate columns exist
        required_cols = [reg_base_col, reg_end_col, nat_base_col, nat_end_col, self._sector_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Performing shift-share analysis for {len(data)} sectors")
        
        # Calculate totals
        reg_base_total = data[reg_base_col].sum()
        reg_end_total = data[reg_end_col].sum()
        nat_base_total = data[nat_base_col].sum()
        nat_end_total = data[nat_end_col].sum()
        
        if reg_base_total == 0 or nat_base_total == 0:
            raise ValueError("ase year totals cannot be zero")
        
        # Calculate growth rates
        national_growth_rate = (nat_end_total - nat_base_total) / nat_base_total
        regional_change = reg_end_total - reg_base_total
        
        # Initialize effect accumulators
        national_effect = 0.0
        industry_mix_effect = 0.0
        competitive_effect = 0.0
        
        # Calculate effects by sector
        sector_results = []
        
        for _, row in data.iterrows():
            sector = row[self._sector_col]
            reg_base = row[reg_base_col]
            reg_end = row[reg_end_col]
            nat_base = row[nat_base_col]
            nat_end = row[nat_end_col]
            
            # National effect: if region grew at national rate
            nat_effect_sector = reg_base * national_growth_rate
            national_effect += nat_effect_sector
            
            # Industry-specific growth rate
            if nat_base > 0:
                industry_growth_rate = (nat_end - nat_base) / nat_base
            else:
                industry_growth_rate = 0.0
            
            # Industry mix effect: differential between industry and national growth
            mix_effect_sector = reg_base * (industry_growth_rate - national_growth_rate)
            industry_mix_effect += mix_effect_sector
            
            # Regional growth rate for this industry
            if reg_base > 0:
                regional_industry_growth = (reg_end - reg_base) / reg_base
            else:
                regional_industry_growth = 0.0
            
            # Competitive effect: regional performance vs national industry
            comp_effect_sector = reg_base * (regional_industry_growth - industry_growth_rate)
            competitive_effect += comp_effect_sector
            
            sector_results.append({
                'sector': sector,
                'regional_base': float(reg_base),
                'regional_end': float(reg_end),
                'regional_change': float(reg_end - reg_base),
                'national_effect': float(nat_effect_sector),
                'industry_mix_effect': float(mix_effect_sector),
                'competitive_effect': float(comp_effect_sector),
                'total_explained': float(nat_effect_sector + mix_effect_sector + comp_effect_sector),
            })
        
        self.sector_effects_ = pd.DataFrame(sector_results)
        
        # Store decomposition
        self.decomposition_ = {
            'regional_change': float(regional_change),
            'national_effect': float(national_effect),
            'industry_mix_effect': float(industry_mix_effect),
            'competitive_effect': float(competitive_effect),
            'total_explained': float(national_effect + industry_mix_effect + competitive_effect),
            'residual': float(regional_change - (national_effect + industry_mix_effect + competitive_effect)),
        }
        
        # Calculate shares (as percentages of total change)
        if abs(regional_change) > 0.001:
            self.decomposition_['national_share'] = (national_effect / regional_change) * 100
            self.decomposition_['industry_mix_share'] = (industry_mix_effect / regional_change) * 100
            self.decomposition_['competitive_share'] = (competitive_effect / regional_change) * 100
        else:
            self.decomposition_['national_share'] = 0.0
            self.decomposition_['industry_mix_share'] = 0.0
            self.decomposition_['competitive_share'] = 0.0
        
        # Identify top contributors
        top_competitive = self.sector_effects_.nlargest(5, 'competitive_effect')
        bottom_competitive = self.sector_effects_.nsmallest(5, 'competitive_effect')
        
        logger.info(f"Regional change: {regional_change:,.f}")
        logger.info(f"National effect: {national_effect:,.f} ({self.decomposition_['national_share']:.f}%)")
        logger.info(f"Industry mix: {industry_mix_effect:,.f} ({self.decomposition_['industry_mix_share']:.f}%)")
        logger.info(f"Competitive: {competitive_effect:,.f} ({self.decomposition_['competitive_share']:.f}%)")
        
        # reate result
        result = ForecastResult(
            payload={
                'decomposition': self.decomposition_,
                'sector_effects': self.sector_effects_.to_dict('records'),
                'top_competitive_sectors': top_competitive['sector'].tolist(),
                'bottom_competitive_sectors': bottom_competitive['sector'].tolist(),
                'n_sectors': len(data),
                'regional_base_total': float(reg_base_total),
                'regional_end_total': float(reg_end_total),
                'national_growth_rate': float(national_growth_rate * 100),  # as percentage
            },
            metadata={
                'model_name': self.meta.name,
                'model_version': self.meta.version,
                'author': self.meta.author,
                'calculated_at': pd.Timestamp.now().isoformat(),
                'base_year': self._base_year_col,
                'end_year': self._end_year_col,
            },
            forecast_index=['national_effect', 'industry_mix_effect', 'competitive_effect'],
            forecast_values=[
                self.decomposition_['national_effect'],
                self.decomposition_['industry_mix_effect'],
                self.decomposition_['competitive_effect']
            ],
            ci_lower=[],
            ci_upper=[]
        )
        
        self._fitted = True
        return result
    
    def predict(self, data: pd.DataFrame) -> ForecastResult:
        """
        Perform shift-share analysis on new data (same as fit).
        
        rgs:
            data: DataFrame with same structure as training data
            
        Returns:
            ForecastResult with decomposition analysis
        """
        # or shift-share, predict is the same as fit
        return self.fit(data)
    
    def get_sector_decomposition(self, sector: str) -> Dict[str, float]:
        """
        Get decomposition for a specific sector.
        
        rgs:
            sector: Sector name
            
        Returns:
            Dict with national, mix, and competitive effects for the sector
        """
        if not self._fitted or self.sector_effects_ is None:
            raise RuntimeError("Model must be fitted first")
        
        sector_row = self.sector_effects_[self.sector_effects_['sector'] == sector]
        if sector_row.empty:
            raise ValueError(f"Sector '{sector}' not found in results")
        
        return sector_row.iloc[0].to_dict()
