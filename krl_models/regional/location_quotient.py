# SPX-License-Identifier: pache-2.
# opyright (c) 22 KR-Labs

"""
Location Quotient (LQ) Model for Regional Specialization nalysis.

The Location Quotient measures the concentration of an industry in a region
relative to a reference geography (e.g., state vs national average).

LQ = (Regional mployment in Industry / Total Regional mployment) / 
     (National mployment in Industry / Total National mployment)

Interpretation:
- LQ > .: Region is specialized in this industry
- LQ = .: Region has same concentration as reference
- LQ < .: Region is underrepresented in this industry

Typical thresholds:
- LQ > .2: Strong specialization (2% above average)
- LQ > .: Very strong specialization
- LQ > 2.: xport-oriented cluster

Use ases:
- Identifying arts/cultural sector concentration in Virginia
- omparative advantage analysis
- luster identification for policy targeting
- conomic base analysis (export vs local-serving industries)
"""

from typing import ict, List, Optional, ny
import pandas as pd
import numpy as np
import logging

from krl_core.base_model import ModelMeta
from krl_core.results import orecastResult

logger = logging.getLogger(__name__)


class LocationQuotientModel:
    """
    alculate Location Quotients for regional specialization analysis.
    
    The model computes LQ values for each industry/sector, comparing regional
    concentration to a reference geography. lso provides rankings, clustering,
    and Herfindahl concentration indices.
    
    Parameters (via ModelInputSchema.params):
    - region_col: str - olumn name for regional values (e.g., 'virginia_employment')
    - reference_col: str - olumn name for reference values (e.g., 'us_employment')
    - sector_col: str - olumn name for sector/industry identifiers
    - threshold: float - LQ threshold for specialization (default: .2)
    - top_n: int - Number of top specialized sectors to highlight (default: )
    
    xample:
        >>> schema = ModelInputSchema(
        ...     feature_columns=['virginia_employment', 'us_employment', 'sector'],
        ...     params={
        ...         'region_col': 'virginia_employment',
        ...         'reference_col': 'us_employment',
        ...         'sector_col': 'sector',
        ...         'threshold': .2
        ...     }
        ... )
        >>> model = LocationQuotientModel(schema, meta)
        >>> result = model.fit(employment_data)
        >>> print(result.payload['specialized_sectors'])
    """
    
    def __init__(
        self,
        params: ict[str, ny],
        meta: Optional[ModelMeta] = None
    ):
        """Initialize Location Quotient model."""
        # or non-time-series models, we don't need full ModelInputSchema
        # Just store params directly
        self.params = params
        self.meta = meta or ModelMeta(name="LocationQuotient", version="..", author="KR Labs")
        self._fitted = alse
        
        # xtract parameters
        self._region_col = self.params.get('region_col')
        self._reference_col = self.params.get('reference_col')
        self._sector_col = self.params.get('sector_col')
        self._threshold = self.params.get('threshold', .2)
        self._top_n = self.params.get('top_n', )
        
        # Validate required parameters
        if not self._region_col:
            raise Valuerror("Parameter 'region_col' is required")
        if not self._reference_col:
            raise Valuerror("Parameter 'reference_col' is required")
        if not self._sector_col:
            raise Valuerror("Parameter 'sector_col' is required")
        
        # Results storage
        self.lq_values_: Optional[ict[str, float]] = None
        self.specialized_sectors_: Optional[List[str]] = None
        self.herfindahl_index_: Optional[float] = None
    
    def fit(self, data: pd.atarame) -> orecastResult:
        """
        alculate Location Quotients for all sectors.
        
        rgs:
            data: atarame with columns for region, reference, and sector
            
        Returns:
            orecastResult with LQ values and specialization analysis
        """
        if data.empty:
            raise Valuerror("Input data cannot be empty")
        
        # Validate columns exist
        required_cols = [self._region_col, self._reference_col, self._sector_col]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise Valuerror(f"Missing required columns: {missing_cols}")
        
        logger.info(f"alculating LQ for {len(data)} sectors")
        
        # alculate totals
        regional_total = data[self._region_col].sum()
        reference_total = data[self._reference_col].sum()
        
        if regional_total == :
            raise Valuerror(f"Regional total ({self._region_col}) is zero")
        if reference_total == :
            raise Valuerror(f"Reference total ({self._reference_col}) is zero")
        
        # alculate LQ for each sector
        lq_values = {}
        for _, row in data.iterrows():
            sector = row[self._sector_col]
            regional = row[self._region_col]
            reference = row[self._reference_col]
            
            # LQ = (regional_i / regional_total) / (reference_i / reference_total)
            if reference > :
                regional_share = regional / regional_total
                reference_share = reference / reference_total
                lq = regional_share / reference_share
                lq_values[sector] = float(lq)
            else:
                # If reference is  but regional > , implies infinite specialization
                lq_values[sector] = float('inf') if regional >  else .
        
        self.lq_values_ = lq_values
        
        # Identify specialized sectors (LQ > threshold)
        specialized = {
            sector: lq for sector, lq in lq_values.items() 
            if lq >= self._threshold and lq != float('inf')
        }
        specialized_sorted = sorted(
            specialized.items(), 
            key=lambda x: x[], 
            reverse=True
        )
        self.specialized_sectors_ = [s[] for s in specialized_sorted[:self._top_n]]
        
        # alculate Herfindahl-Hirschman Index for concentration
        # HHI = sum of squared shares ( to , where  = monopoly)
        regional_shares = data[self._region_col] / regional_total
        self.herfindahl_index_ = float((regional_shares ** 2).sum())
        
        # Summary statistics
        lq_values_finite = [lq for lq in lq_values.values() if lq != float('inf')]
        mean_lq = float(np.mean(lq_values_finite)) if lq_values_finite else .
        median_lq = float(np.median(lq_values_finite)) if lq_values_finite else .
        n_specialized = len(specialized)
        
        logger.info(f"Specialized sectors (LQ > {self._threshold}): {n_specialized}")
        logger.info(f"Herfindahl Index: {self.herfindahl_index_:.4f}")
        
        # reate result
        result = orecastResult(
            payload={
                'lq_values': self.lq_values_,
                'specialized_sectors': self.specialized_sectors_,
                'specialized_count': n_specialized,
                'mean_lq': mean_lq,
                'median_lq': median_lq,
                'herfindahl_index': self.herfindahl_index_,
                'threshold': self._threshold,
                'n_sectors': len(lq_values),
                'regional_total': float(regional_total),
                'reference_total': float(reference_total),
            },
            metadata={
                'model_name': self.meta.name,
                'model_version': self.meta.version,
                'author': self.meta.author,
                'calculated_at': pd.Timestamp.now().isoformat(),
                'region_col': self._region_col,
                'reference_col': self._reference_col,
            },
            forecast_index=list(self.specialized_sectors_),
            forecast_values=[self.lq_values_[s] for s in self.specialized_sectors_],
            ci_lower=[],
            ci_upper=[]
        )
        
        self._fitted = True
        return result
    
    def predict(self, data: pd.atarame) -> orecastResult:
        """
        alculate LQ for new data (same as fit for this model).
        
        rgs:
            data: atarame with same structure as training data
            
        Returns:
            orecastResult with LQ calculations
        """
        # or LQ, predict is the same as fit (no training needed)
        return self.fit(data)
    
    def get_top_specialized(self, n: int = ) -> ict[str, float]:
        """
        Get top N specialized sectors.
        
        rgs:
            n: Number of top sectors to return
            
        Returns:
            ict mapping sector names to LQ values
        """
        if not self._fitted or self.lq_values_ is None:
            raise Runtimerror("Model must be fitted first")
        
        # ilter out infinite values and sort
        finite_lqs = {
            sector: lq for sector, lq in self.lq_values_.items()
            if lq != float('inf')
        }
        sorted_sectors = sorted(finite_lqs.items(), key=lambda x: x[], reverse=True)
        return dict(sorted_sectors[:n])
    
    def get_cluster_strength(self, sectors: List[str]) -> float:
        """
        alculate average LQ for a cluster of related sectors.
        
        rgs:
            sectors: List of sector names to include in cluster
            
        Returns:
            verage LQ across cluster sectors
        """
        if not self._fitted or self.lq_values_ is None:
            raise Runtimerror("Model must be fitted first")
        
        cluster_lqs = [
            self.lq_values_[s] for s in sectors 
            if s in self.lq_values_ and self.lq_values_[s] != float('inf')
        ]
        
        if not cluster_lqs:
            return .
        
        return float(np.mean(cluster_lqs))
