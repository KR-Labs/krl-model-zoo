"""
Model Registry with SQLite Backend
===================================

Apache 2.0 License - Gate 1 Foundation
Author: KR Labs

Tracks model versions, parameters, and performance metrics.
Provides persistent storage for model metadata and experiment tracking.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np


class ModelRegistry:
    """
    SQLite-backed registry for tracking model experiments.
    
    Stores model metadata, parameters, performance metrics,
    and enables querying/comparison of model versions.
    
    Parameters
    ----------
    db_path : str or Path, default='models.db'
        Path to SQLite database file
    
    Attributes
    ----------
    db_path : Path
        Path to database
    conn : sqlite3.Connection
        Database connection
    
    Examples
    --------
    >>> registry = ModelRegistry('experiments.db')
    >>> registry.register_model(
    ...     name='KalmanFilter',
    ...     version='1.0.0',
    ...     parameters={'variance': 1.0},
    ...     metrics={'rmse': 0.5}
    ... )
    >>> models = registry.search(name='KalmanFilter')
    >>> best = registry.get_best_model('KalmanFilter', metric='rmse')
    """
    
    def __init__(self, db_path: str = 'models.db'):
        self.db_path = Path(db_path)
        self.conn = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Return rows as dicts
        
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                hash TEXT UNIQUE NOT NULL,
                parameters TEXT NOT NULL,
                metrics TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tags TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_name ON models(name)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_hash ON models(hash)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_created ON models(created_at)
        ''')
        
        self.conn.commit()
    
    def register_model(
        self,
        name: str,
        version: str,
        parameters: Dict[str, Any],
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Register a model in the registry.
        
        Parameters
        ----------
        name : str
            Model name
        version : str
            Model version
        parameters : dict
            Model hyperparameters
        metrics : dict, optional
            Performance metrics (e.g., {'rmse': 0.5})
        metadata : dict, optional
            Additional metadata
        tags : list of str, optional
            Tags for categorization
        
        Returns
        -------
        hash : str
            SHA256 hash of model configuration
        """
        # Compute hash
        hash_input = json.dumps({
            'name': name,
            'version': version,
            'parameters': parameters
        }, sort_keys=True)
        model_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        
        # Prepare data
        now = datetime.now().isoformat()
        params_json = json.dumps(parameters)
        metrics_json = json.dumps(metrics) if metrics else None
        metadata_json = json.dumps(metadata) if metadata else None
        tags_json = json.dumps(tags) if tags else None
        
        # Insert or update
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO models (name, version, hash, parameters, metrics, metadata, created_at, updated_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (name, version, model_hash, params_json, metrics_json, metadata_json, now, now, tags_json))
            self.conn.commit()
        except sqlite3.IntegrityError:
            # Hash exists, update instead
            cursor.execute('''
                UPDATE models
                SET version = ?, metrics = ?, metadata = ?, updated_at = ?, tags = ?
                WHERE hash = ?
            ''', (version, metrics_json, metadata_json, now, tags_json, model_hash))
            self.conn.commit()
        
        return model_hash
    
    def get_model(self, model_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve model by hash.
        
        Parameters
        ----------
        model_hash : str
            Model hash
        
        Returns
        -------
        model : dict or None
            Model record or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM models WHERE hash = ?', (model_hash,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        return self._row_to_dict(row)
    
    def search(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for models by criteria.
        
        Parameters
        ----------
        name : str, optional
            Filter by model name
        version : str, optional
            Filter by version
        tags : list of str, optional
            Filter by tags (ANY match)
        limit : int, default=100
            Maximum results
        
        Returns
        -------
        models : list of dict
            Matching model records
        """
        query = 'SELECT * FROM models WHERE 1=1'
        params = []
        
        if name:
            query += ' AND name = ?'
            params.append(name)
        if version:
            query += ' AND version = ?'
            params.append(version)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = [self._row_to_dict(row) for row in rows]
        
        # Filter by tags if specified
        if tags:
            results = [
                r for r in results
                if r.get('tags') and any(t in r['tags'] for t in tags)
            ]
        
        return results
    
    def get_best_model(
        self,
        name: str,
        metric: str,
        minimize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get best-performing model by metric.
        
        Parameters
        ----------
        name : str
            Model name
        metric : str
            Metric key (e.g., 'rmse', 'r2')
        minimize : bool, default=True
            If True, select minimum value; else maximum
        
        Returns
        -------
        model : dict or None
            Best model or None if no models found
        """
        models = self.search(name=name, limit=1000)
        models = [m for m in models if m.get('metrics') and metric in m['metrics']]
        
        if not models:
            return None
        
        if minimize:
            return min(models, key=lambda m: m['metrics'][metric])
        else:
            return max(models, key=lambda m: m['metrics'][metric])
    
    def delete_model(self, model_hash: str) -> bool:
        """
        Delete model by hash.
        
        Parameters
        ----------
        model_hash : str
            Model hash
        
        Returns
        -------
        success : bool
            True if deleted, False if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM models WHERE hash = ?', (model_hash,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    def list_models(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List all models.
        
        Parameters
        ----------
        limit : int, default=100
            Maximum results
        
        Returns
        -------
        models : list of dict
            All model records
        """
        return self.search(limit=limit)
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dict with JSON parsing."""
        result = dict(row)
        result['parameters'] = json.loads(result['parameters'])
        if result.get('metrics'):
            result['metrics'] = json.loads(result['metrics'])
        if result.get('metadata'):
            result['metadata'] = json.loads(result['metadata'])
        if result.get('tags'):
            result['tags'] = json.loads(result['tags'])
        return result
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
