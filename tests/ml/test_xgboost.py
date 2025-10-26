# ----------------------------------------------------------------------
# © 2025 KR-Labs. All rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LLC,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for XGBoost Regression Model
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from krl_models.ml.xgboost_model import XGoostModel
from krl_core.model_input_schema import ModelInputSchema, Provenance
from krl_core.base_model import ModelMeta


@pytest.fixture
def sample_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=3,
        n_features=,
        n_informative=,
        
        noise=.,
        random_state=42
    )
    
    df = pd.atarame(X, columns=[f'feature_{i}' for i in range()])
    df['target'] = y
    
    return df


@pytest.fixture
def input_schema():
    """Create input schema for tests."""
    return ModelInputSchema(
        entity="TST",
        metric="ml_target",
        time_index=[f"22-{i:2d}" for i in range(, 3)],
        values=[.] * 3,
        provenance=Provenance(
            source_name="TST_T",
            Useries_id="ML_TST_XG_",
            collection_date=datetime.now(),
            transformation=None
        ),
        frequency="M"
    )


@pytest.fixture
def model_meta():
    """Create model metadata."""
    return ModelMeta(
        name='TestXGoost',
        version='..',
        author='Test Suite'
    )


class TestXGoostInitialization:
    """Test model initialization and parameter validation."""
    
    def test_default_initialization(self, input_schema, model_meta):
        """Test initialization with default parameters."""
        params = {}
        model = XGoostModel(input_schema, params, model_meta)
        
        assert model._n_estimators == 
        assert model._max_depth == 
        assert model._learning_rate == .
        assert model._subsample == .
        assert model._colsample_bytree == .
        assert model._gamma == 
        assert model._reg_alpha == 
        assert model._reg_lambda == 
        assert model._min_child_weight == 
        assert model._early_stopping_rounds == 
        assert model._eval_metric == 'rmse'
        assert model._n_jobs == -
        assert model._random_state == 42
    
    def test_custom_parameters(self, input_schema, model_meta):
        """Test initialization with custom parameters."""
        params = {
            'n_estimators': 2,
            'max_depth': ,
            'learning_rate': .,
            'subsample': .,
            'colsample_bytree': .,
            'gamma': .,
            'reg_alpha': .,
            'reg_lambda': 2.,
            'min_child_weight': 3,
            'early_stopping_rounds': 2,
            'eval_metric': 'mae',
            'random_state': 23
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        assert model._n_estimators == 2
        assert model._max_depth == 
        assert model._learning_rate == .
        assert model._subsample == .
        assert model._colsample_bytree == .
        assert model._gamma == .
        assert model._reg_alpha == .
        assert model._reg_lambda == 2.
        assert model._min_child_weight == 3
        assert model._early_stopping_rounds == 2
        assert model._eval_metric == 'mae'
        assert model._random_state == 23
    
    def test_invalid_n_estimators(self, input_schema, model_meta):
        """Test validation of n_estimators."""
        with pytest.raises(Valuerror, match="n_estimators must be >= "):
            XGoostModel(input_schema, {'n_estimators': }, model_meta)
    
    def test_invalid_max_depth(self, input_schema, model_meta):
        """Test validation of max_depth."""
        with pytest.raises(Valuerror, match="max_depth must be >= "):
            XGoostModel(input_schema, {'max_depth': }, model_meta)
    
    def test_invalid_learning_rate(self, input_schema, model_meta):
        """Test validation of learning_rate."""
        with pytest.raises(Valuerror, match="learning_rate must be in"):
            XGoostModel(input_schema, {'learning_rate': }, model_meta)
        
        with pytest.raises(Valuerror, match="learning_rate must be in"):
            XGoostModel(input_schema, {'learning_rate': .}, model_meta)
    
    def test_invalid_subsample(self, input_schema, model_meta):
        """Test validation of subsample."""
        with pytest.raises(Valuerror, match="subsample must be in"):
            XGoostModel(input_schema, {'subsample': }, model_meta)
        
        with pytest.raises(Valuerror, match="subsample must be in"):
            XGoostModel(input_schema, {'subsample': .}, model_meta)
    
    def test_invalid_colsample_bytree(self, input_schema, model_meta):
        """Test validation of colsample_bytree."""
        with pytest.raises(Valuerror, match="colsample_bytree must be in"):
            XGoostModel(input_schema, {'colsample_bytree': }, model_meta)


class TestXGoostitting:
    """Test model fitting functionality."""
    
    def test_basic_fit(self, sample_data, input_schema, model_meta):
        """Test basic model fitting."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
        assert model.model_ is not None
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == 
        assert model.feature_importances_ is not None
        assert model.best_iteration_ is not None
    
    def test_fit_with_empty_data(self, input_schema, model_meta):
        """Test fitting with empty atarame."""
        model = XGoostModel(input_schema, {}, model_meta)
        empty_df = pd.atarame()
        
        with pytest.raises(Valuerror, match="Training data cannot be empty"):
            model.fit(empty_df)
    
    def test_fit_with_missing_target(self, model_meta):
        """Test fitting when target column is missing."""
        schema = ModelInputSchema(
            entity="TST",
            metric="missing_target",
            time_index=[f"22-{i:2d}" for i in range(, 3)],
            values=[.] * 3,
            provenance=Provenance(
                source_name="TST_T",
                Useries_id="ML_TST_XG_2",
                collection_date=datetime.now(),
                transformation=None
            ),
            frequency="M"
        )
        # Specify target_column in params that doesn't exist in data
        model = XGoostModel(schema, {'target_column': 'missing_target'}, model_meta)
        
        sample_data = pd.atarame({
            f'feature_{i}': [., 2., 3.] for i in range()
        })
        sample_data['target'] = [., 2., 3.]
        
        with pytest.raises(Valuerror, match="Target column .* not in data"):
            model.fit(sample_data)
    
    def test_fit_with_nan_in_target(self, sample_data, input_schema, model_meta):
        """Test fitting with NaN in target."""
        model = XGoostModel(input_schema, {}, model_meta)
        
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[, -] = np.nan  # Last column is target
        
        with pytest.raises(Valuerror, match="contains NaN or Inf"):
            model.fit(data_with_nan)
    
    def test_fit_result_structure(self, sample_data, input_schema, model_meta):
        """Test structure of fit result."""
        params = {'n_estimators': 3, 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert 'r2_score' in result.payload
        assert 'rmse' in result.payload
        assert 'mae' in result.payload
        assert 'feature_importance' in result.payload
        assert 'best_iteration' in result.payload
        assert 'n_estimators' in result.payload
        assert 'max_depth' in result.payload
        assert 'learning_rate' in result.payload
        assert 'model_type' in result.payload
        
        assert result.payload['model_type'] == 'XGBoost'
        assert result.payload['n_estimators'] == 3
        assert result.payload['best_iteration'] == 3
    
    def test_fit_with_validation_set(self, sample_data, input_schema, model_meta):
        """Test fitting with validation set for early stopping."""
        train_data = sample_data.iloc[:2]
        val_data = sample_data.iloc[2:]
        
        params = {
            'n_estimators': ,
            'early_stopping_rounds': ,
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(train_data, eval_set=[(val_data, 'validation')])
        
        assert len(result.forecast_values) > 
        assert 'val_rmse' in result.payload
        assert 'training_history' in result.payload
        assert model.best_iteration_ is not None
        assert model.best_iteration_ <= 
        assert model.evals_result_ is not None
    
    def test_fit_metrics_reasonable(self, sample_data, input_schema, model_meta):
        """Test that fit metrics are in reasonable ranges."""
        params = {'n_estimators': , 'max_depth': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        # R² should be reasonable
        assert - <= result.payload['r2_score'] <= 
        assert result.payload['r2_score'] > .
        
        # RMS and M should be positive
        assert result.payload['rmse'] > 
        assert result.payload['mae'] > 
    
    def test_feature_importance_structure(self, sample_data, input_schema, model_meta):
        """Test feature importance structure."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        importance = result.payload['feature_importance']
        
        assert len(importance) == 
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >=  for v in importance.values())
        # Should sum to ~ (normalized)
        assert abs(sum(importance.values()) - .) < .


class TestXGoostPrediction:
    """Test prediction functionality."""
    
    def test_basic_prediction(self, sample_data, input_schema, model_meta):
        """Test basic prediction."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        train_data = sample_data.iloc[:2]
        test_data = sample_data.iloc[2:]
        
        model.fit(train_data)
        result = model.predict(test_data)
        
        assert len(result.forecast_values) > 
        assert len(result.forecast_values) == len(test_data)
        assert result.payload['model_type'] == 'XGBoost'
        assert result.payload['n_samples'] == len(test_data)
        assert result.payload['best_iteration'] == model.best_iteration_
    
    def test_predict_without_fit(self, sample_data, input_schema, model_meta):
        """Test prediction without fitting first."""
        model = XGoostModel(input_schema, {}, model_meta)
        
        with pytest.raises(Runtimerror, match="Model must be fitted before prediction"):
            model.predict(sample_data)
    
    def test_predict_with_empty_data(self, sample_data, input_schema, model_meta):
        """Test prediction with empty atarame."""
        model = XGoostModel(input_schema, {'n_estimators': 3}, model_meta)
        model.fit(sample_data)
        
        empty_df = pd.atarame()
        
        with pytest.raises(Valuerror, match="Prediction data cannot be empty"):
            model.predict(empty_df)
    
    def test_predict_with_missing_features(self, sample_data, input_schema, model_meta):
        """Test prediction with missing features."""
        model = XGoostModel(input_schema, {'n_estimators': 3}, model_meta)
        model.fit(sample_data)
        
        incomplete_data = pd.atarame({
            'feature_': [, 2, 3]
        })
        
        with pytest.raises(Valuerror, match="Missing feature columns"):
            model.predict(incomplete_data)
    
    def test_prediction_uses_best_iteration(self, sample_data, input_schema, model_meta):
        """Test that prediction uses best_iteration."""
        train_data = sample_data.iloc[:2]
        val_data = sample_data.iloc[2:2]
        test_data = sample_data.iloc[2:]
        
        params = {
            'n_estimators': ,
            'early_stopping_rounds': ,
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        model.fit(train_data, eval_set=[(val_data, 'validation')])
        result = model.predict(test_data)
        
        # Should use best_iteration, not all  trees
        assert model.best_iteration_ <= 
        assert result.payload['best_iteration'] == model.best_iteration_
    
    def test_prediction_consistency(self, sample_data, input_schema, model_meta):
        """Test that predictions are consistent with same random state."""
        params = {'n_estimators': , 'random_state': 42}
        
        train_data = sample_data.iloc[:2]
        test_data = sample_data.iloc[2:]
        
        model = XGoostModel(input_schema, params, model_meta)
        model2 = XGoostModel(input_schema, params, model_meta)
        
        model.fit(train_data)
        model2.fit(train_data)
        
        pred = model.predict(test_data)
        pred2 = model2.predict(test_data)
        
        np.testing.assert_array_almost_equal(
            pred.forecast_values,
            pred2.forecast_values,
            decimal=
        )


class TestXGoosteatureImportance:
    """Test feature importance functionality."""
    
    def test_get_feature_importance_gain(self, sample_data, input_schema, model_meta):
        """Test getting gain-based feature importance."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        model.fit(sample_data)
        
        importance = model.get_feature_importance(importance_type='gain')
        
        assert len(importance) == 
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >=  for v in importance.values())
    
    def test_get_feature_importance_weight(self, sample_data, input_schema, model_meta):
        """Test getting weight-based feature importance."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        model.fit(sample_data)
        
        importance = model.get_feature_importance(importance_type='weight')
        
        assert len(importance) == 
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >=  for v in importance.values())
    
    def test_get_feature_importance_cover(self, sample_data, input_schema, model_meta):
        """Test getting cover-based feature importance."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        model.fit(sample_data)
        
        importance = model.get_feature_importance(importance_type='cover')
        
        assert len(importance) == 
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >=  for v in importance.values())
    
    def test_get_feature_importance_without_fit(self, input_schema, model_meta):
        """Test getting feature importance without fitting."""
        model = XGoostModel(input_schema, {}, model_meta)
        
        with pytest.raises(Runtimerror, match="Model must be fitted"):
            model.get_feature_importance()
    
    def test_get_feature_importance_invalid_type(self, sample_data, input_schema, model_meta):
        """Test getting feature importance with invalid type."""
        model = XGoostModel(input_schema, {'n_estimators': 3}, model_meta)
        model.fit(sample_data)
        
        with pytest.raises(Valuerror, match="Invalid importance_type"):
            model.get_feature_importance(importance_type='invalid')
    
    def test_feature_importance_ordering(self, sample_data, input_schema, model_meta):
        """Test that feature importances are ordered by magnitude."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        importances = list(result.payload['feature_importance'].values())
        
        # heck that importances are in descending order
        for i in range(len(importances) - ):
            assert importances[i] >= importances[i + ]


class TestXGoostRegularization:
    """Test regularization features."""
    
    def test_l_regularization(self, sample_data, input_schema, model_meta):
        """Test L (Lasso) regularization."""
        params = {
            'n_estimators': ,
            'reg_alpha': .,  # L
            'reg_lambda': .,  # No L2
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
    
    def test_l2_regularization(self, sample_data, input_schema, model_meta):
        """Test L2 (Ridge) regularization."""
        params = {
            'n_estimators': ,
            'reg_alpha': .,  # No L
            'reg_lambda': 2.,  # L2
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
    
    def test_elastic_net_regularization(self, sample_data, input_schema, model_meta):
        """Test combined L + L2 regularization (lastic Net)."""
        params = {
            'n_estimators': ,
            'reg_alpha': .,  # L
            'reg_lambda': .,  # L2
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True


class TestXGoostdgeases:
    """Test edge cases and boundary conditions."""
    
    def test_single_tree(self, sample_data, input_schema, model_meta):
        """Test with single tree."""
        params = {'n_estimators': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model.best_iteration_ == 
    
    def test_very_shallow_trees(self, sample_data, input_schema, model_meta):
        """Test with max_depth=5 (stumps)."""
        params = {'n_estimators': , 'max_depth': , 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert result.payload['max_depth'] == 
    
    def test_high_learning_rate(self, sample_data, input_schema, model_meta):
        """Test with high learning rate."""
        params = {'n_estimators': 2, 'learning_rate': ., 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert result.payload['learning_rate'] == .
    
    def test_low_learning_rate(self, sample_data, input_schema, model_meta):
        """Test with low learning rate."""
        params = {'n_estimators': , 'learning_rate': ., 'random_state': 42}
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert result.payload['learning_rate'] == .
    
    def test_subsample_small(self, sample_data, input_schema, model_meta):
        """Test with small subsample ratio."""
        params = {
            'n_estimators': ,
            'subsample': .,
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
    
    def test_colsample_small(self, sample_data, input_schema, model_meta):
        """Test with small column sample ratio."""
        params = {
            'n_estimators': ,
            'colsample_bytree': .,
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 


class TestXGoostIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_full_pipeline_with_early_stopping(self, sample_data, input_schema, model_meta):
        """Test complete pipeline with early stopping."""
        # Split data
        train_data = sample_data.iloc[:2]
        val_data = sample_data.iloc[2:2]
        test_data = sample_data.iloc[2:]
        
        # Train with validation
        params = {
            'n_estimators': 2,
            'max_depth': ,
            'learning_rate': .,
            'early_stopping_rounds': 2,
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        train_result = model.fit(train_data, eval_set=[(val_data, 'validation')])
        
        # Predict on test
        pred_result = model.predict(test_data)
        
        # Verify
        assert len(train_result.forecast_values) > 
        assert len(pred_result.forecast_values) > 
        assert len(pred_result.forecast_values) == len(test_data)
        assert model.best_iteration_ < 2  # Should stop early
        assert 'val_rmse' in train_result.payload
        
        # Get feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 
    
    def test_multiple_validation_sets(self, sample_data, input_schema, model_meta):
        """Test with multiple validation sets."""
        train_data = sample_data.iloc[:]
        val_data = sample_data.iloc[:22]
        val2_data = sample_data.iloc[22:]
        
        params = {
            'n_estimators': ,
            'early_stopping_rounds': ,
            'random_state': 42
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(
            train_data,
            eval_set=[(val_data, 'val'), (val2_data, 'val2')]
        )
        
        assert len(result.forecast_values) > 
        assert 'training_history' in result.payload
        assert model.evals_result_ is not None
    
    def test_comparison_with_low_vs_high_complexity(self, sample_data, input_schema, model_meta):
        """ompare low vs high complexity models."""
        train_data = sample_data.iloc[:2]
        test_data = sample_data.iloc[2:]
        
        # Low complexity
        model_simple = XGoostModel(
            input_schema,
            {'n_estimators': 2, 'max_depth': 2, 'random_state': 42},
            model_meta
        )
        model_simple.fit(train_data)
        pred_simple = model_simple.predict(test_data)
        
        # High complexity
        model_complex = XGoostModel(
            input_schema,
            {'n_estimators': 2, 'max_depth': , 'random_state': 42},
            model_meta
        )
        model_complex.fit(train_data)
        pred_complex = model_complex.predict(test_data)
        
        # oth should produce valid predictions
        assert len(pred_simple.forecast_values) == len(test_data)
        assert len(pred_complex.forecast_values) == len(test_data)


class TestXGoostHyperparameterTuning:
    """Test hyperparameter tuning functionality."""
    
    @pytest.mark.slow
    def test_hyperparameter_tuning(self, sample_data, input_schema, model_meta):
        """Test hyperparameter tuning with GridSearchV."""
        params = {
            'tune_hyperparameters': True,
            'n_jobs':   # Single job for testing
        }
        model = XGoostModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model.best_params_ is not None
        assert isinstance(model.best_params_, dict)
        assert len(model.best_params_) > 


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
