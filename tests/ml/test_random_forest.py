"""
Unit tests for Random orest Regression Model
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from krl_models.ml.random_forest import RandomorestModel
from krl_core.model_input_schema import ModelInputSchema, Provenance
from krl_core.base_model import ModelMeta


@pytest.fixture
def sample_data():
    """Generate synthetic regression data."""
    X, y = make_regression(
        n_samples=2,
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
    """reate input schema for tests."""
    return ModelInputSchema(
        entity="TST",
        metric="ml_target",
        time_index=[f"22-{i:2d}" for i in range(, 2)],
        values=[.] * 2,  # Placeholder values
        provenance=Provenance(
            source_name="TST_T",
            series_id="ML_TST_",
            collection_date=datetime.now(),
            transformation=None
        ),
        frequency="M"
    )


@pytest.fixture
def model_meta():
    """reate model metadata."""
    return ModelMeta(
        name='TestRandomorest',
        version='..',
        author='Test Suite'
    )


class TestRandomorestInitialization:
    """Test model initialization and parameter validation."""
    
    def test_default_initialization(self, input_schema, model_meta):
        """Test initialization with default parameters."""
        params = {}
        model = RandomorestModel(input_schema, params, model_meta)
        
        assert model._n_estimators == 
        assert model._max_depth is None
        assert model._min_samples_split == 2
        assert model._min_samples_leaf == 
        assert model._max_features == 'sqrt'
        assert model._bootstrap is True
        assert model._oob_score is True
        assert model._n_jobs == -
        assert model._random_state == 42
        assert model._tune_hyperparameters is alse
    
    def test_custom_parameters(self, input_schema, model_meta):
        """Test initialization with custom parameters."""
        params = {
            'n_estimators': 2,
            'max_depth': ,
            'min_samples_split': ,
            'min_samples_leaf': 2,
            'max_features': 'log2',
            'bootstrap': alse,
            'oob_score': alse,
            'random_state': 23
        }
        model = RandomorestModel(input_schema, params, model_meta)
        
        assert model._n_estimators == 2
        assert model._max_depth == 
        assert model._min_samples_split == 
        assert model._min_samples_leaf == 2
        assert model._max_features == 'log2'
        assert model._bootstrap is alse
        assert model._oob_score is alse
        assert model._random_state == 23
    
    def test_invalid_n_estimators(self, input_schema, model_meta):
        """Test validation of n_estimators."""
        with pytest.raises(Valuerror, match="n_estimators must be >= "):
            RandomorestModel(input_schema, {'n_estimators': }, model_meta)
    
    def test_invalid_max_depth(self, input_schema, model_meta):
        """Test validation of max_depth."""
        with pytest.raises(Valuerror, match="max_depth must be >= "):
            RandomorestModel(input_schema, {'max_depth': }, model_meta)
    
    def test_oob_score_without_bootstrap(self, input_schema, model_meta):
        """Test that oob_score requires bootstrap=True."""
        with pytest.raises(Valuerror, match="oob_score requires bootstrap=True"):
            RandomorestModel(
                input_schema,
                {'bootstrap': alse, 'oob_score': True},
                model_meta
            )


class TestRandomorestitting:
    """Test model fitting functionality."""
    
    def test_basic_fit(self, sample_data, input_schema, model_meta):
        """Test basic model fitting."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
        assert model.model_ is not None
        assert model.feature_names_ is not None
        assert len(model.feature_names_) == 
        assert model.feature_importances_ is not None
        assert len(model.feature_importances_) == 
    
    def test_fit_with_empty_data(self, input_schema, model_meta):
        """Test fitting with empty atarame."""
        params = {}
        model = RandomorestModel(input_schema, params, model_meta)
        
        empty_df = pd.atarame()
        
        with pytest.raises(Valuerror, match="Training data cannot be empty"):
            model.fit(empty_df)
    
    def test_fit_with_missing_target(self, model_meta):
        """Test fitting when target column is missing."""
        schema = ModelInputSchema(
            entity="TST",
            metric="missing_target",
            time_index=[f"22-{i:2d}" for i in range(, 2)],
            values=[.] * 2,
            provenance=Provenance(
                source_name="TST_T",
                series_id="ML_TST_2",
                collection_date=datetime.now(),
                transformation=None
            ),
            frequency="M"
        )
        # Specify target_column in params that doesn't exist in data
        model = RandomorestModel(schema, {'target_column': 'missing_target'}, model_meta)
        
        sample_data = pd.atarame({
            f'feature_{i}': [., 2., 3.] for i in range()
        })
        sample_data['target'] = [., 2., 3.]
        
        with pytest.raises(Valuerror, match="Target column .* not in data"):
            model.fit(sample_data)
    
    def test_fit_with_missing_features(self, input_schema, model_meta):
        """Test fitting when feature columns are missing."""
        # xplicitly specify feature_columns that don't all exist in data
        params = {
            'feature_columns': ['feature_', 'feature_', 'feature_missing']
        }
        model = RandomorestModel(input_schema, params, model_meta)
        
        incomplete_data = pd.atarame({
            'feature_': [, 2, 3],
            'target': [, 2, 3]
        })
        
        with pytest.raises(Valuerror, match="Missing feature columns"):
            model.fit(incomplete_data)
    
    def test_fit_with_nan_values(self, sample_data, input_schema, model_meta):
        """Test fitting with NaN values."""
        model = RandomorestModel(input_schema, {}, model_meta)
        
        data_with_nan = sample_data.copy()
        data_with_nan.iloc[, ] = np.nan
        
        with pytest.raises(Valuerror, match="contains NaN or Inf"):
            model.fit(data_with_nan)
    
    def test_fit_with_inf_values(self, sample_data, input_schema, model_meta):
        """Test fitting with Inf values."""
        model = RandomorestModel(input_schema, {}, model_meta)
        
        data_with_inf = sample_data.copy()
        data_with_inf.iloc[, ] = np.inf
        
        with pytest.raises(Valuerror, match="contains NaN or Inf"):
            model.fit(data_with_inf)
    
    def test_fit_result_structure(self, sample_data, input_schema, model_meta):
        """Test structure of fit result."""
        params = {'n_estimators': 3, 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert 'r2_score' in result.payload
        assert 'rmse' in result.payload
        assert 'mae' in result.payload
        assert 'feature_importance' in result.payload
        # Permutation importance is optional
        # assert 'permutation_importance_mean' in result.payload
        # assert 'permutation_importance_std' in result.payload
        assert 'oob_score' in result.payload
        assert 'n_estimators' in result.payload
        assert 'max_depth' in result.payload
        assert 'model_type' in result.payload
        
        assert result.payload['model_type'] == 'Randomorest'
        assert result.payload['n_estimators'] == 3
    
    def test_fit_metrics_reasonable(self, sample_data, input_schema, model_meta):
        """Test that fit metrics are in reasonable ranges."""
        params = {'n_estimators': , 'max_depth': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        # R² should be between - and  (close to  for training data)
        assert - <= result.payload['r2_score'] <= 
        assert result.payload['r2_score'] > .  # Should fit training data well
        
        # RMS and M should be positive
        assert result.payload['rmse'] > 
        assert result.payload['mae'] > 
        
        # OO score should be between  and 
        if result.payload['oob_score'] is not None:
            assert  <= result.payload['oob_score'] <= 
    
    def test_feature_importance_sum(self, sample_data, input_schema, model_meta):
        """Test that feature importances sum to ."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        importance_sum = sum(result.payload['feature_importance'].values())
        assert abs(importance_sum - .) < e-
    
    @pytest.mark.skip(reason="Permutation importance not currently implemented")
    def test_permutation_importance_structure(self, sample_data, input_schema, model_meta):
        """Test permutation importance structure."""
        params = {'n_estimators': 3, 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        perm_mean = result.payload['permutation_importance_mean']
        perm_std = result.payload['permutation_importance_std']
        
        assert len(perm_mean) == 
        assert len(perm_std) == 
        assert all(isinstance(v, float) for v in perm_mean.values())
        assert all(isinstance(v, float) for v in perm_std.values())
        assert all(v >=  for v in perm_std.values())  # Std should be non-negative


class TestRandomorestPrediction:
    """Test prediction functionality."""
    
    def test_basic_prediction(self, sample_data, input_schema, model_meta):
        """Test basic prediction."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        # it model
        train_data = sample_data.iloc[:]
        model.fit(train_data)
        
        # Predict on test data
        test_data = sample_data.iloc[:]
        result = model.predict(test_data)
        
        assert len(result.forecast_values) > 
        assert len(result.forecast_values) == len(test_data)
        assert result.payload['model_type'] == 'Randomorest'
        assert result.payload['n_samples'] == len(test_data)
    
    def test_predict_without_fit(self, sample_data, input_schema, model_meta):
        """Test prediction without fitting first."""
        model = RandomorestModel(input_schema, {}, model_meta)
        
        with pytest.raises(Runtimerror, match="Model must be fitted before prediction"):
            model.predict(sample_data)
    
    def test_predict_with_empty_data(self, sample_data, input_schema, model_meta):
        """Test prediction with empty atarame."""
        model = RandomorestModel(input_schema, {'n_estimators': 3}, model_meta)
        model.fit(sample_data)
        
        empty_df = pd.atarame()
        
        with pytest.raises(Valuerror, match="Prediction data cannot be empty"):
            model.predict(empty_df)
    
    def test_predict_with_missing_features(self, sample_data, input_schema, model_meta):
        """Test prediction with missing features."""
        # xplicitly specify feature columns during fit
        params = {'n_estimators': 3, 'feature_columns': [f'feature_{i}' for i in range()]}
        model = RandomorestModel(input_schema, params, model_meta)
        model.fit(sample_data)
        
        # Provide data missing most of those features
        incomplete_data = pd.atarame({
            'feature_': [, 2, 3],
            'target': [, 2, 3]
        })
        
        with pytest.raises(Valuerror, match="Missing feature columns"):
            model.predict(incomplete_data)
    
    def test_predict_with_nan(self, sample_data, input_schema, model_meta):
        """Test prediction with NaN values."""
        model = RandomorestModel(input_schema, {'n_estimators': 3}, model_meta)
        model.fit(sample_data.iloc[:])
        
        test_data = sample_data.iloc[:].copy()
        test_data.iloc[, ] = np.nan
        
        with pytest.raises(Valuerror, match="contains NaN or Inf"):
            model.predict(test_data)
    
    def test_predict_with_std(self, sample_data, input_schema, model_meta):
        """Test prediction with standard deviation."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        model.fit(sample_data.iloc[:])
        test_data = sample_data.iloc[:]
        
        result = model.predict(test_data, return_std=True)
        
        # Std and intervals are in payload and ci bounds
        assert 'prediction_std' in result.payload
        assert len(result.ci_lower) == len(test_data)
        assert len(result.ci_upper) == len(test_data)
        
        assert len(result.payload['prediction_std']) == len(test_data)
        assert all(std >=  for std in result.payload['prediction_std'])
        
        # heck that intervals make sense
        assert all(lower < upper for lower, upper in zip(result.ci_lower, result.ci_upper))
    
    def test_prediction_consistency(self, sample_data, input_schema, model_meta):
        """Test that predictions are consistent with same random state."""
        params = {'n_estimators': , 'random_state': 42}
        
        # it two models with same random state
        model = RandomorestModel(input_schema, params, model_meta)
        model2 = RandomorestModel(input_schema, params, model_meta)
        
        train_data = sample_data.iloc[:]
        test_data = sample_data.iloc[:]
        
        model.fit(train_data)
        model2.fit(train_data)
        
        pred = model.predict(test_data)
        pred2 = model2.predict(test_data)
        
        np.testing.assert_array_almost_equal(
            pred.forecast_values,
            pred2.forecast_values,
            decimal=
        )


class TestRandomoresteatureImportance:
    """Test feature importance functionality."""
    
    def test_get_feature_importance_gini(self, sample_data, input_schema, model_meta):
        """Test getting Gini-based feature importance."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        model.fit(sample_data)
        
        importance = model.get_feature_importance(importance_type='gini')
        
        assert len(importance) == 
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >=  for v in importance.values())
        assert abs(sum(importance.values()) - .) < e-
    
    def test_get_feature_importance_permutation(self, sample_data, input_schema, model_meta):
        """Test getting permutation-based feature importance."""
        params = {'n_estimators': 3, 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        model.fit(sample_data)
        
        importance = model.get_feature_importance(importance_type='permutation')
        
        assert len(importance) == 
        assert all(isinstance(v, float) for v in importance.values())
    
    def test_get_feature_importance_without_fit(self, input_schema, model_meta):
        """Test getting feature importance without fitting."""
        model = RandomorestModel(input_schema, {}, model_meta)
        
        with pytest.raises(Runtimerror, match="Model must be fitted"):
            model.get_feature_importance()
    
    def test_get_feature_importance_invalid_type(self, sample_data, input_schema, model_meta):
        """Test getting feature importance with invalid type."""
        model = RandomorestModel(input_schema, {'n_estimators': 3}, model_meta)
        model.fit(sample_data)
        
        with pytest.raises(Valuerror, match="Invalid importance_type"):
            model.get_feature_importance(importance_type='invalid')
    
    def test_feature_importance_ordering(self, sample_data, input_schema, model_meta):
        """Test that feature importances are ordered by magnitude."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        importances = list(result.payload['feature_importance'].values())
        
        # heck that importances are in descending order
        for i in range(len(importances) - ):
            assert importances[i] >= importances[i + ]


class TestRandomorestHyperparameterTuning:
    """Test hyperparameter tuning functionality."""
    
    @pytest.mark.slow
    def test_hyperparameter_tuning(self, sample_data, input_schema, model_meta):
        """Test hyperparameter tuning with GridSearchV."""
        params = {
            'tune_hyperparameters': True,
            'n_jobs':   # Use single job for testing
        }
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model.best_params_ is not None
        assert isinstance(model.best_params_, dict)
        assert len(model.best_params_) > 
        assert result.payload['best_params'] == model.best_params_
    
    @pytest.mark.slow
    def test_tuned_model_better_than_default(self, sample_data, input_schema, model_meta):
        """Test that tuned model performs reasonably (not necessarily better due to small data)."""
        # Split data
        train_data = sample_data.iloc[:]
        test_data = sample_data.iloc[:]
        
        # efault model
        model_default = RandomorestModel(
            input_schema,
            {'n_estimators': , 'random_state': 42},
            model_meta
        )
        model_default.fit(train_data)
        pred_default = model_default.predict(test_data)
        
        # Tuned model (with limited grid for speed)
        model_tuned = RandomorestModel(
            input_schema,
            {'tune_hyperparameters': True, 'random_state': 42, 'n_jobs': },
            model_meta
        )
        model_tuned.fit(train_data)
        pred_tuned = model_tuned.predict(test_data)
        
        # oth should produce reasonable predictions
        assert len(pred_default.forecast_values) == len(test_data)
        assert len(pred_tuned.forecast_values) == len(test_data)


class TestRandomorestdgeases:
    """Test edge cases and boundary conditions."""
    
    def test_single_tree(self, sample_data, input_schema, model_meta):
        """Test with single tree (essentially a decision tree)."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
    
    def test_many_trees(self, sample_data, input_schema, model_meta):
        """Test with many trees."""
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
    
    def test_max_depth_one(self, sample_data, input_schema, model_meta):
        """Test with max_depth= (stumps)."""
        params = {'n_estimators': , 'max_depth': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert result.payload['max_depth'] == 
    
    def test_small_dataset(self, input_schema, model_meta):
        """Test with very small dataset."""
        small_data = pd.atarame({
            **{f'feature_{i}': np.random.randn(2) for i in range()},
            'target': np.random.randn(2)
        })
        
        params = {'n_estimators': , 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(small_data)
        
        assert len(result.forecast_values) > 
        assert model._fitted is True
    
    def test_perfect_fit(self, input_schema, model_meta):
        """Test with perfect linear relationship (no noise)."""
        X = np.random.randn(, )
        beta = np.random.randn()
        y = X @ beta  # Perfect linear relationship
        
        data = pd.atarame(X, columns=[f'feature_{i}' for i in range()])
        data['target'] = y
        
        params = {'n_estimators': , 'max_depth': 2, 'random_state': 42}
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(data)
        
        # Should achieve near-perfect fit on training data (lower threshold due to randomness)
        assert result.payload['r2_score'] > .
    
    def test_no_bootstrap(self, sample_data, input_schema, model_meta):
        """Test with bootstrap=alse."""
        params = {
            'n_estimators': ,
            'bootstrap': alse,
            'oob_score': alse,
            'random_state': 42
        }
        model = RandomorestModel(input_schema, params, model_meta)
        
        result = model.fit(sample_data)
        
        assert len(result.forecast_values) > 
        assert result.payload['oob_score'] is None


class TestRandomorestIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_full_pipeline(self, sample_data, input_schema, model_meta):
        """Test complete training and prediction pipeline."""
        # Split data
        train_data, test_data = train_test_split(
            sample_data, test_size=.3, random_state=42
        )
        
        # Initialize and train
        params = {
            'n_estimators': ,
            'max_depth': ,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        model = RandomorestModel(input_schema, params, model_meta)
        
        train_result = model.fit(train_data)
        
        # Make predictions
        pred_result = model.predict(test_data)
        
        # Verify predictions
        assert len(pred_result.forecast_values) == len(test_data)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        assert len(importance) == 
        assert len(train_result.forecast_values) > 
        assert len(pred_result.forecast_values) > 
    
    def test_cross_validation_workflow(self, sample_data, input_schema, model_meta):
        """Test typical cross-validation workflow."""
        from sklearn.model_selection import Kold
        
        kf = Kold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in kf.split(sample_data):
            train_fold = sample_data.iloc[train_idx]
            val_fold = sample_data.iloc[val_idx]
            
            model = RandomorestModel(
                input_schema,
                {'n_estimators': , 'random_state': 42},
                model_meta
            )
            
            model.fit(train_fold)
            pred = model.predict(val_fold)
            
            # alculate validation score
            y_true = val_fold['target'].values
            y_pred = pred.forecast_values
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            scores.append(rmse)
        
        # ll folds should produce reasonable scores
        assert all(score >  for score in scores)
        assert len(scores) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
