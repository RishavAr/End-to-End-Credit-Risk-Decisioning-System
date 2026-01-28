"""
Model Training and Evaluation Module
Compares multiple models:
- Logistic Regression (baseline, explainability)
- Random Forest
- Gradient Boosting (sklearn)
- Histogram-based Gradient Boosting (faster, handles missing values)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
from sklearn.model_selection import (
    GridSearchCV, 
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import (
    roc_auc_score, 
    precision_recall_curve,
    average_precision_score,
    classification_report,
    confusion_matrix,
    brier_score_loss,
    log_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from typing import Dict, List, Tuple, Optional
import time
import pickle
import warnings
warnings.filterwarnings('ignore')


class CreditRiskModelTrainer:
    """Train and compare multiple credit risk models"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_model_configs(self, quick_mode: bool = False) -> Dict:
        """Get model configurations for grid search"""
        
        if quick_mode:
            # Reduced parameter grid for faster execution
            return {
                'logistic_regression': {
                    'model': LogisticRegression(
                        solver='saga',
                        max_iter=500,
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    'params': {
                        'C': [0.1, 1.0],
                        'penalty': ['l2'],
                    }
                },
                'random_forest': {
                    'model': RandomForestClassifier(
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    'params': {
                        'n_estimators': [100],
                        'max_depth': [10, 20],
                        'min_samples_split': [5],
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(
                        random_state=self.random_state
                    ),
                    'params': {
                        'n_estimators': [100],
                        'max_depth': [3, 5],
                        'learning_rate': [0.1],
                    }
                },
                'hist_gradient_boosting': {
                    'model': HistGradientBoostingClassifier(
                        random_state=self.random_state
                    ),
                    'params': {
                        'max_iter': [100],
                        'max_depth': [5, 10],
                        'learning_rate': [0.1],
                    }
                }
            }
        else:
            # Full parameter grid
            return {
                'logistic_regression': {
                    'model': LogisticRegression(
                        solver='saga',
                        max_iter=1000,
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    'params': {
                        'C': [0.01, 0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2'],
                    }
                },
                'random_forest': {
                    'model': RandomForestClassifier(
                        random_state=self.random_state,
                        n_jobs=-1
                    ),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [10, 20, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2],
                    }
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(
                        random_state=self.random_state
                    ),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1],
                        'subsample': [0.8, 1.0],
                    }
                },
                'hist_gradient_boosting': {
                    'model': HistGradientBoostingClassifier(
                        random_state=self.random_state
                    ),
                    'params': {
                        'max_iter': [100, 200],
                        'max_depth': [5, 10, None],
                        'learning_rate': [0.01, 0.1],
                    }
                }
            }
    
    def train_model(self, 
                    model_name: str,
                    model_config: Dict,
                    X_train: np.ndarray, 
                    y_train: np.ndarray,
                    cv: int = 5) -> Dict:
        """Train a single model with grid search"""
        
        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Setup cross-validation
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        # Grid search
        grid_search = GridSearchCV(
            model_config['model'],
            model_config['params'],
            cv=cv_splitter,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1,
            refit=True
        )
        
        grid_search.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Cross-validation predictions for calibration analysis
        cv_proba = cross_val_predict(
            best_model, X_train, y_train, 
            cv=cv_splitter, method='predict_proba'
        )[:, 1]
        
        result = {
            'model': best_model,
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'cv_scores_all': grid_search.cv_results_['mean_test_score'],
            'train_time': train_time,
            'cv_probabilities': cv_proba,
        }
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"CV AUC Score: {grid_search.best_score_:.4f}")
        print(f"Training time: {train_time:.2f}s")
        
        self.models[model_name] = best_model
        self.results[model_name] = result
        
        return result
    
    def train_all_models(self, 
                         X_train: np.ndarray, 
                         y_train: np.ndarray,
                         quick_mode: bool = False,
                         cv: int = 5) -> Dict:
        """Train all models and compare"""
        
        configs = self.get_model_configs(quick_mode)
        
        for model_name, config in configs.items():
            self.train_model(model_name, config, X_train, y_train, cv)
        
        # Determine best model
        best_name = max(self.results, key=lambda x: self.results[x]['cv_score'])
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\n{'='*60}")
        print(f"BEST MODEL: {best_name} (CV AUC: {self.results[best_name]['cv_score']:.4f})")
        print(f"{'='*60}")
        
        return self.results
    
    def evaluate_on_test(self, 
                         X_test: np.ndarray, 
                         y_test: np.ndarray) -> pd.DataFrame:
        """Evaluate all trained models on test set"""
        
        eval_results = []
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = self._calculate_metrics(y_test, y_pred, y_proba)
            metrics['model'] = model_name
            metrics['is_best'] = model_name == self.best_model_name
            
            eval_results.append(metrics)
        
        return pd.DataFrame(eval_results).set_index('model')
    
    def _calculate_metrics(self, 
                           y_true: np.ndarray, 
                           y_pred: np.ndarray, 
                           y_proba: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_proba),
            'auc_pr': average_precision_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba),
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        }
        
        # Gini coefficient (common in credit scoring)
        metrics['gini'] = 2 * metrics['auc_roc'] - 1
        
        # KS statistic
        metrics['ks_statistic'] = self._calculate_ks(y_true, y_proba)
        
        return metrics
    
    def _calculate_ks(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate Kolmogorov-Smirnov statistic"""
        # Sort by probability
        df = pd.DataFrame({'y': y_true, 'prob': y_proba})
        df = df.sort_values('prob', ascending=False).reset_index(drop=True)
        
        # Cumulative distributions
        df['cum_events'] = df['y'].cumsum() / df['y'].sum()
        df['cum_non_events'] = (1 - df['y']).cumsum() / (1 - df['y']).sum()
        
        # KS is max difference
        ks = (df['cum_events'] - df['cum_non_events']).abs().max()
        
        return ks
    
    def get_feature_importances(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importances from all models"""
        
        importance_dfs = []
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                imp = model.feature_importances_
            elif hasattr(model, 'coef_'):
                imp = np.abs(model.coef_[0])
            else:
                continue
            
            df = pd.DataFrame({
                'feature': feature_names,
                'importance': imp,
                'model': model_name
            })
            importance_dfs.append(df)
        
        if importance_dfs:
            combined = pd.concat(importance_dfs, ignore_index=True)
            # Normalize within each model
            combined['importance_normalized'] = combined.groupby('model')['importance'].transform(
                lambda x: x / x.sum()
            )
            return combined
        
        return pd.DataFrame()
    
    def calibrate_model(self, 
                        model_name: str,
                        X_train: np.ndarray, 
                        y_train: np.ndarray,
                        method: str = 'isotonic') -> CalibratedClassifierCV:
        """Calibrate a model's probability outputs"""
        
        base_model = self.models[model_name]
        
        calibrated = CalibratedClassifierCV(
            base_model, 
            method=method, 
            cv=5
        )
        calibrated.fit(X_train, y_train)
        
        return calibrated
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a saved model"""
        with open(filepath, 'rb') as f:
            self.models[model_name] = pickle.load(f)
        print(f"Model loaded from {filepath}")


class ModelExplainer:
    """Model explanation and interpretability"""
    
    def __init__(self, model, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
    def get_logistic_coefficients(self) -> pd.DataFrame:
        """Get coefficients for logistic regression (interpretability)"""
        if not hasattr(self.model, 'coef_'):
            raise ValueError("Model doesn't have coefficients (not logistic regression)")
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'odds_ratio': np.exp(self.model.coef_[0])
        })
        
        coef_df['abs_coefficient'] = np.abs(coef_df['coefficient'])
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
        
        # Add interpretation
        coef_df['direction'] = coef_df['coefficient'].apply(
            lambda x: 'Increases default risk' if x > 0 else 'Decreases default risk'
        )
        
        return coef_df
    
    def get_tree_feature_importance(self) -> pd.DataFrame:
        """Get feature importance for tree-based models"""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model doesn't have feature importances")
        
        imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        imp_df['cumulative_importance'] = imp_df['importance'].cumsum()
        imp_df['rank'] = range(1, len(imp_df) + 1)
        
        return imp_df
    
    def permutation_importance(self, 
                               X: np.ndarray, 
                               y: np.ndarray,
                               n_repeats: int = 10,
                               random_state: int = 42) -> pd.DataFrame:
        """Calculate permutation importance"""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        imp_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        return imp_df


if __name__ == "__main__":
    # Test model training
    from data_generator import CreditDataGenerator
    from feature_engineering import CreditFeatureEngineer
    
    # Generate data
    generator = CreditDataGenerator(n_samples=10000)
    df = generator.generate()
    splits = generator.create_train_test_split(df)
    
    # Feature engineering
    fe = CreditFeatureEngineer()
    X_train, y_train = fe.fit_transform(splits['train'])
    X_test, y_test = fe.transform(splits['test'])
    
    # Train models (quick mode for testing)
    trainer = CreditRiskModelTrainer()
    results = trainer.train_all_models(X_train, y_train, quick_mode=True, cv=3)
    
    # Evaluate
    eval_df = trainer.evaluate_on_test(X_test, y_test)
    print("\nTest Set Evaluation:")
    print(eval_df)
