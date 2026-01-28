"""
Feature Engineering Pipeline for Credit Risk Model
- Handles missing values
- Encodes categorical features
- Creates derived features
- Scales numeric features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CreditFeatureEngineer:
    """Feature engineering pipeline for credit risk modeling"""
    
    def __init__(self):
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.feature_names = None
        self.fitted = False
        
    def identify_features(self, df: pd.DataFrame, target_col: str = 'default') -> Tuple[List[str], List[str]]:
        """Identify numeric and categorical features"""
        exclude_cols = [target_col]
        
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.numeric_features = [c for c in self.numeric_features if c not in exclude_cols]
        
        self.categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.categorical_features = [c for c in self.categorical_features if c not in exclude_cols]
        
        return self.numeric_features, self.categorical_features
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific derived features"""
        df = df.copy()
        
        # Payment-to-income ratio
        df['payment_to_income'] = df['installment_amount'] * 12 / df['annual_income']
        
        # Loan-to-income ratio
        df['loan_to_income'] = df['loan_amount'] / df['annual_income']
        
        # Credit utilization buckets (WoE-style)
        df['high_utilization'] = (df['credit_utilization'] > 70).astype(int)
        df['very_high_utilization'] = (df['credit_utilization'] > 90).astype(int)
        
        # Delinquency indicator
        df['has_delinquency'] = (df['num_delinquencies'] > 0).astype(int)
        df['multiple_delinquencies'] = (df['num_delinquencies'] > 2).astype(int)
        
        # Recent delinquency (within 12 months)
        df['recent_delinquency'] = (
            (df['has_delinquency'] == 1) & 
            (df['months_since_delinquency'].fillna(999) < 12)
        ).astype(int)
        
        # Credit line density (credit per line)
        df['credit_per_line'] = df['total_credit_limit'] / (df['num_credit_lines'] + 1)
        
        # Inquiry intensity
        df['high_inquiry'] = (df['num_inquiries_6m'] > 3).astype(int)
        
        # Employment stability
        df['stable_employment'] = (df['months_employed'].fillna(0) > 24).astype(int)
        df['long_employment'] = (df['months_employed'].fillna(0) > 60).astype(int)
        
        # DTI risk buckets
        df['high_dti'] = (df['debt_to_income'] > 30).astype(int)
        df['very_high_dti'] = (df['debt_to_income'] > 40).astype(int)
        
        # Interest rate risk indicator
        df['high_rate'] = (df['interest_rate'] > 15).astype(int)
        
        # Interaction features
        df['dti_x_utilization'] = df['debt_to_income'] * df['credit_utilization'] / 100
        df['income_x_term'] = df['annual_income'] / df['loan_term']
        
        return df
    
    def build_preprocessor(self, df: pd.DataFrame) -> ColumnTransformer:
        """Build sklearn preprocessing pipeline"""
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])
        
        # Combine
        self.preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, self.numeric_features),
            ('categorical', categorical_pipeline, self.categorical_features)
        ])
        
        return self.preprocessor
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'default') -> Tuple[np.ndarray, np.ndarray]:
        """Fit preprocessor and transform data"""
        # Create derived features
        df_engineered = self.create_derived_features(df)
        
        # Identify features
        self.identify_features(df_engineered, target_col)
        
        # Build and fit preprocessor
        self.build_preprocessor(df_engineered)
        
        X = df_engineered[self.numeric_features + self.categorical_features]
        y = df_engineered[target_col].values
        
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names
        self._extract_feature_names()
        
        self.fitted = True
        
        return X_transformed, y
    
    def transform(self, df: pd.DataFrame, target_col: str = 'default') -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform new data using fitted preprocessor"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted first")
        
        # Create derived features
        df_engineered = self.create_derived_features(df)
        
        X = df_engineered[self.numeric_features + self.categorical_features]
        
        X_transformed = self.preprocessor.transform(X)
        
        if target_col in df_engineered.columns:
            y = df_engineered[target_col].values
            return X_transformed, y
        
        return X_transformed, None
    
    def _extract_feature_names(self):
        """Extract feature names after fitting"""
        feature_names = []
        
        # Numeric features (unchanged names)
        feature_names.extend(self.numeric_features)
        
        # Categorical features (one-hot encoded names)
        cat_encoder = self.preprocessor.named_transformers_['categorical'].named_steps['encoder']
        cat_features = cat_encoder.get_feature_names_out(self.categorical_features)
        feature_names.extend(cat_features.tolist())
        
        self.feature_names = feature_names
        
    def get_feature_names(self) -> List[str]:
        """Return feature names"""
        return self.feature_names if self.feature_names else []
    
    def get_feature_importance_df(self, importances: np.ndarray) -> pd.DataFrame:
        """Create feature importance DataFrame"""
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)


class WOEEncoder:
    """
    Weight of Evidence encoder for categorical variables
    Used in traditional credit scoring for better interpretability
    """
    
    def __init__(self, min_samples: int = 100, regularization: float = 0.5):
        self.min_samples = min_samples
        self.regularization = regularization
        self.woe_maps = {}
        self.iv_scores = {}
        
    def fit(self, df: pd.DataFrame, categorical_cols: List[str], target_col: str = 'default'):
        """Fit WOE encoding"""
        for col in categorical_cols:
            woe_map, iv = self._calculate_woe(df, col, target_col)
            self.woe_maps[col] = woe_map
            self.iv_scores[col] = iv
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical columns to WOE values"""
        df_transformed = df.copy()
        
        for col, woe_map in self.woe_maps.items():
            if col in df_transformed.columns:
                df_transformed[f'{col}_woe'] = df_transformed[col].map(woe_map).fillna(0)
        
        return df_transformed
    
    def _calculate_woe(self, df: pd.DataFrame, col: str, target_col: str) -> Tuple[Dict, float]:
        """Calculate WOE and IV for a single column"""
        # Total goods and bads
        total_events = df[target_col].sum()
        total_non_events = len(df) - total_events
        
        # Calculate by category
        grouped = df.groupby(col)[target_col].agg(['sum', 'count'])
        grouped.columns = ['events', 'total']
        grouped['non_events'] = grouped['total'] - grouped['events']
        
        # Distribution of events and non-events
        grouped['dist_events'] = grouped['events'] / total_events
        grouped['dist_non_events'] = grouped['non_events'] / total_non_events
        
        # Apply regularization to avoid log(0)
        grouped['dist_events'] = grouped['dist_events'].clip(lower=self.regularization / total_events)
        grouped['dist_non_events'] = grouped['dist_non_events'].clip(lower=self.regularization / total_non_events)
        
        # WOE calculation
        grouped['woe'] = np.log(grouped['dist_non_events'] / grouped['dist_events'])
        
        # IV calculation
        grouped['iv'] = (grouped['dist_non_events'] - grouped['dist_events']) * grouped['woe']
        iv = grouped['iv'].sum()
        
        woe_map = grouped['woe'].to_dict()
        
        return woe_map, iv
    
    def get_iv_summary(self) -> pd.DataFrame:
        """Get Information Value summary for variable selection"""
        iv_df = pd.DataFrame({
            'feature': list(self.iv_scores.keys()),
            'iv': list(self.iv_scores.values())
        }).sort_values('iv', ascending=False)
        
        # Add predictive power interpretation
        def interpret_iv(iv):
            if iv < 0.02:
                return 'Not useful'
            elif iv < 0.1:
                return 'Weak'
            elif iv < 0.3:
                return 'Medium'
            elif iv < 0.5:
                return 'Strong'
            else:
                return 'Suspicious (check overfitting)'
        
        iv_df['interpretation'] = iv_df['iv'].apply(interpret_iv)
        
        return iv_df


if __name__ == "__main__":
    # Test feature engineering
    from data_generator import CreditDataGenerator
    
    generator = CreditDataGenerator(n_samples=10000)
    df = generator.generate()
    
    # Test feature engineer
    fe = CreditFeatureEngineer()
    X, y = fe.fit_transform(df)
    
    print(f"Original features: {len(df.columns)}")
    print(f"Engineered features: {X.shape[1]}")
    print(f"Feature names: {fe.get_feature_names()[:10]}...")
    
    # Test WOE encoder
    woe = WOEEncoder()
    cat_cols = ['home_ownership', 'employment_status', 'loan_purpose', 'grade']
    woe.fit(df, cat_cols)
    
    print("\nInformation Value Summary:")
    print(woe.get_iv_summary())
