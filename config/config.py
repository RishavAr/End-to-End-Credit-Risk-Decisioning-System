"""
Configuration settings for Credit Risk & Decisioning System
"""

# Data Generation Settings
DATA_CONFIG = {
    'n_samples': 100000,  # Number of samples to generate (can scale to millions)
    'default_rate': 0.15,  # ~15% default rate (realistic class imbalance)
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.1,
}

# Feature Engineering Settings
FEATURE_CONFIG = {
    # Numeric features to generate
    'numeric_features': [
        'annual_income',
        'debt_to_income',
        'credit_utilization',
        'num_credit_lines',
        'num_delinquencies',
        'months_since_delinquency',
        'total_credit_limit',
        'revolving_balance',
        'installment_amount',
        'num_inquiries_6m',
        'months_employed',
        'loan_amount',
        'interest_rate',
        'loan_term',
    ],
    # Categorical features
    'categorical_features': [
        'home_ownership',
        'employment_status',
        'loan_purpose',
        'verification_status',
        'grade',
    ],
}

# Model Training Settings
MODEL_CONFIG = {
    'logistic_regression': {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': 'saga',
        'max_iter': 1000,
    },
    'random_forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
    },
    'hist_gradient_boosting': {
        'max_iter': [100, 200],
        'max_depth': [5, 10, None],
        'learning_rate': [0.01, 0.1],
    },
}

# Scoring Settings (FICO-style)
SCORING_CONFIG = {
    'score_min': 300,
    'score_max': 850,
    'score_bands': {
        'Excellent': (750, 850),
        'Good': (700, 749),
        'Fair': (650, 699),
        'Poor': (550, 649),
        'Very Poor': (300, 549),
    },
}

# Monitoring Settings
MONITORING_CONFIG = {
    'psi_threshold': 0.25,  # Population Stability Index threshold
    'csi_threshold': 0.25,  # Characteristic Stability Index threshold
    'ks_threshold': 0.05,   # KS test p-value threshold for drift
}

# Output Settings
OUTPUT_CONFIG = {
    'figures_dpi': 150,
    'figure_size': (12, 8),
}
