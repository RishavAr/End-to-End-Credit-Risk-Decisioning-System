"""
Synthetic Credit Data Generator
Generates realistic LendingClub/FICO-style credit data with:
- Strong class imbalance
- Mixed numeric and categorical features
- Realistic correlations between features and default
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

class CreditDataGenerator:
    """Generate synthetic credit data mimicking LendingClub/FICO datasets"""
    
    def __init__(self, n_samples: int = 100000, default_rate: float = 0.15, 
                 random_state: int = 42):
        self.n_samples = n_samples
        self.default_rate = default_rate
        self.random_state = random_state
        np.random.seed(random_state)
        
    def generate(self) -> pd.DataFrame:
        """Generate complete synthetic credit dataset"""
        print(f"Generating {self.n_samples:,} synthetic credit records...")
        
        # Generate base features
        df = self._generate_numeric_features()
        df = self._add_categorical_features(df)
        
        # Generate target based on realistic risk factors
        df = self._generate_target(df)
        
        # Add some noise and missing values (realistic)
        df = self._add_realistic_noise(df)
        
        print(f"Generated dataset with {len(df):,} records")
        print(f"Default rate: {df['default'].mean():.2%}")
        
        return df
    
    def _generate_numeric_features(self) -> pd.DataFrame:
        """Generate correlated numeric features"""
        n = self.n_samples
        
        # Annual income (log-normal distribution)
        annual_income = np.exp(np.random.normal(10.8, 0.8, n))  # Median ~50k
        annual_income = np.clip(annual_income, 15000, 500000)
        
        # Loan amount (correlated with income)
        loan_amount = annual_income * np.random.uniform(0.1, 0.8, n)
        loan_amount = np.clip(loan_amount, 1000, 100000)
        
        # Debt-to-income ratio
        dti = np.random.beta(2, 5, n) * 60  # Right-skewed, mostly < 30%
        dti = np.clip(dti, 0, 60)
        
        # Credit utilization (correlated with risk)
        credit_util = np.random.beta(2, 3, n) * 100
        
        # Number of credit lines
        num_credit_lines = np.random.poisson(8, n)
        num_credit_lines = np.clip(num_credit_lines, 1, 30)
        
        # Total credit limit (correlated with income and credit lines)
        total_credit_limit = (annual_income * 0.5 + 
                             num_credit_lines * 2000 + 
                             np.random.normal(0, 5000, n))
        total_credit_limit = np.clip(total_credit_limit, 5000, 200000)
        
        # Revolving balance
        revolving_balance = total_credit_limit * (credit_util / 100)
        revolving_balance = np.clip(revolving_balance, 0, 150000)
        
        # Number of delinquencies (zero-inflated)
        has_delinquency = np.random.binomial(1, 0.25, n)
        num_delinquencies = has_delinquency * np.random.poisson(1.5, n)
        num_delinquencies = np.clip(num_delinquencies, 0, 10)
        
        # Months since last delinquency
        months_since_delinq = np.where(
            num_delinquencies > 0,
            np.random.exponential(24, n),
            np.nan  # No delinquency
        )
        
        # Inquiries in last 6 months
        num_inquiries = np.random.poisson(1.5, n)
        num_inquiries = np.clip(num_inquiries, 0, 15)
        
        # Employment duration (months)
        months_employed = np.random.exponential(60, n)
        months_employed = np.clip(months_employed, 0, 480)
        
        # Interest rate (based on risk profile)
        base_rate = 8 + dti * 0.15 + credit_util * 0.08 + num_delinquencies * 2
        interest_rate = base_rate + np.random.normal(0, 2, n)
        interest_rate = np.clip(interest_rate, 5.5, 30)
        
        # Loan term (36 or 60 months typically)
        loan_term = np.random.choice([36, 60], n, p=[0.6, 0.4])
        
        # Monthly installment
        r = interest_rate / 100 / 12
        installment = loan_amount * (r * (1 + r) ** loan_term) / ((1 + r) ** loan_term - 1)
        
        return pd.DataFrame({
            'annual_income': annual_income,
            'loan_amount': loan_amount,
            'debt_to_income': dti,
            'credit_utilization': credit_util,
            'num_credit_lines': num_credit_lines,
            'total_credit_limit': total_credit_limit,
            'revolving_balance': revolving_balance,
            'num_delinquencies': num_delinquencies,
            'months_since_delinquency': months_since_delinq,
            'num_inquiries_6m': num_inquiries,
            'months_employed': months_employed,
            'interest_rate': interest_rate,
            'loan_term': loan_term,
            'installment_amount': installment,
        })
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical features"""
        n = len(df)
        
        # Home ownership
        df['home_ownership'] = np.random.choice(
            ['RENT', 'MORTGAGE', 'OWN', 'OTHER'],
            n,
            p=[0.40, 0.45, 0.12, 0.03]
        )
        
        # Employment status
        df['employment_status'] = np.random.choice(
            ['Employed', 'Self-Employed', 'Unemployed', 'Retired', 'Student'],
            n,
            p=[0.70, 0.15, 0.05, 0.07, 0.03]
        )
        
        # Loan purpose
        df['loan_purpose'] = np.random.choice(
            ['debt_consolidation', 'credit_card', 'home_improvement', 
             'major_purchase', 'medical', 'car', 'small_business', 'other'],
            n,
            p=[0.40, 0.20, 0.10, 0.08, 0.07, 0.05, 0.05, 0.05]
        )
        
        # Verification status
        df['verification_status'] = np.random.choice(
            ['Verified', 'Source Verified', 'Not Verified'],
            n,
            p=[0.35, 0.35, 0.30]
        )
        
        # Grade (A-G, correlated with interest rate)
        grade_probs = self._get_grade_from_rate(df['interest_rate'])
        df['grade'] = [np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], p=p) 
                       for p in grade_probs]
        
        return df
    
    def _get_grade_from_rate(self, rates: np.ndarray) -> np.ndarray:
        """Get grade probabilities based on interest rate"""
        probs = []
        for rate in rates:
            if rate < 8:
                p = [0.6, 0.25, 0.1, 0.03, 0.01, 0.005, 0.005]
            elif rate < 12:
                p = [0.2, 0.4, 0.25, 0.1, 0.03, 0.01, 0.01]
            elif rate < 16:
                p = [0.05, 0.15, 0.35, 0.3, 0.1, 0.03, 0.02]
            elif rate < 20:
                p = [0.01, 0.05, 0.15, 0.35, 0.3, 0.1, 0.04]
            else:
                p = [0.005, 0.02, 0.05, 0.15, 0.3, 0.3, 0.175]
            probs.append(p)
        return np.array(probs)
    
    def _generate_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate default target with realistic correlations"""
        n = len(df)
        
        # Calculate risk score based on features
        risk_score = (
            # Higher DTI = higher risk
            0.03 * df['debt_to_income'] +
            # Higher credit utilization = higher risk
            0.025 * df['credit_utilization'] +
            # More delinquencies = higher risk
            0.4 * df['num_delinquencies'] +
            # More inquiries = higher risk
            0.1 * df['num_inquiries_6m'] +
            # Lower income relative to loan = higher risk
            0.5 * (df['loan_amount'] / df['annual_income']) +
            # Higher interest rate (proxy for lender's risk assessment) = higher risk
            0.1 * df['interest_rate'] +
            # Shorter employment = higher risk
            -0.005 * np.minimum(df['months_employed'], 120)
        )
        
        # Add categorical effects
        risk_score += np.where(df['home_ownership'] == 'RENT', 0.3, 0)
        risk_score += np.where(df['employment_status'] == 'Unemployed', 1.0, 0)
        risk_score += np.where(df['verification_status'] == 'Not Verified', 0.2, 0)
        
        # Grade effects
        grade_risk = {'A': -1.0, 'B': -0.5, 'C': 0, 'D': 0.5, 'E': 1.0, 'F': 1.5, 'G': 2.0}
        risk_score += df['grade'].map(grade_risk)
        
        # Convert to probability using logistic function
        # Calibrate to achieve target default rate
        risk_centered = risk_score - np.median(risk_score)
        prob_default = 1 / (1 + np.exp(-(risk_centered - np.log(self.default_rate / (1 - self.default_rate)))))
        
        # Add noise
        prob_default = np.clip(prob_default + np.random.normal(0, 0.05, n), 0.01, 0.99)
        
        # Generate binary default
        df['default'] = np.random.binomial(1, prob_default)
        
        # Adjust to match target rate more closely
        current_rate = df['default'].mean()
        if current_rate > self.default_rate * 1.1:
            # Randomly flip some 1s to 0s
            excess = int(n * (current_rate - self.default_rate))
            idx_ones = df[df['default'] == 1].index
            flip_idx = np.random.choice(idx_ones, min(excess, len(idx_ones)), replace=False)
            df.loc[flip_idx, 'default'] = 0
        elif current_rate < self.default_rate * 0.9:
            # Randomly flip some 0s to 1s
            deficit = int(n * (self.default_rate - current_rate))
            idx_zeros = df[df['default'] == 0].index
            flip_idx = np.random.choice(idx_zeros, min(deficit, len(idx_zeros)), replace=False)
            df.loc[flip_idx, 'default'] = 1
        
        return df
    
    def _add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic missing values and noise"""
        n = len(df)
        
        # Missing months_since_delinquency for those with no delinquencies (already done)
        
        # Missing employment duration for some
        missing_emp = np.random.choice(n, int(n * 0.02), replace=False)
        df.loc[missing_emp, 'months_employed'] = np.nan
        
        # Missing verification for some
        missing_verif = np.random.choice(n, int(n * 0.01), replace=False)
        df.loc[missing_verif, 'verification_status'] = np.nan
        
        return df
    
    def create_train_test_split(self, df: pd.DataFrame, 
                                test_size: float = 0.2,
                                val_size: float = 0.1) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits (time-based simulation)"""
        n = len(df)
        
        # Simulate time-based split by index
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))
        
        # Shuffle first to simulate random application order
        df_shuffled = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return {
            'train': df_shuffled.iloc[:train_end].copy(),
            'validation': df_shuffled.iloc[train_end:val_end].copy(),
            'test': df_shuffled.iloc[val_end:].copy(),
        }


def generate_production_simulation_data(base_df: pd.DataFrame, 
                                        drift_factor: float = 0.1,
                                        n_samples: int = 10000,
                                        random_state: int = 123) -> pd.DataFrame:
    """
    Generate simulated production data with slight drift for monitoring
    """
    np.random.seed(random_state)
    
    # Sample from original distribution with modifications
    prod_df = base_df.sample(n=n_samples, replace=True, random_state=random_state).copy()
    prod_df = prod_df.reset_index(drop=True)
    
    # Introduce drift in some features
    # Economic downturn simulation - slightly higher DTI, more delinquencies
    prod_df['debt_to_income'] *= (1 + drift_factor * np.random.uniform(0.5, 1.5, len(prod_df)))
    prod_df['debt_to_income'] = np.clip(prod_df['debt_to_income'], 0, 60)
    
    prod_df['num_inquiries_6m'] += np.random.poisson(drift_factor * 2, len(prod_df))
    prod_df['num_inquiries_6m'] = np.clip(prod_df['num_inquiries_6m'], 0, 15)
    
    # Slight shift in credit utilization
    prod_df['credit_utilization'] += np.random.normal(drift_factor * 5, 3, len(prod_df))
    prod_df['credit_utilization'] = np.clip(prod_df['credit_utilization'], 0, 100)
    
    return prod_df


if __name__ == "__main__":
    # Test data generation
    generator = CreditDataGenerator(n_samples=10000, default_rate=0.15)
    df = generator.generate()
    print("\nDataset shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)
    print("\nDefault distribution:")
    print(df['default'].value_counts(normalize=True))
    print("\nSample records:")
    print(df.head())
