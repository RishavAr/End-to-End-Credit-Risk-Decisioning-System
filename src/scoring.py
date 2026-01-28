"""
FICO-Style Credit Scoring System
- Converts model probabilities to credit scores (300-850 scale)
- Creates score bands/categories
- Implements scorecard logic
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ScoreBand:
    """Represents a score band/category"""
    name: str
    min_score: int
    max_score: int
    description: str
    typical_apr_range: str


class FICOStyleScorer:
    """
    Convert model probabilities to FICO-style credit scores (300-850)
    
    The scoring follows the FICO philosophy:
    - Higher score = lower risk (inverse of probability of default)
    - Scores follow a roughly normal distribution
    - Score points have consistent meaning across the range
    """
    
    def __init__(self, 
                 score_min: int = 300, 
                 score_max: int = 850,
                 pdo: int = 20,  # Points to Double the Odds
                 base_score: int = 600,
                 base_odds: float = 1.0):  # 1:1 odds at base score
        
        self.score_min = score_min
        self.score_max = score_max
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        
        # Calculate scaling factors
        # Score = Offset - Factor * ln(Odds)
        # where Odds = P(Good) / P(Bad) = (1-p) / p
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        
        # Score bands (FICO-style)
        self.score_bands = [
            ScoreBand('Exceptional', 800, 850, 
                     'Well above average, demonstrating exceptional creditworthiness',
                     '10-12%'),
            ScoreBand('Very Good', 740, 799,
                     'Above average, likely to receive better than average rates',
                     '13-15%'),
            ScoreBand('Good', 670, 739,
                     'Near or slightly above average, acceptable to most lenders',
                     '16-18%'),
            ScoreBand('Fair', 580, 669,
                     'Below average, subprime borrower',
                     '19-24%'),
            ScoreBand('Poor', 300, 579,
                     'Well below average, difficulty getting approved',
                     '25%+'),
        ]
    
    def probability_to_score(self, prob_default: np.ndarray) -> np.ndarray:
        """Convert probability of default to credit score"""
        
        # Avoid division by zero and log of zero
        prob_default = np.clip(prob_default, 1e-6, 1 - 1e-6)
        
        # Calculate odds (good/bad)
        odds = (1 - prob_default) / prob_default
        
        # Calculate score
        scores = self.offset - self.factor * np.log(odds)
        
        # Clip to valid range
        scores = np.clip(scores, self.score_min, self.score_max)
        
        # Round to integers
        return np.round(scores).astype(int)
    
    def score_to_probability(self, scores: np.ndarray) -> np.ndarray:
        """Convert credit score back to probability of default"""
        
        # Inverse of the scoring formula
        log_odds = (self.offset - scores) / self.factor
        odds = np.exp(log_odds)
        
        # prob_default = 1 / (1 + odds)
        prob_default = 1 / (1 + odds)
        
        return prob_default
    
    def get_score_band(self, score: int) -> ScoreBand:
        """Get the score band for a given score"""
        for band in self.score_bands:
            if band.min_score <= score <= band.max_score:
                return band
        return self.score_bands[-1]  # Return lowest band if out of range
    
    def get_score_bands(self, scores: np.ndarray) -> np.ndarray:
        """Get score band names for array of scores"""
        band_names = []
        for score in scores:
            band = self.get_score_band(score)
            band_names.append(band.name)
        return np.array(band_names)
    
    def generate_score_distribution_report(self, 
                                           scores: np.ndarray, 
                                           y_true: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Generate detailed score distribution report"""
        
        report_data = []
        
        for band in self.score_bands:
            mask = (scores >= band.min_score) & (scores <= band.max_score)
            count = mask.sum()
            pct = count / len(scores) * 100
            
            row = {
                'Score Band': band.name,
                'Score Range': f'{band.min_score}-{band.max_score}',
                'Count': count,
                'Percentage': f'{pct:.1f}%',
                'Typical APR': band.typical_apr_range,
            }
            
            if y_true is not None:
                band_defaults = y_true[mask]
                if len(band_defaults) > 0:
                    row['Default Rate'] = f'{band_defaults.mean():.1%}'
                    row['Defaults'] = band_defaults.sum()
                else:
                    row['Default Rate'] = 'N/A'
                    row['Defaults'] = 0
            
            report_data.append(row)
        
        return pd.DataFrame(report_data)
    
    def calculate_score_statistics(self, scores: np.ndarray) -> Dict:
        """Calculate comprehensive score statistics"""
        return {
            'mean': np.mean(scores),
            'median': np.median(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'percentile_10': np.percentile(scores, 10),
            'percentile_25': np.percentile(scores, 25),
            'percentile_50': np.percentile(scores, 50),
            'percentile_75': np.percentile(scores, 75),
            'percentile_90': np.percentile(scores, 90),
            'skewness': stats.skew(scores),
            'kurtosis': stats.kurtosis(scores),
        }


class ScorecardBuilder:
    """
    Build traditional credit scorecard from logistic regression model
    Converts coefficients to point system for transparency
    """
    
    def __init__(self, base_points: int = 600, pdo: int = 20, target_odds: float = 1.0):
        self.base_points = base_points
        self.pdo = pdo
        self.target_odds = target_odds
        self.factor = pdo / np.log(2)
        self.scorecard = None
        
    def build_scorecard(self, 
                        model, 
                        feature_names: List[str],
                        feature_bins: Optional[Dict] = None) -> pd.DataFrame:
        """
        Build scorecard from logistic regression coefficients
        
        Score = Base + sum(Points_i)
        where Points_i = -Coefficient_i * Factor * (Value_i - Reference_i)
        """
        
        if not hasattr(model, 'coef_'):
            raise ValueError("Model must be logistic regression with coefficients")
        
        coefficients = model.coef_[0]
        intercept = model.intercept_[0]
        
        scorecard_rows = []
        
        # Base score contribution from intercept
        base_contribution = -self.factor * intercept
        
        for feat_name, coef in zip(feature_names, coefficients):
            # Points per unit change
            points_per_unit = -self.factor * coef
            
            scorecard_rows.append({
                'Feature': feat_name,
                'Coefficient': coef,
                'Points_per_unit': points_per_unit,
                'Impact_direction': 'Increases score' if coef < 0 else 'Decreases score'
            })
        
        self.scorecard = pd.DataFrame(scorecard_rows)
        self.scorecard['Abs_points_per_unit'] = np.abs(self.scorecard['Points_per_unit'])
        self.scorecard = self.scorecard.sort_values('Abs_points_per_unit', ascending=False)
        
        return self.scorecard
    
    def explain_score(self, 
                      score: int, 
                      feature_values: Dict[str, float],
                      top_n: int = 5) -> pd.DataFrame:
        """Explain what factors contributed most to a score"""
        
        if self.scorecard is None:
            raise ValueError("Scorecard must be built first")
        
        contributions = []
        
        for _, row in self.scorecard.iterrows():
            feat = row['Feature']
            if feat in feature_values:
                value = feature_values[feat]
                points = row['Points_per_unit'] * value
                
                contributions.append({
                    'Feature': feat,
                    'Value': value,
                    'Points_contributed': points,
                    'Impact': 'Positive' if points > 0 else 'Negative'
                })
        
        contrib_df = pd.DataFrame(contributions)
        contrib_df['Abs_points'] = np.abs(contrib_df['Points_contributed'])
        contrib_df = contrib_df.sort_values('Abs_points', ascending=False).head(top_n)
        
        return contrib_df


class ScoreSimulator:
    """Simulate score changes based on behavior changes"""
    
    def __init__(self, model, feature_engineer, scorer: FICOStyleScorer):
        self.model = model
        self.feature_engineer = feature_engineer
        self.scorer = scorer
        
    def simulate_score_change(self,
                              base_data: pd.DataFrame,
                              changes: Dict[str, float]) -> Dict:
        """
        Simulate how score would change if certain features changed
        
        Example:
        changes = {'credit_utilization': -20}  # Reduce utilization by 20 points
        """
        
        # Get base score
        X_base, _ = self.feature_engineer.transform(base_data)
        prob_base = self.model.predict_proba(X_base)[:, 1]
        score_base = self.scorer.probability_to_score(prob_base)
        
        # Apply changes
        modified_data = base_data.copy()
        for feature, delta in changes.items():
            if feature in modified_data.columns:
                modified_data[feature] = modified_data[feature] + delta
        
        # Get new score
        X_new, _ = self.feature_engineer.transform(modified_data)
        prob_new = self.model.predict_proba(X_new)[:, 1]
        score_new = self.scorer.probability_to_score(prob_new)
        
        return {
            'base_score': score_base[0],
            'new_score': score_new[0],
            'score_change': score_new[0] - score_base[0],
            'base_probability': prob_base[0],
            'new_probability': prob_new[0],
            'base_band': self.scorer.get_score_band(score_base[0]).name,
            'new_band': self.scorer.get_score_band(score_new[0]).name,
        }


if __name__ == "__main__":
    # Test scoring system
    scorer = FICOStyleScorer()
    
    # Test probability to score conversion
    probs = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70])
    scores = scorer.probability_to_score(probs)
    
    print("Probability to Score Conversion:")
    print("-" * 40)
    for p, s in zip(probs, scores):
        band = scorer.get_score_band(s)
        print(f"P(Default)={p:.2%} -> Score={s} ({band.name})")
    
    # Test score distribution report
    print("\n\nScore Distribution Report (sample data):")
    print("-" * 60)
    sample_scores = np.random.normal(680, 80, 10000).clip(300, 850).astype(int)
    sample_defaults = (np.random.random(10000) < scorer.score_to_probability(sample_scores)).astype(int)
    
    report = scorer.generate_score_distribution_report(sample_scores, sample_defaults)
    print(report.to_string(index=False))
    
    print("\n\nScore Statistics:")
    print("-" * 40)
    stats = scorer.calculate_score_statistics(sample_scores)
    for k, v in stats.items():
        print(f"{k}: {v:.2f}")
