"""
Production Monitoring Module
- Population Stability Index (PSI)
- Characteristic Stability Index (CSI)
- Score Distribution Monitoring
- Model Performance Drift Detection
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StabilityResult:
    """Container for stability analysis results"""
    index_value: float
    interpretation: str
    is_significant: bool
    details: Dict


class PopulationStabilityMonitor:
    """
    Population Stability Index (PSI) monitoring
    
    PSI measures how much a variable's distribution has shifted
    compared to a baseline distribution.
    
    PSI < 0.10: No significant change
    0.10 <= PSI < 0.25: Moderate change, investigation needed
    PSI >= 0.25: Significant change, action required
    """
    
    def __init__(self, n_bins: int = 10, min_pct: float = 0.0001):
        self.n_bins = n_bins
        self.min_pct = min_pct  # Avoid log(0)
        
    def calculate_psi(self, 
                      expected: np.ndarray, 
                      actual: np.ndarray,
                      bins: Optional[np.ndarray] = None) -> StabilityResult:
        """
        Calculate PSI between expected (baseline) and actual (current) distributions
        
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        
        # Create bins from expected distribution if not provided
        if bins is None:
            bins = np.percentile(expected, np.linspace(0, 100, self.n_bins + 1))
            bins[0] = -np.inf
            bins[-1] = np.inf
        
        # Calculate percentages in each bin
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
        
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Apply minimum percentage to avoid division by zero
        expected_pct = np.maximum(expected_pct, self.min_pct)
        actual_pct = np.maximum(actual_pct, self.min_pct)
        
        # Calculate PSI
        psi_components = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        psi = np.sum(psi_components)
        
        # Interpretation
        if psi < 0.10:
            interpretation = "No significant change"
            is_significant = False
        elif psi < 0.25:
            interpretation = "Moderate change - investigation recommended"
            is_significant = True
        else:
            interpretation = "Significant change - action required"
            is_significant = True
        
        details = {
            'psi_by_bin': psi_components.tolist(),
            'expected_pct': expected_pct.tolist(),
            'actual_pct': actual_pct.tolist(),
            'bins': bins.tolist(),
            'max_shift_bin': int(np.argmax(np.abs(actual_pct - expected_pct))),
        }
        
        return StabilityResult(
            index_value=psi,
            interpretation=interpretation,
            is_significant=is_significant,
            details=details
        )
    
    def calculate_score_psi(self,
                            expected_scores: np.ndarray,
                            actual_scores: np.ndarray) -> StabilityResult:
        """Calculate PSI specifically for credit scores"""
        
        # Use fixed score bins (FICO-style)
        bins = np.array([300, 500, 550, 600, 650, 700, 750, 800, 850])
        
        return self.calculate_psi(expected_scores, actual_scores, bins=bins)


class CharacteristicStabilityMonitor:
    """
    Characteristic Stability Index (CSI) monitoring
    
    CSI measures shifts in individual feature distributions,
    helping identify which inputs have drifted.
    """
    
    def __init__(self, n_bins: int = 10, min_pct: float = 0.0001):
        self.n_bins = n_bins
        self.min_pct = min_pct
        self.psi_calculator = PopulationStabilityMonitor(n_bins, min_pct)
        
    def calculate_csi(self,
                      expected_df: pd.DataFrame,
                      actual_df: pd.DataFrame,
                      numeric_features: List[str],
                      categorical_features: List[str]) -> pd.DataFrame:
        """Calculate CSI for all features"""
        
        results = []
        
        # Numeric features
        for feature in numeric_features:
            if feature in expected_df.columns and feature in actual_df.columns:
                exp_vals = expected_df[feature].dropna().values
                act_vals = actual_df[feature].dropna().values
                
                if len(exp_vals) > 0 and len(act_vals) > 0:
                    result = self.psi_calculator.calculate_psi(exp_vals, act_vals)
                    
                    results.append({
                        'feature': feature,
                        'type': 'numeric',
                        'csi': result.index_value,
                        'interpretation': result.interpretation,
                        'is_significant': result.is_significant,
                        'expected_mean': np.mean(exp_vals),
                        'actual_mean': np.mean(act_vals),
                        'mean_shift_pct': (np.mean(act_vals) - np.mean(exp_vals)) / np.mean(exp_vals) * 100 if np.mean(exp_vals) != 0 else 0,
                    })
        
        # Categorical features
        for feature in categorical_features:
            if feature in expected_df.columns and feature in actual_df.columns:
                csi = self._calculate_categorical_csi(
                    expected_df[feature].dropna(),
                    actual_df[feature].dropna()
                )
                
                results.append({
                    'feature': feature,
                    'type': 'categorical',
                    'csi': csi['value'],
                    'interpretation': csi['interpretation'],
                    'is_significant': csi['is_significant'],
                    'expected_mean': None,
                    'actual_mean': None,
                    'mean_shift_pct': None,
                })
        
        return pd.DataFrame(results).sort_values('csi', ascending=False)
    
    def _calculate_categorical_csi(self, 
                                   expected: pd.Series, 
                                   actual: pd.Series) -> Dict:
        """Calculate CSI for categorical feature"""
        
        # Get all categories
        all_cats = set(expected.unique()) | set(actual.unique())
        
        # Calculate percentages
        exp_pct = expected.value_counts(normalize=True)
        act_pct = actual.value_counts(normalize=True)
        
        csi = 0
        for cat in all_cats:
            exp_p = exp_pct.get(cat, self.min_pct)
            act_p = act_pct.get(cat, self.min_pct)
            
            exp_p = max(exp_p, self.min_pct)
            act_p = max(act_p, self.min_pct)
            
            csi += (act_p - exp_p) * np.log(act_p / exp_p)
        
        if csi < 0.10:
            interpretation = "No significant change"
            is_significant = False
        elif csi < 0.25:
            interpretation = "Moderate change"
            is_significant = True
        else:
            interpretation = "Significant change"
            is_significant = True
        
        return {
            'value': csi,
            'interpretation': interpretation,
            'is_significant': is_significant
        }


class ModelPerformanceMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, performance_threshold: float = 0.05):
        self.performance_threshold = performance_threshold
        self.baseline_metrics = None
        
    def set_baseline(self, 
                     y_true: np.ndarray, 
                     y_proba: np.ndarray,
                     scores: np.ndarray):
        """Set baseline performance metrics"""
        
        from sklearn.metrics import roc_auc_score, brier_score_loss
        
        self.baseline_metrics = {
            'auc': roc_auc_score(y_true, y_proba),
            'brier': brier_score_loss(y_true, y_proba),
            'ks': self._calculate_ks(y_true, y_proba),
            'default_rate': np.mean(y_true),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
        }
        
        return self.baseline_metrics
    
    def evaluate_performance(self,
                            y_true: np.ndarray,
                            y_proba: np.ndarray,
                            scores: np.ndarray) -> Dict:
        """Evaluate current performance vs baseline"""
        
        from sklearn.metrics import roc_auc_score, brier_score_loss
        
        current_metrics = {
            'auc': roc_auc_score(y_true, y_proba),
            'brier': brier_score_loss(y_true, y_proba),
            'ks': self._calculate_ks(y_true, y_proba),
            'default_rate': np.mean(y_true),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
        }
        
        # Compare to baseline
        comparisons = {}
        alerts = []
        
        if self.baseline_metrics:
            for metric, current_val in current_metrics.items():
                baseline_val = self.baseline_metrics[metric]
                
                if baseline_val != 0:
                    pct_change = (current_val - baseline_val) / abs(baseline_val) * 100
                else:
                    pct_change = 0
                
                comparisons[metric] = {
                    'baseline': baseline_val,
                    'current': current_val,
                    'change': current_val - baseline_val,
                    'pct_change': pct_change,
                }
                
                # Check for significant degradation
                if metric == 'auc' and pct_change < -5:
                    alerts.append(f"AUC degraded by {abs(pct_change):.1f}%")
                elif metric == 'ks' and pct_change < -10:
                    alerts.append(f"KS statistic degraded by {abs(pct_change):.1f}%")
                elif metric == 'default_rate' and abs(pct_change) > 20:
                    alerts.append(f"Default rate changed by {pct_change:.1f}%")
        
        return {
            'current_metrics': current_metrics,
            'comparisons': comparisons,
            'alerts': alerts,
            'requires_action': len(alerts) > 0
        }
    
    def _calculate_ks(self, y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Calculate KS statistic"""
        df = pd.DataFrame({'y': y_true, 'prob': y_proba})
        df = df.sort_values('prob', ascending=False)
        
        df['cum_events'] = df['y'].cumsum() / df['y'].sum()
        df['cum_non_events'] = (1 - df['y']).cumsum() / (1 - df['y']).sum()
        
        return (df['cum_events'] - df['cum_non_events']).abs().max()


class DriftDetector:
    """Statistical drift detection using various tests"""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def detect_drift_ks_test(self,
                            expected: np.ndarray,
                            actual: np.ndarray) -> Dict:
        """Kolmogorov-Smirnov test for distribution drift"""
        
        statistic, p_value = stats.ks_2samp(expected, actual)
        
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < self.significance_level,
            'interpretation': 'Distributions are different' if p_value < self.significance_level else 'No significant difference'
        }
    
    def detect_drift_chi_squared(self,
                                 expected: pd.Series,
                                 actual: pd.Series) -> Dict:
        """Chi-squared test for categorical drift"""
        
        # Get combined categories
        all_cats = list(set(expected.unique()) | set(actual.unique()))
        
        # Count frequencies
        exp_counts = expected.value_counts()
        act_counts = actual.value_counts()
        
        # Align counts
        exp_freq = np.array([exp_counts.get(c, 0) for c in all_cats])
        act_freq = np.array([act_counts.get(c, 0) for c in all_cats])
        
        # Scale expected to match actual total
        exp_freq_scaled = exp_freq * (act_freq.sum() / exp_freq.sum())
        
        # Chi-squared test
        # Only include categories with expected count > 0
        mask = exp_freq_scaled > 0
        
        if mask.sum() < 2:
            return {
                'test': 'Chi-squared',
                'statistic': 0,
                'p_value': 1,
                'drift_detected': False,
                'interpretation': 'Insufficient categories for test'
            }
        
        statistic, p_value = stats.chisquare(act_freq[mask], exp_freq_scaled[mask])
        
        return {
            'test': 'Chi-squared',
            'statistic': statistic,
            'p_value': p_value,
            'drift_detected': p_value < self.significance_level,
            'interpretation': 'Distributions are different' if p_value < self.significance_level else 'No significant difference'
        }
    
    def comprehensive_drift_analysis(self,
                                     expected_df: pd.DataFrame,
                                     actual_df: pd.DataFrame,
                                     numeric_features: List[str],
                                     categorical_features: List[str]) -> pd.DataFrame:
        """Run comprehensive drift analysis on all features"""
        
        results = []
        
        for feature in numeric_features:
            if feature in expected_df.columns and feature in actual_df.columns:
                exp_vals = expected_df[feature].dropna().values
                act_vals = actual_df[feature].dropna().values
                
                if len(exp_vals) > 10 and len(act_vals) > 10:
                    ks_result = self.detect_drift_ks_test(exp_vals, act_vals)
                    
                    results.append({
                        'feature': feature,
                        'type': 'numeric',
                        'test': ks_result['test'],
                        'statistic': ks_result['statistic'],
                        'p_value': ks_result['p_value'],
                        'drift_detected': ks_result['drift_detected'],
                    })
        
        for feature in categorical_features:
            if feature in expected_df.columns and feature in actual_df.columns:
                chi_result = self.detect_drift_chi_squared(
                    expected_df[feature].dropna(),
                    actual_df[feature].dropna()
                )
                
                results.append({
                    'feature': feature,
                    'type': 'categorical',
                    'test': chi_result['test'],
                    'statistic': chi_result['statistic'],
                    'p_value': chi_result['p_value'],
                    'drift_detected': chi_result['drift_detected'],
                })
        
        return pd.DataFrame(results).sort_values('p_value')


class MonitoringReport:
    """Generate comprehensive monitoring reports"""
    
    def __init__(self):
        self.psi_monitor = PopulationStabilityMonitor()
        self.csi_monitor = CharacteristicStabilityMonitor()
        self.perf_monitor = ModelPerformanceMonitor()
        self.drift_detector = DriftDetector()
        
    def generate_full_report(self,
                             baseline_data: Dict,
                             current_data: Dict,
                             model,
                             feature_engineer) -> Dict:
        """Generate full monitoring report"""
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'summary': {},
            'psi_analysis': {},
            'csi_analysis': {},
            'performance_analysis': {},
            'drift_analysis': {},
            'recommendations': []
        }
        
        # Get predictions for both datasets
        X_baseline, y_baseline = feature_engineer.transform(baseline_data['df'])
        X_current, y_current = feature_engineer.transform(current_data['df'])
        
        prob_baseline = model.predict_proba(X_baseline)[:, 1]
        prob_current = model.predict_proba(X_current)[:, 1]
        
        # Score PSI
        from scoring import FICOStyleScorer
        scorer = FICOStyleScorer()
        
        scores_baseline = scorer.probability_to_score(prob_baseline)
        scores_current = scorer.probability_to_score(prob_current)
        
        psi_result = self.psi_monitor.calculate_score_psi(scores_baseline, scores_current)
        report['psi_analysis'] = {
            'score_psi': psi_result.index_value,
            'interpretation': psi_result.interpretation,
            'is_significant': psi_result.is_significant,
        }
        
        # CSI Analysis
        numeric_features = baseline_data['df'].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = baseline_data['df'].select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from features
        numeric_features = [f for f in numeric_features if f != 'default']
        
        csi_df = self.csi_monitor.calculate_csi(
            baseline_data['df'],
            current_data['df'],
            numeric_features,
            categorical_features
        )
        report['csi_analysis'] = {
            'summary': csi_df.to_dict('records'),
            'features_with_drift': csi_df[csi_df['is_significant']]['feature'].tolist(),
        }
        
        # Performance Analysis
        self.perf_monitor.set_baseline(y_baseline, prob_baseline, scores_baseline)
        perf_result = self.perf_monitor.evaluate_performance(y_current, prob_current, scores_current)
        report['performance_analysis'] = perf_result
        
        # Statistical Drift
        drift_df = self.drift_detector.comprehensive_drift_analysis(
            baseline_data['df'],
            current_data['df'],
            numeric_features,
            categorical_features
        )
        report['drift_analysis'] = {
            'summary': drift_df.to_dict('records'),
            'features_with_drift': drift_df[drift_df['drift_detected']]['feature'].tolist(),
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        # Summary
        report['summary'] = {
            'score_psi': psi_result.index_value,
            'features_drifted': len(report['csi_analysis']['features_with_drift']),
            'performance_alerts': len(perf_result['alerts']),
            'overall_status': 'ACTION REQUIRED' if (
                psi_result.is_significant or 
                perf_result['requires_action'] or 
                len(report['csi_analysis']['features_with_drift']) > 3
            ) else 'STABLE'
        }
        
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # PSI recommendations
        if report['psi_analysis']['is_significant']:
            recommendations.append(
                "Score distribution has shifted significantly. "
                "Consider retraining the model or investigating population changes."
            )
        
        # CSI recommendations
        drifted_features = report['csi_analysis']['features_with_drift']
        if len(drifted_features) > 0:
            recommendations.append(
                f"Features with significant drift: {', '.join(drifted_features[:5])}. "
                "Investigate data quality and upstream changes."
            )
        
        # Performance recommendations
        if report['performance_analysis']['requires_action']:
            for alert in report['performance_analysis']['alerts']:
                recommendations.append(f"Performance Alert: {alert}")
        
        if not recommendations:
            recommendations.append("Model is performing within expected parameters. Continue monitoring.")
        
        return recommendations


if __name__ == "__main__":
    # Test monitoring components
    print("Testing Population Stability Monitor...")
    
    # Generate sample data
    np.random.seed(42)
    expected = np.random.normal(680, 80, 10000)
    actual = np.random.normal(670, 90, 10000)  # Slight drift
    
    psi_monitor = PopulationStabilityMonitor()
    result = psi_monitor.calculate_psi(expected, actual)
    
    print(f"PSI: {result.index_value:.4f}")
    print(f"Interpretation: {result.interpretation}")
    print(f"Significant: {result.is_significant}")
    
    # Test KS drift detection
    print("\nTesting Drift Detection...")
    detector = DriftDetector()
    ks_result = detector.detect_drift_ks_test(expected, actual)
    print(f"KS Test: {ks_result}")
