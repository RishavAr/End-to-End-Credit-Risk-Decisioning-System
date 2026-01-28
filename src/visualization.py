"""
Visualization Module for Credit Risk System
- Model comparison charts
- Score distributions
- Feature importance
- Calibration curves
- Monitoring dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class CreditRiskVisualizer:
    """Visualization suite for credit risk models"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_roc_curves(self, 
                        models: Dict,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curves for all models"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for (name, model), color in zip(models.items(), colors):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, y_proba)
            
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{name} (AUC = {auc:.3f})')
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_precision_recall_curves(self,
                                     models: Dict,
                                     X_test: np.ndarray,
                                     y_test: np.ndarray,
                                     save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curves for all models"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        for (name, model), color in zip(models.items(), colors):
            y_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            
            from sklearn.metrics import average_precision_score
            ap = average_precision_score(y_test, y_proba)
            
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{name} (AP = {ap:.3f})')
        
        # Baseline
        baseline = y_test.mean()
        ax.axhline(y=baseline, color='k', linestyle='--', lw=1, 
                   label=f'Baseline ({baseline:.3f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_calibration_curves(self,
                                models: Dict,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                n_bins: int = 10,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot calibration curves for all models"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        # Calibration curve
        ax1 = axes[0]
        for (name, model), color in zip(models.items(), colors):
            y_proba = model.predict_proba(X_test)[:, 1]
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)
            
            ax1.plot(prob_pred, prob_true, marker='o', color=color, 
                    lw=2, label=name)
        
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
        ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax1.set_ylabel('Fraction of Positives', fontsize=12)
        ax1.set_title('Calibration Curves', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=9)
        
        # Prediction distribution
        ax2 = axes[1]
        for (name, model), color in zip(models.items(), colors):
            y_proba = model.predict_proba(X_test)[:, 1]
            ax2.hist(y_proba, bins=50, alpha=0.5, color=color, 
                    label=name, density=True)
        
        ax2.set_xlabel('Predicted Probability', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self,
                                importance_df: pd.DataFrame,
                                top_n: int = 20,
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot feature importance comparison across models"""
        
        models = importance_df['model'].unique()
        n_models = len(models)
        
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, model_name in zip(axes, models):
            model_df = importance_df[importance_df['model'] == model_name].head(top_n)
            
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_df)))
            
            ax.barh(range(len(model_df)), 
                   model_df['importance_normalized'].values,
                   color=colors)
            ax.set_yticks(range(len(model_df)))
            ax.set_yticklabels(model_df['feature'].values, fontsize=9)
            ax.set_xlabel('Normalized Importance', fontsize=11)
            ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
            ax.invert_yaxis()
        
        plt.suptitle(f'Top {top_n} Feature Importances by Model', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_score_distribution(self,
                                scores: np.ndarray,
                                y_true: np.ndarray,
                                score_bands: List[Tuple[str, int, int]],
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot FICO-style score distribution"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall distribution
        ax1 = axes[0, 0]
        ax1.hist(scores, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.axvline(np.median(scores), color='red', linestyle='--', lw=2, 
                   label=f'Median: {np.median(scores):.0f}')
        ax1.axvline(np.mean(scores), color='orange', linestyle='--', lw=2,
                   label=f'Mean: {np.mean(scores):.0f}')
        ax1.set_xlabel('Credit Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Overall Score Distribution', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # Distribution by outcome
        ax2 = axes[0, 1]
        scores_good = scores[y_true == 0]
        scores_bad = scores[y_true == 1]
        
        ax2.hist(scores_good, bins=40, alpha=0.6, label='Non-Default', 
                color='green', density=True)
        ax2.hist(scores_bad, bins=40, alpha=0.6, label='Default', 
                color='red', density=True)
        ax2.set_xlabel('Credit Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Score Distribution by Outcome', fontsize=14, fontweight='bold')
        ax2.legend()
        
        # Default rate by score band
        ax3 = axes[1, 0]
        score_bins = [300, 500, 580, 670, 740, 800, 850]
        score_labels = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        
        score_cats = pd.cut(scores, bins=score_bins, labels=score_labels[:len(score_bins)-1])
        df_temp = pd.DataFrame({'score_band': score_cats, 'default': y_true})
        default_rates = df_temp.groupby('score_band')['default'].mean()
        
        bars = ax3.bar(default_rates.index, default_rates.values, 
                      color=['darkred', 'red', 'orange', 'yellowgreen', 'green'])
        ax3.set_xlabel('Score Band', fontsize=12)
        ax3.set_ylabel('Default Rate', fontsize=12)
        ax3.set_title('Default Rate by Score Band', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, rate in zip(bars, default_rates.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', fontsize=10)
        
        # Cumulative distribution
        ax4 = axes[1, 1]
        sorted_scores = np.sort(scores)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        
        ax4.plot(sorted_scores, cumulative, lw=2, color='steelblue')
        ax4.fill_between(sorted_scores, cumulative, alpha=0.3)
        ax4.set_xlabel('Credit Score', fontsize=12)
        ax4.set_ylabel('Cumulative Proportion', fontsize=12)
        ax4.set_title('Cumulative Score Distribution', fontsize=14, fontweight='bold')
        
        # Add percentile lines
        for pct in [25, 50, 75]:
            val = np.percentile(scores, pct)
            ax4.axvline(val, color='gray', linestyle='--', alpha=0.7)
            ax4.text(val, 0.02, f'P{pct}: {val:.0f}', rotation=90, 
                    fontsize=9, va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_ks_curve(self,
                      y_true: np.ndarray,
                      y_proba: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot KS (Kolmogorov-Smirnov) curve"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by probability
        df = pd.DataFrame({'y': y_true, 'prob': y_proba})
        df = df.sort_values('prob', ascending=True).reset_index(drop=True)
        
        # Cumulative distributions
        df['pct_population'] = (df.index + 1) / len(df)
        df['cum_events'] = df['y'].cumsum() / df['y'].sum()
        df['cum_non_events'] = (1 - df['y']).cumsum() / (1 - df['y']).sum()
        
        # Find max KS point
        df['ks_diff'] = df['cum_non_events'] - df['cum_events']
        max_ks_idx = df['ks_diff'].idxmax()
        max_ks = df.loc[max_ks_idx, 'ks_diff']
        max_ks_pct = df.loc[max_ks_idx, 'pct_population']
        
        # Plot
        ax.plot(df['pct_population'], df['cum_events'], 
               lw=2, label='Cumulative Default', color='red')
        ax.plot(df['pct_population'], df['cum_non_events'], 
               lw=2, label='Cumulative Non-Default', color='green')
        
        # Max KS line
        ax.axvline(max_ks_pct, color='blue', linestyle='--', lw=2, alpha=0.7)
        ax.annotate(f'Max KS = {max_ks:.3f}\nat {max_ks_pct:.1%}',
                   xy=(max_ks_pct, 0.5), fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.fill_between(df['pct_population'], 
                       df['cum_events'], df['cum_non_events'],
                       alpha=0.2, color='blue')
        
        ax.set_xlabel('Cumulative % of Population (sorted by probability)', fontsize=12)
        ax.set_ylabel('Cumulative %', fontsize=12)
        ax.set_title(f'KS Curve (KS Statistic = {max_ks:.3f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_gains_chart(self,
                         y_true: np.ndarray,
                         y_proba: np.ndarray,
                         save_path: Optional[str] = None) -> plt.Figure:
        """Plot cumulative gains chart"""
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Sort by probability descending
        df = pd.DataFrame({'y': y_true, 'prob': y_proba})
        df = df.sort_values('prob', ascending=False).reset_index(drop=True)
        
        # Cumulative calculations
        df['pct_population'] = (df.index + 1) / len(df) * 100
        df['cum_events'] = df['y'].cumsum()
        df['pct_events'] = df['cum_events'] / df['y'].sum() * 100
        
        # Plot model curve
        ax.plot(df['pct_population'], df['pct_events'], 
               lw=2, label='Model', color='blue')
        
        # Perfect model
        total_events = df['y'].sum()
        perfect_x = [0, total_events / len(df) * 100, 100]
        perfect_y = [0, 100, 100]
        ax.plot(perfect_x, perfect_y, lw=2, linestyle='--', 
               label='Perfect Model', color='green')
        
        # Random model
        ax.plot([0, 100], [0, 100], lw=2, linestyle=':', 
               label='Random', color='gray')
        
        ax.fill_between(df['pct_population'], df['pct_events'], 
                       df['pct_population'], alpha=0.2, color='blue')
        
        ax.set_xlabel('% of Population (ranked by probability)', fontsize=12)
        ax.set_ylabel('% of Defaults Captured', fontsize=12)
        ax.set_title('Cumulative Gains Chart', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_psi_analysis(self,
                          expected_scores: np.ndarray,
                          actual_scores: np.ndarray,
                          psi_value: float,
                          save_path: Optional[str] = None) -> plt.Figure:
        """Plot PSI analysis comparison"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Score distributions
        ax1 = axes[0]
        ax1.hist(expected_scores, bins=40, alpha=0.6, label='Baseline', 
                color='blue', density=True)
        ax1.hist(actual_scores, bins=40, alpha=0.6, label='Current', 
                color='orange', density=True)
        ax1.set_xlabel('Credit Score', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title(f'Score Distribution Comparison (PSI = {psi_value:.4f})', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        
        # Add PSI interpretation
        if psi_value < 0.10:
            status = "STABLE"
            color = 'green'
        elif psi_value < 0.25:
            status = "MODERATE SHIFT"
            color = 'orange'
        else:
            status = "SIGNIFICANT SHIFT"
            color = 'red'
        
        ax1.text(0.02, 0.98, f'Status: {status}', transform=ax1.transAxes,
                fontsize=12, fontweight='bold', color=color,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bin-by-bin comparison
        ax2 = axes[1]
        bins = np.linspace(300, 850, 12)
        
        exp_hist, _ = np.histogram(expected_scores, bins=bins, density=True)
        act_hist, _ = np.histogram(actual_scores, bins=bins, density=True)
        
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = (bins[1] - bins[0]) * 0.35
        
        ax2.bar(bin_centers - width, exp_hist, width * 2, alpha=0.7, 
               label='Baseline', color='blue')
        ax2.bar(bin_centers + width, act_hist, width * 2, alpha=0.7, 
               label='Current', color='orange')
        
        ax2.set_xlabel('Credit Score', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Bin-by-Bin Distribution Comparison', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison_summary(self,
                                      eval_df: pd.DataFrame,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive model comparison summary"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        metrics = ['auc_roc', 'gini', 'ks_statistic', 'brier_score']
        titles = ['AUC-ROC', 'Gini Coefficient', 'KS Statistic', 'Brier Score (lower is better)']
        
        for ax, metric, title in zip(axes.flatten(), metrics, titles):
            values = eval_df[metric].values
            models = eval_df.index.tolist()
            
            colors = ['green' if eval_df.loc[m, 'is_best'] else 'steelblue' 
                     for m in models]
            
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', fontsize=10)
            
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle('Model Comparison Summary', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig


def create_monitoring_dashboard(report: Dict, save_path: str):
    """Create comprehensive monitoring dashboard"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('Credit Risk Model Monitoring Dashboard', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall Status
    ax1 = fig.add_subplot(gs[0, 0])
    status = report['summary']['overall_status']
    color = 'green' if status == 'STABLE' else 'red'
    ax1.text(0.5, 0.5, status, fontsize=24, fontweight='bold',
            color=color, ha='center', va='center',
            transform=ax1.transAxes)
    ax1.set_title('Overall Status', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. PSI Gauge
    ax2 = fig.add_subplot(gs[0, 1])
    psi = report['psi_analysis']['score_psi']
    ax2.barh(['PSI'], [psi], color='steelblue')
    ax2.axvline(0.10, color='orange', linestyle='--', label='Moderate threshold')
    ax2.axvline(0.25, color='red', linestyle='--', label='Action threshold')
    ax2.set_xlim([0, max(0.35, psi * 1.2)])
    ax2.set_title(f'Score PSI: {psi:.4f}', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    
    # 3. Key Metrics
    ax3 = fig.add_subplot(gs[0, 2])
    if 'current_metrics' in report['performance_analysis']:
        metrics_text = '\n'.join([
            f"AUC: {report['performance_analysis']['current_metrics']['auc']:.3f}",
            f"KS: {report['performance_analysis']['current_metrics']['ks']:.3f}",
            f"Default Rate: {report['performance_analysis']['current_metrics']['default_rate']:.1%}",
        ])
    else:
        metrics_text = "Metrics unavailable"
    ax3.text(0.5, 0.5, metrics_text, fontsize=12, ha='center', va='center',
            transform=ax3.transAxes, family='monospace')
    ax3.set_title('Current Performance', fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # 4. Features with Drift
    ax4 = fig.add_subplot(gs[1, :2])
    drifted = report['csi_analysis']['features_with_drift'][:10]
    if drifted:
        ax4.barh(range(len(drifted)), [1] * len(drifted), color='orange')
        ax4.set_yticks(range(len(drifted)))
        ax4.set_yticklabels(drifted)
        ax4.set_title(f'Features with Significant Drift ({len(drifted)} total)', 
                     fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No significant drift detected', 
                fontsize=14, ha='center', va='center')
        ax4.set_title('Features with Drift', fontsize=12, fontweight='bold')
    ax4.set_xlim([0, 1.5])
    
    # 5. Alerts
    ax5 = fig.add_subplot(gs[1, 2])
    alerts = report['performance_analysis']['alerts']
    if alerts:
        alert_text = '\n'.join([f"⚠ {a}" for a in alerts[:5]])
    else:
        alert_text = "✓ No alerts"
    ax5.text(0.1, 0.5, alert_text, fontsize=10, ha='left', va='center',
            transform=ax5.transAxes, wrap=True)
    ax5.set_title('Performance Alerts', fontsize=12, fontweight='bold')
    ax5.axis('off')
    
    # 6. Recommendations
    ax6 = fig.add_subplot(gs[2, :])
    recs = report['recommendations'][:3]
    rec_text = '\n\n'.join([f"{i+1}. {r}" for i, r in enumerate(recs)])
    ax6.text(0.02, 0.95, rec_text, fontsize=10, ha='left', va='top',
            transform=ax6.transAxes, wrap=True)
    ax6.set_title('Recommendations', fontsize=12, fontweight='bold')
    ax6.axis('off')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


if __name__ == "__main__":
    print("Visualization module loaded successfully")
