#!/usr/bin/env python3
"""
End-to-End Credit Risk & Decisioning System
Production-Grade Implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from data_generator import CreditDataGenerator, generate_production_simulation_data
from feature_engineering import CreditFeatureEngineer, WOEEncoder
from model_training import CreditRiskModelTrainer, ModelExplainer
from scoring import FICOStyleScorer, ScorecardBuilder
from monitoring import PopulationStabilityMonitor, CharacteristicStabilityMonitor
from visualization import CreditRiskVisualizer


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}\n")


def main():
    OUTPUT_DIR = 'outputs'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. DATA GENERATION
    print_section("1. DATA GENERATION (LendingClub/FICO-style)")
    generator = CreditDataGenerator(n_samples=100000, default_rate=0.15, random_state=42)
    df = generator.generate()
    splits = generator.create_train_test_split(df, test_size=0.2, val_size=0.1)
    
    print(f"Training:   {len(splits['train']):,} records ({splits['train']['default'].mean():.1%} default)")
    print(f"Validation: {len(splits['validation']):,} records ({splits['validation']['default'].mean():.1%} default)")
    print(f"Test:       {len(splits['test']):,} records ({splits['test']['default'].mean():.1%} default)")

    # 2. FEATURE ENGINEERING
    print_section("2. FEATURE ENGINEERING")
    feature_engineer = CreditFeatureEngineer()
    X_train, y_train = feature_engineer.fit_transform(splits['train'])
    X_val, y_val = feature_engineer.transform(splits['validation'])
    X_test, y_test = feature_engineer.transform(splits['test'])
    feature_names = feature_engineer.get_feature_names()
    
    print(f"Engineered features: {len(feature_names)}")
    print(f"Sample features: {feature_names[:5]}...")

    # WOE Analysis
    woe_encoder = WOEEncoder()
    cat_cols = [c for c in ['home_ownership', 'employment_status', 'loan_purpose', 'grade'] 
                if c in splits['train'].columns]
    woe_encoder.fit(splits['train'], cat_cols)
    iv_summary = woe_encoder.get_iv_summary()
    print("\nInformation Value (IV) by Feature:")
    print(iv_summary.to_string(index=False))
    iv_summary.to_csv(f'{OUTPUT_DIR}/information_value.csv', index=False)

    # 3. MODEL TRAINING & COMPARISON
    print_section("3. MODEL TRAINING & COMPARISON")
    trainer = CreditRiskModelTrainer(random_state=42)
    results = trainer.train_all_models(X_train, y_train, quick_mode=True, cv=5)
    
    eval_df = trainer.evaluate_on_test(X_test, y_test)
    print("\nTest Set Performance:")
    print(eval_df[['auc_roc', 'gini', 'ks_statistic', 'precision', 'recall']].round(4))
    eval_df.to_csv(f'{OUTPUT_DIR}/model_comparison.csv')

    importance_df = trainer.get_feature_importances(feature_names)
    importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance.csv', index=False)
    
    print(f"\nTop 10 Features ({trainer.best_model_name}):")
    top_imp = importance_df[importance_df['model'] == trainer.best_model_name].head(10)
    print(top_imp[['feature', 'importance_normalized']].to_string(index=False))

    # Logistic Regression Coefficients
    if 'logistic_regression' in trainer.models:
        lr_explainer = ModelExplainer(trainer.models['logistic_regression'], feature_names)
        coef_df = lr_explainer.get_logistic_coefficients()
        coef_df.to_csv(f'{OUTPUT_DIR}/logistic_coefficients.csv', index=False)

    # 4. FICO-STYLE SCORING
    print_section("4. FICO-STYLE CREDIT SCORING")
    scorer = FICOStyleScorer()
    best_model = trainer.best_model
    y_proba_test = best_model.predict_proba(X_test)[:, 1]
    scores_test = scorer.probability_to_score(y_proba_test)
    
    score_stats = scorer.calculate_score_statistics(scores_test)
    print("Score Statistics:")
    for k, v in list(score_stats.items())[:8]:
        print(f"  {k}: {v:.1f}")
    
    score_report = scorer.generate_score_distribution_report(scores_test, y_test)
    print("\nScore Distribution by Band:")
    print(score_report.to_string(index=False))
    score_report.to_csv(f'{OUTPUT_DIR}/score_distribution.csv', index=False)

    if 'logistic_regression' in trainer.models:
        scorecard_builder = ScorecardBuilder()
        scorecard = scorecard_builder.build_scorecard(trainer.models['logistic_regression'], feature_names)
        scorecard.to_csv(f'{OUTPUT_DIR}/scorecard.csv', index=False)

    # 5. PRODUCTION MONITORING
    print_section("5. PRODUCTION MONITORING SIMULATION")
    prod_data = generate_production_simulation_data(splits['test'], drift_factor=0.15, n_samples=10000)
    X_prod, _ = feature_engineer.transform(prod_data)
    y_proba_prod = best_model.predict_proba(X_prod)[:, 1]
    scores_prod = scorer.probability_to_score(y_proba_prod)
    
    psi_monitor = PopulationStabilityMonitor()
    psi_result = psi_monitor.calculate_score_psi(scores_test, scores_prod)
    print(f"Score PSI: {psi_result.index_value:.4f}")
    print(f"Interpretation: {psi_result.interpretation}")

    csi_monitor = CharacteristicStabilityMonitor()
    numeric_feats = [c for c in splits['test'].select_dtypes(include=[np.number]).columns if c != 'default']
    cat_feats = splits['test'].select_dtypes(include=['object']).columns.tolist()
    csi_df = csi_monitor.calculate_csi(splits['test'], prod_data, numeric_feats, cat_feats)
    
    print("\nFeatures with Significant Drift:")
    drifted = csi_df[csi_df['is_significant']]
    if len(drifted) > 0:
        print(drifted[['feature', 'csi', 'interpretation']].head(10).to_string(index=False))
    else:
        print("No significant drift detected")
    csi_df.to_csv(f'{OUTPUT_DIR}/csi_analysis.csv', index=False)

    # 6. VISUALIZATION
    print_section("6. GENERATING VISUALIZATIONS")
    viz = CreditRiskVisualizer()
    
    fig = viz.plot_roc_curves(trainer.models, X_test, y_test)
    plt.savefig(f'{OUTPUT_DIR}/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ ROC curves")

    fig = viz.plot_precision_recall_curves(trainer.models, X_test, y_test)
    plt.savefig(f'{OUTPUT_DIR}/pr_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Precision-Recall curves")

    fig = viz.plot_calibration_curves(trainer.models, X_test, y_test)
    plt.savefig(f'{OUTPUT_DIR}/calibration_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Calibration curves")

    fig = viz.plot_feature_importance(importance_df, top_n=15)
    plt.savefig(f'{OUTPUT_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Feature importance")

    fig = viz.plot_score_distribution(scores_test, y_test, [])
    plt.savefig(f'{OUTPUT_DIR}/score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Score distribution")

    fig = viz.plot_ks_curve(y_test, y_proba_test)
    plt.savefig(f'{OUTPUT_DIR}/ks_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ KS curve")

    fig = viz.plot_gains_chart(y_test, y_proba_test)
    plt.savefig(f'{OUTPUT_DIR}/gains_chart.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Gains chart")

    fig = viz.plot_psi_analysis(scores_test, scores_prod, psi_result.index_value)
    plt.savefig(f'{OUTPUT_DIR}/psi_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ PSI analysis")

    fig = viz.plot_model_comparison_summary(eval_df)
    plt.savefig(f'{OUTPUT_DIR}/model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Model comparison summary")

    # FINAL SUMMARY
    print_section("PIPELINE COMPLETE")
    print(f"Best Model: {trainer.best_model_name}")
    print(f"Test AUC:   {eval_df.loc[trainer.best_model_name, 'auc_roc']:.4f}")
    print(f"Test KS:    {eval_df.loc[trainer.best_model_name, 'ks_statistic']:.4f}")
    print(f"Gini:       {eval_df.loc[trainer.best_model_name, 'gini']:.4f}")
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    
    return {
        'trainer': trainer, 
        'feature_engineer': feature_engineer, 
        'scorer': scorer, 
        'eval_df': eval_df,
        'score_stats': score_stats,
        'psi_result': psi_result
    }


if __name__ == "__main__":
    results = main()
