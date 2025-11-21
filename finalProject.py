# CS 320 Final Project
# Project Members: Cooper Jumamil, Havanna Robbins, Haylee Marks-Mitchell, June Phillips
# Title: Caffeine Productivity & Rest Analysis
# Date: November 14, 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# DATA LOADING AND EXPLORATION
# ============================================================================

def load_and_explore_data(filepath):
    """Load the dataset and display basic information"""
    print("="*80)
    print("CAFFEINE INTAKE TRACKER ANALYSIS")
    print("="*80)
    
    # Load data
    df = pd.read_csv(filepath)
    
    print("\n1. DATASET OVERVIEW")
    print("-" * 80)
    print(f"Total number of records: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print("\nColumns:")
    print(df.columns.tolist())
    
    print("\n2. FIRST FIVE ROWS OF DATA:")
    print("-" * 80)
    print(df.head())
    
    print("\n3. DATA TYPES:")
    print("-" * 80)
    print(df.dtypes)
    
    print("\n4. SUMMARY STATISTICS:")
    print("-" * 80)
    print(df.describe())
    
    print("\n5. MISSING VALUES CHECK:")
    print("-" * 80)
    print(df.isnull().sum())
    
    return df

def prepare_beverage_data(df):
    """Create a categorical column for beverage type"""
    # Create beverage type column based on boolean columns
    def get_beverage(row):
        if row['beverage_coffee']:
            return 'Coffee'
        elif row['beverage_tea']:
            return 'Tea'
        elif row['beverage_energy_drink']:
            return 'Energy Drink'
        else:
            return 'Unknown'
    
    df['beverage_type'] = df.apply(get_beverage, axis=1)
    
    print("\n6. BEVERAGE TYPE DISTRIBUTION:")
    print(df['beverage_type'].value_counts())
    print(f"\nPercentages:")
    print(df['beverage_type'].value_counts(normalize=True) * 100)
    
    return df

# ============================================================================
# HYPOTHESIS ANALYSIS: FOCUS LEVELS
# ============================================================================

def analyze_focus_by_beverage(df):
    """Analyze focus levels across different beverage types"""
    print("\n" + "="*80)
    print("ANALYSIS 1: CAFFEINE TYPE IMPACT ON FOCUS LEVELS")
    print("="*80)
    
    # Group by beverage type
    focus_by_beverage = df.groupby('beverage_type')['focus_level'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    
    print("\n1. FOCUS LEVEL STATISTICS BY BEVERAGE TYPE:")
    print(focus_by_beverage)
    
    # Statistical test: ANOVA
    coffee_focus = df[df['beverage_type'] == 'Coffee']['focus_level']
    tea_focus = df[df['beverage_type'] == 'Tea']['focus_level']
    energy_focus = df[df['beverage_type'] == 'Energy Drink']['focus_level']
    
    f_stat, p_value = stats.f_oneway(coffee_focus, tea_focus, energy_focus)
    
    print(f"\n2. ANOVA TEST (Focus Level Differences):")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   P-value: {p_value:.6f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"   Result: Significant difference in focus levels between beverage types (p < {alpha})")
    else:
        print(f"   Result: No significant difference in focus levels between beverage types (p >= {alpha})")
    
    # Pairwise t-tests
    print("\n3. PAIRWISE T-TESTS:")
    t_ct, p_ct = stats.ttest_ind(coffee_focus, tea_focus, equal_var=False)
    t_ce, p_ce = stats.ttest_ind(coffee_focus, energy_focus, equal_var=False)
    t_te, p_te = stats.ttest_ind(tea_focus, energy_focus, equal_var=False)
    
    print(f"   Coffee vs. Tea: t={t_ct:.4f}, p={p_ct:.6f}")
    print(f"   Coffee vs. Energy Drink: t={t_ce:.4f}, p={p_ce:.6f}")
    print(f"   Tea vs. Energy Drink: t={t_te:.4f}, p={p_te:.6f}")
    
    return {
        'focus_by_beverage': focus_by_beverage,
        'anova': (f_stat, p_value),
        'pairwise_tests': {
            'coffee_vs_tea': (t_ct, p_ct),
            'coffee_vs_energy': (t_ce, p_ce),
            'tea_vs_energy': (t_te, p_te)
        }
    }

# ============================================================================
# HYPOTHESIS ANALYSIS: SLEEP QUALITY AND SLEEP IMPACT
# ============================================================================

def analyze_sleep_by_beverage(df):
    """Analyze sleep quality and sleep impact across different beverage types"""
    print("\n" + "="*80)
    print("ANALYSIS 2: CAFFEINE TYPE IMPACT ON SLEEP")
    print("="*80)
    
    # Sleep Quality Analysis
    print("\n1. SLEEP QUALITY STATISTICS BY BEVERAGE TYPE:")
    sleep_quality_stats = df.groupby('beverage_type')['sleep_quality'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(4)
    print(sleep_quality_stats)
    
    # Sleep Impact Analysis
    print("\n2. SLEEP IMPACT RATE BY BEVERAGE TYPE:")
    sleep_impact_stats = df.groupby('beverage_type')['sleep_impacted'].agg([
        'count', 'sum', 'mean'
    ]).round(4)
    sleep_impact_stats.columns = ['Total', 'Impacted Count', 'Impact Rate']
    print(sleep_impact_stats)
    
    # Statistical tests for sleep quality
    coffee_sleep = df[df['beverage_type'] == 'Coffee']['sleep_quality']
    tea_sleep = df[df['beverage_type'] == 'Tea']['sleep_quality']
    energy_sleep = df[df['beverage_type'] == 'Energy Drink']['sleep_quality']
    
    f_stat, p_value = stats.f_oneway(coffee_sleep, tea_sleep, energy_sleep)
    
    print(f"\n3. ANOVA TEST (Sleep Quality Differences):")
    print(f"   F-statistic: {f_stat:.4f}")
    print(f"   P-value: {p_value:.6f}")
    
    alpha = 0.05
    if p_value < alpha:
        print(f"   Result: Significant difference in sleep quality between beverage types (p < {alpha})")
    else:
        print(f"   Result: No significant difference in sleep quality between beverage types (p >= {alpha})")
    
    # Sleep impact as categorical (chi-square test)
    print("\n4. CHI-SQUARE TEST FOR SLEEP IMPACT BY BEVERAGE TYPE:")
    contingency_table = pd.crosstab(df['beverage_type'], df['sleep_impacted'])
    chi2, chi_p, dof, expected = stats.chi2_contingency(contingency_table)
    
    print("   Contingency Table:")
    print(contingency_table)
    print(f"\n   Chi-square statistic: {chi2:.4f}")
    print(f"   P-value: {chi_p:.6f}")
    print(f"   Degrees of freedom: {dof}")
    
    if chi_p < alpha:
        print(f"   Result: Significant association between beverage type and sleep impact (p < {alpha})")
    else:
        print(f"   Result: No significant association between beverage type and sleep impact (p >= {alpha})")
    
    return {
        'sleep_quality_stats': sleep_quality_stats,
        'sleep_impact_stats': sleep_impact_stats,
        'anova_sleep': (f_stat, p_value),
        'chi_square': (chi2, chi_p, dof, expected)
    }

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def create_visualizations(df):
    """Create comprehensive visualizations for the analysis"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Focus Level by Beverage Type (Box Plot)
    plt.subplot(2, 3, 1)
    sns.boxplot(data=df, x='beverage_type', y='focus_level', palette='Set2')
    plt.title('Focus Level by Beverage Type', fontsize=14, fontweight='bold')
    plt.xlabel('Beverage Type')
    plt.ylabel('Focus Level')
    plt.xticks(rotation=45)
    
    # 2. Sleep Quality by Beverage Type (Box Plot)
    plt.subplot(2, 3, 2)
    sns.boxplot(data=df, x='beverage_type', y='sleep_quality', palette='Set3')
    plt.title('Sleep Quality by Beverage Type', fontsize=14, fontweight='bold')
    plt.xlabel('Beverage Type')
    plt.ylabel('Sleep Quality')
    plt.xticks(rotation=45)
    
    # 3. Sleep Impact Rate by Beverage Type (Bar Plot)
    plt.subplot(2, 3, 3)
    impact_rates = df.groupby('beverage_type')['sleep_impacted'].mean()
    sns.barplot(x=impact_rates.index, y=impact_rates.values, palette='coolwarm')
    plt.title('Sleep Impact Rate by Beverage Type', fontsize=14, fontweight='bold')
    plt.xlabel('Beverage Type')
    plt.ylabel('Impact Rate')
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    
    # 4. Focus Level vs Caffeine (Scatter Plot)
    plt.subplot(2, 3, 4)
    sns.scatterplot(data=df, x='caffeine_mg', y='focus_level', hue='beverage_type', alpha=0.7)
    plt.xlabel('Caffeine (mg)')
    plt.ylabel('Focus Level')
    plt.title('Focus Level vs Caffeine Intake', fontsize=14, fontweight='bold')
    plt.legend(title='Beverage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Sleep Quality vs Caffeine (Scatter Plot)
    plt.subplot(2, 3, 5)
    sns.scatterplot(data=df, x='caffeine_mg', y='sleep_quality', hue='beverage_type', alpha=0.7)
    plt.xlabel('Caffeine (mg)')
    plt.ylabel('Sleep Quality')
    plt.title('Sleep Quality vs Caffeine Intake', fontsize=14, fontweight='bold')
    plt.legend(title='Beverage', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Distribution of Caffeine Intake by Beverage
    plt.subplot(2, 3, 6)
    for beverage in df['beverage_type'].unique():
        if beverage != 'Unknown':
            subset = df[df['beverage_type'] == beverage]['caffeine_mg']
            plt.hist(subset, alpha=0.5, label=beverage, bins=20)
    plt.xlabel('Caffeine (mg) - Normalized')
    plt.ylabel('Frequency')
    plt.title('Caffeine Intake Distribution by Beverage', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('caffeine_analysis_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'caffeine_analysis_results.png'")
    plt.show()

# ============================================================================
# HYPOTHESIS EVALUATION
# ============================================================================

def evaluate_hypothesis(df):
    """Evaluate the research hypothesis"""
    print("\n" + "="*80)
    print("HYPOTHESIS EVALUATION")
    print("="*80)
    
    print("\nOriginal Hypothesis:")
    print("'Coffee and energy drinks will have higher focus levels on average,")
    print("but will have a negative impact on sleep and sleep quality as opposed to tea.'\n")
    
    # Calculate key metrics
    focus_means = df.groupby('beverage_type')['focus_level'].mean()
    sleep_quality_means = df.groupby('beverage_type')['sleep_quality'].mean()
    sleep_impact_rates = df.groupby('beverage_type')['sleep_impacted'].mean()
    
    print("KEY FINDINGS:")
    print("-" * 80)
    print("\nAverage Focus Level by Beverage Type:")
    print(focus_means.round(4))
    
    print("\nAverage Sleep Quality by Beverage Type:")
    print(sleep_quality_means.round(4))
    
    print("\nSleep Impact Rate by Beverage Type:")
    print(sleep_impact_rates.round(4))
    
    # Interpret results relative to hypothesis
    coffee_focus = focus_means.get('Coffee', np.nan)
    tea_focus = focus_means.get('Tea', np.nan)
    energy_focus = focus_means.get('Energy Drink', np.nan)
    
    coffee_sleep = sleep_quality_means.get('Coffee', np.nan)
    tea_sleep = sleep_quality_means.get('Tea', np.nan)
    energy_sleep = sleep_quality_means.get('Energy Drink', np.nan)
    
    coffee_impact = sleep_impact_rates.get('Coffee', np.nan)
    tea_impact = sleep_impact_rates.get('Tea', np.nan)
    energy_impact = sleep_impact_rates.get('Energy Drink', np.nan)
    
    print("\nINTERPRETATION:")
    print("-" * 80)
    
    # Focus level component
    print("\n1. Focus Levels:")
    print(f"   Coffee mean focus:        {coffee_focus:.4f}")
    print(f"   Tea mean focus:           {tea_focus:.4f}")
    print(f"   Energy Drink mean focus:  {energy_focus:.4f}")
    
    if coffee_focus > tea_focus and energy_focus > tea_focus:
        print("   SUPPORTED: Coffee and energy drinks show higher focus than tea.")
    else:
        print("   NOT FULLY SUPPORTED: Focus pattern does not fully match hypothesis.")
    
    # Sleep quality component
    print("\n2. Sleep Quality:")
    print(f"   Coffee mean sleep quality:        {coffee_sleep:.4f}")
    print(f"   Tea mean sleep quality:           {tea_sleep:.4f}")
    print(f"   Energy Drink mean sleep quality:  {energy_sleep:.4f}")
    
    if tea_sleep > coffee_sleep and tea_sleep > energy_sleep:
        print("   SUPPORTED: Tea shows better sleep quality than coffee and energy drinks.")
    else:
        print("   NOT FULLY SUPPORTED: Sleep quality pattern does not fully match hypothesis.")
    
    # Sleep impact component
    print("\n3. Sleep Impact Rate (higher = more negative impact):")
    print(f"   Coffee sleep impact rate:        {coffee_impact:.4f}")
    print(f"   Tea sleep impact rate:           {tea_impact:.4f}")
    print(f"   Energy Drink sleep impact rate:  {energy_impact:.4f}")
    
    if coffee_impact > tea_impact and energy_impact > tea_impact:
        print("   SUPPORTED: Coffee and energy drinks have a more negative impact on sleep than tea.")
    else:
        print("   NOT SUPPORTED: Sleep impact patterns differ from hypothesis")
    
    # Overall conclusion
    print("\n" + "="*80)
    print("OVERALL CONCLUSION:")
    print("="*80)
    
    focus_check = coffee_focus > tea_focus and energy_focus > tea_focus
    sleep_quality_check = tea_sleep > coffee_sleep and tea_sleep > energy_sleep
    sleep_impact_check = coffee_impact > tea_impact and energy_impact > tea_impact
    
    supported_parts = sum([focus_check, sleep_quality_check, sleep_impact_check])
    
    print(f"\nHypothesis components supported: {supported_parts}/3")
    
    if supported_parts == 3:
        print("FULLY SUPPORTED: The hypothesis is FULLY SUPPORTED by the data.")
    elif supported_parts >= 2:
        print("PARTIALLY SUPPORTED: The hypothesis is PARTIALLY SUPPORTED by the data.")
    else:
        print("NOT SUPPORTED: The hypothesis is NOT SUPPORTED by the data.")
    
    print("\nNote: Statistical significance tests should be considered")
    print("alongside these mean comparisons for robust conclusions.")

# ============================================================================
# PREDICTIVE MODELING: SLEEP IMPACT CLASSIFICATION
# ============================================================================

def run_predictive_models(df):
    """
    Build and evaluate two classification models to predict whether sleep is impacted (sleep_impacted).
    This aligns with the project requirement to implement at least two models and compare performance.
    """
    print("\n" + "="*80)
    print("PREDICTIVE MODELING: SLEEP IMPACT CLASSIFICATION")
    print("="*80)

    # Local imports so we do not change the global import section
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )
    import numpy as np

    # Use caffeine + context features (not the subjective outcomes) to predict sleep impact
    feature_cols = [
        "caffeine_mg",
        "age",
        "beverage_coffee",
        "beverage_energy_drink",
        "beverage_tea",
        "time_of_day_afternoon",
        "time_of_day_evening",
        "time_of_day_morning",
        "gender_female",
        "gender_male",
    ]

    # Basic safety check in case the dataset changes
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        print("\nWARNING: The following expected feature columns are missing from the dataframe:")
        for col in missing_cols:
            print(f"   - {col}")
        print("Predictive models cannot be trained without these columns.")
        return

    X = df[feature_cols].astype(float)
    y = df["sleep_impacted"].astype(int)

    # Train / test split (similar style to Week 8 / Week 9)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    print("\n1. TRAIN / TEST SPLIT:")
    print("-" * 80)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples    : {len(X_test)}")

    def print_metrics(name, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"\n{name} PERFORMANCE (test set):")
        print("-" * 80)
        print(f"Accuracy : {acc:.3f}")
        print(f"Precision: {prec:.3f}")
        print(f"Recall   : {rec:.3f}")
        print(f"F1-score : {f1:.3f}")
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=3, zero_division=0))
        print("Confusion matrix (rows = true, cols = predicted):")
        print(confusion_matrix(y_true, y_pred))

        return acc

    # ------------------------------------------------------------
    # Model 1: Logistic Regression (baseline linear classifier)
    # ------------------------------------------------------------
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    y_pred_lr = log_reg.predict(X_test)
    acc_lr = print_metrics("Logistic Regression", y_test, y_pred_lr)

    # ------------------------------------------------------------
    # Model 2: Random Forest (non-linear ensemble)
    # ------------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = print_metrics("Random Forest", y_test, y_pred_rf)

    # Feature importance from Random Forest
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("\n3. RANDOM FOREST FEATURE IMPORTANCES:")
    print("-" * 80)
    for idx in indices:
        print(f"   {feature_cols[idx]}: {importances[idx]:.3f}")

    # Comparison summary and simple "pass/fail" check like Week 9
    threshold = 0.50
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"Logistic Regression accuracy : {acc_lr:.3f}")
    print(f"Random Forest accuracy       : {acc_rf:.3f}")
    print(f"Logistic Regression > 50%? {'YES' if acc_lr >= threshold else 'NO'}  (acc={acc_lr:.3f})")
    print(f"Random Forest > 50%?       {'YES' if acc_rf >= threshold else 'NO'}  (acc={acc_rf:.3f})")

    if acc_lr > acc_rf:
        print("\nLogistic Regression is the better model on this split.")
    elif acc_rf > acc_lr:
        print("\nRandom Forest is the better model on this split.")
    else:
        print("\nThe two models have very similar accuracy on this split.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    # Load and explore data
    df = load_and_explore_data('data.csv')
    
    # Prepare beverage categories
    df = prepare_beverage_data(df)
    
    # Analyze focus levels
    analyze_focus_by_beverage(df)
    
    # Analyze sleep patterns
    analyze_sleep_by_beverage(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Evaluate hypothesis
    evaluate_hypothesis(df)

    # Predictive modeling: sleep impact classification
    run_predictive_models(df)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll results have been displayed and visualizations saved.")
    print("Review 'caffeine_analysis_results.png' for graphical summaries.")

if __name__ == "__main__":
    main()
