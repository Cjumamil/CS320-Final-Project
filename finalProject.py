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
    
    print("\n2. FIRST FEW ROWS:")
    print(df.head())
    
    print("\n3. DATA TYPES:")
    print(df.dtypes)
    
    print("\n4. BASIC STATISTICS:")
    print(df.describe())
    
    print("\n5. MISSING VALUES:")
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
    
    if p_value < 0.05:
        print("   RESULT: Significant difference in focus levels between beverage types (p < 0.05)")
    else:
        print("   RESULT: No significant difference in focus levels between beverage types (p >= 0.05)")
    
    # Pairwise comparisons
    print("\n3. PAIRWISE T-TESTS:")
    
    # Coffee vs Tea
    t_stat, p_val = stats.ttest_ind(coffee_focus, tea_focus)
    print(f"   Coffee vs Tea: t={t_stat:.4f}, p={p_val:.6f}")
    
    # Coffee vs Energy Drink
    t_stat, p_val = stats.ttest_ind(coffee_focus, energy_focus)
    print(f"   Coffee vs Energy Drink: t={t_stat:.4f}, p={p_val:.6f}")
    
    # Tea vs Energy Drink
    t_stat, p_val = stats.ttest_ind(tea_focus, energy_focus)
    print(f"   Tea vs Energy Drink: t={t_stat:.4f}, p={p_val:.6f}")
    
    return focus_by_beverage

# ============================================================================
# HYPOTHESIS ANALYSIS: SLEEP IMPACT
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
    
    if p_value < 0.05:
        print("   RESULT: Significant difference in sleep quality between beverage types (p < 0.05)")
    else:
        print("   RESULT: No significant difference in sleep quality between beverage types (p >= 0.05)")
    
    # Chi-square test for sleep impact
    contingency_table = pd.crosstab(df['beverage_type'], df['sleep_impacted'])
    chi2, p_value_chi, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\n4. CHI-SQUARE TEST (Sleep Impact Rates):")
    print(f"   Chi-square statistic: {chi2:.4f}")
    print(f"   P-value: {p_value_chi:.6f}")
    print(f"   Degrees of freedom: {dof}")
    
    if p_value_chi < 0.05:
        print("   RESULT: Significant association between beverage type and sleep impact (p < 0.05)")
    else:
        print("   RESULT: No significant association between beverage type and sleep impact (p >= 0.05)")
    
    print("\n5. CONTINGENCY TABLE:")
    print(contingency_table)
    
    return sleep_quality_stats, sleep_impact_stats

# ============================================================================
# VISUALIZATION
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
    impact_rates = df.groupby('beverage_type')['sleep_impacted'].mean() * 100
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    impact_rates.plot(kind='bar', color=colors)
    plt.title('Sleep Impact Rate by Beverage Type', fontsize=14, fontweight='bold')
    plt.xlabel('Beverage Type')
    plt.ylabel('Sleep Impact Rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    # 4. Focus Level vs Sleep Quality (Scatter)
    plt.subplot(2, 3, 4)
    for beverage in df['beverage_type'].unique():
        if beverage != 'Unknown':
            subset = df[df['beverage_type'] == beverage]
            plt.scatter(subset['focus_level'], subset['sleep_quality'], 
                       label=beverage, alpha=0.6, s=30)
    plt.xlabel('Focus Level')
    plt.ylabel('Sleep Quality')
    plt.title('Focus Level vs Sleep Quality', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Mean Comparison: Focus and Sleep
    plt.subplot(2, 3, 5)
    beverage_means = df.groupby('beverage_type')[['focus_level', 'sleep_quality']].mean()
    beverage_means = beverage_means[beverage_means.index != 'Unknown']
    beverage_means.plot(kind='bar', width=0.8)
    plt.title('Mean Focus & Sleep Quality by Beverage', fontsize=14, fontweight='bold')
    plt.xlabel('Beverage Type')
    plt.ylabel('Mean Value')
    plt.legend(['Focus Level', 'Sleep Quality'])
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # 6. Distribution of Caffeine Intake
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
    
    # Part 1: Focus Levels
    print("\n1. FOCUS LEVELS:")
    coffee_focus = focus_means.get('Coffee', 0)
    tea_focus = focus_means.get('Tea', 0)
    energy_focus = focus_means.get('Energy Drink', 0)
    
    print(f"   Coffee:        {coffee_focus:.4f}")
    print(f"   Tea:           {tea_focus:.4f}")
    print(f"   Energy Drink:  {energy_focus:.4f}")
    
    if coffee_focus > tea_focus and energy_focus > tea_focus:
        print("   SUPPORTED: Coffee and Energy Drinks show higher focus levels than Tea")
    else:
        print("   NOT SUPPORTED: Not all predictions about focus levels are supported")
    
    # Part 2: Sleep Quality
    print("\n2. SLEEP QUALITY:")
    coffee_sleep = sleep_quality_means.get('Coffee', 0)
    tea_sleep = sleep_quality_means.get('Tea', 0)
    energy_sleep = sleep_quality_means.get('Energy Drink', 0)
    
    print(f"   Coffee:        {coffee_sleep:.4f}")
    print(f"   Tea:           {tea_sleep:.4f}")
    print(f"   Energy Drink:  {energy_sleep:.4f}")
    
    if tea_sleep > coffee_sleep and tea_sleep > energy_sleep:
        print("   SUPPORTED: Tea shows better sleep quality than Coffee and Energy Drinks")
    else:
        print("   PARTIALLY SUPPORTED: Sleep quality patterns differ from hypothesis")
    
    # Part 3: Sleep Impact
    print("\n3. SLEEP IMPACT RATES:")
    coffee_impact = sleep_impact_rates.get('Coffee', 0)
    tea_impact = sleep_impact_rates.get('Tea', 0)
    energy_impact = sleep_impact_rates.get('Energy Drink', 0)
    
    print(f"   Coffee:        {coffee_impact*100:.2f}%")
    print(f"   Tea:           {tea_impact*100:.2f}%")
    print(f"   Energy Drink:  {energy_impact*100:.2f}%")
    
    if coffee_impact > tea_impact and energy_impact > tea_impact:
        print("   SUPPORTED: Coffee and Energy Drinks have higher sleep impact rates than Tea")
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
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nAll results have been displayed and visualizations saved.")
    print("Review 'caffeine_analysis_results.png' for graphical summaries.")

if __name__ == "__main__":
    main()
