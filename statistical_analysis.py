import logging
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import ast


# Statistical Analysis
def perform_statistical_analysis(df):
    # Convert string representations of dictionaries to actual dictionaries
    df['OverallScores'] = df['OverallScores'].apply(ast.literal_eval)

    df['RetrievalPrecision'] = df['OverallScores'].apply(lambda x: x['retrieval_precision'])
    alpha = 0.05

    # Shapiro-Wilk test
    for exp in df['Experiment'].unique():
        stat, p = shapiro(df[df['Experiment'] == exp]['RetrievalPrecision'])
        print(f"{exp} - Normality test: Statistics={stat}, p={p}")

    # Levene's test for equal variances
    stat, p = levene(*(df[df['Experiment'] == exp]['RetrievalPrecision'] for exp in df['Experiment'].unique()))
    print(f"Levene’s test: Statistics={stat}, p={p}")

    # ANOVA test
    f_value, p_value = f_oneway(*(df[df['Experiment'] == exp]['RetrievalPrecision'] for exp in df['Experiment'].unique()))
    print(f"ANOVA F-Value: {f_value}, P-Value: {p_value}")

    # Tukey's HSD test
    tukey_result = pairwise_tukeyhsd(endog=df['RetrievalPrecision'], groups=df['Experiment'], alpha=0.05)
    tukey_result.plot_simultaneous()
    plt.show()


def perform_statistical_analysis_2(results_df):
    # Prepare DataFrame for analysis
    results_df['RetrievalPrecision'] = results_df['OverallScores'].apply(lambda x: x['retrieval_precision'])

    # Normality Test
    normality_results = {}
    for exp in results_df['Experiment'].unique():
        scores = results_df[results_df['Experiment'] == exp]['RetrievalPrecision']
        stat, p = shapiro(scores)
        normality_results[exp] = {'Shapiro-Wilk Stat': stat, 'p-value': p}

    # Homogeneity of Variances Test
    scores_list = [results_df[results_df['Experiment'] == exp]['RetrievalPrecision'] for exp in results_df['Experiment'].unique()]
    stat, p = levene(*scores_list)
    homogeneity_result = {'Levene Stat': stat, 'p-value': p}

    # ANOVA
    f_value, p_value = f_oneway(*scores_list)
    anova_result = {'ANOVA F-Value': f_value, 'ANOVA P-Value': p_value}

    # Tukey's HSD Test
    tukey_result = pairwise_tukeyhsd(endog=results_df['RetrievalPrecision'], groups=results_df['Experiment'], alpha=0.05)
    tukey_data = pd.DataFrame(data=tukey_result.summary().data[1:], columns=tukey_result.summary().data[0])

    # Save results to Excel
    with pd.ExcelWriter('statistical_analysis_results.xlsx') as writer:
        pd.DataFrame(normality_results).T.to_excel(writer, sheet_name='Normality Test')
        pd.DataFrame([homogeneity_result]).to_excel(writer, sheet_name='Homogeneity Test')
        pd.DataFrame([anova_result]).to_excel(writer, sheet_name='ANOVA')
        tukey_data.to_excel(writer, sheet_name='Tukey HSD Test')

    logging.info('Statistical analysis results have been saved to statistical_analysis_results.xlsx')


if __name__ == '__main__':
    results_df = pd.read_excel('evaluation_results.xlsx')

    perform_statistical_analysis(results_df)
    perform_statistical_analysis_2(results_df)