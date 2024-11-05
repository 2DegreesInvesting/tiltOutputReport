import pandas as pd
import numpy as np

# Define the determine_score function outside
def determine_score(row):
    """
    Determines the score based on profile ranking and indicator type.

    Parameters:
    row (pd.Series): A row of the DataFrame containing columns: average_ranking, Indicator, and benchmark.

    Returns:
    str: The score classification based on the profile ranking.
    """
    average_ranking = row.get('average_ranking', np.nan)
    indicator = row['Indicator']
    benchmark = row.get('benchmark', None)

    if pd.isna(average_ranking):
        return 'na'
    
    if indicator in ['EP', 'EPU']:
        if average_ranking < 1/3:
            return 'low'
        elif average_ranking > 2/3:
            return 'high'
        else:
            return 'medium'
    elif indicator in ['SP', 'SPU']:
        if benchmark in ['ipr_1.5c rps_2030', 'weo_nz 2050_2030']:
            if average_ranking < 1/9:
                return 'low'
            elif average_ranking > 2/9:
                return 'high'
            else:
                return 'medium'
        if benchmark in ['ipr_1.5c rps_2050', 'weo_nz 2050_2050']:
            if average_ranking < 1/3:
                return 'low'
            elif average_ranking > 2/3:
                return 'high'
            else:
                return 'medium'
    elif indicator == 'TR':
        if benchmark == '1.5c rps_2030_tilt_sector':
            if average_ranking < 0.32:
                return 'low'
            elif average_ranking > 0.48:
                return 'high'
            else:
                return 'medium'
        if benchmark == 'nz 2050_2030_tilt_sector':
            if average_ranking < 0.3:
                return 'low'
            elif average_ranking > 0.47:
                return 'high'
            else:
                return 'medium'
    elif indicator in ['S1', 'S2', 'S3']:
        return 'na'  # Placeholder for no specific scoring

def create_single_activity_type_df(data):
    """
    Creates a new DataFrame with one profile_ranking score for each unique combination of company_id, CPC_Name, Indicator, and benchmark.
    If multiple activity_types are present for the same combination, their profile_ranking scores are averaged.

    Parameters:
    data (pd.DataFrame): The input DataFrame containing columns: company_id, CPC_Name, Indicator, benchmark, profile_ranking.

    Returns:
    pd.DataFrame: A new DataFrame with columns: company_id, CPC_Name, Indicator, benchmark, profile_ranking, and score.
    """

    # Separate indicators with and without benchmarks
    with_benchmarks = data[~data['Indicator'].isin(['S1', 'S2', 'S3'])]
    without_benchmarks = data[data['Indicator'].isin(['S1', 'S2', 'S3'])]

    # Calculate average profile_ranking for indicators with benchmarks
    with_benchmarks_avg = (
        with_benchmarks
        .groupby(['company_id', 'CPC_Name', 'Indicator', 'benchmark'], as_index=False)
        .agg(profile_ranking=('profile_ranking', lambda x: x.dropna().loc[x != ''].mean()))
    )

    # Calculate average profile_ranking for indicators without benchmarks
    without_benchmarks_avg = (
        without_benchmarks
        .groupby(['company_id', 'CPC_Name', 'Indicator'], as_index=False)
        .agg(profile_ranking=('profile_ranking', lambda x: x.dropna().loc[x != ''].mean()))
    )

    # Apply score determination for both DataFrames
    with_benchmarks_avg['score'] = with_benchmarks_avg.apply(determine_score, axis=1)
    without_benchmarks_avg['score'] = without_benchmarks_avg.apply(determine_score, axis=1)

    # Combine both DataFrames
    combined_data = pd.concat([with_benchmarks_avg, without_benchmarks_avg], ignore_index=True)

    # Merge back with original data to keep other columns
    result = data.drop(columns=['profile_ranking', 'score'], errors='ignore').merge(
        combined_data,
        on=['company_id', 'CPC_Name', 'Indicator', 'benchmark'],
        how='left'
    )

    # Drop duplicates to keep only one entry per unique combination
    result = result.drop_duplicates(subset=['company_id', 'CPC_Name', 'Indicator', 'benchmark'])

    return result

def calculate_average_ranking(product_df, df):
    """
    Replace the existing average_ranking and company_score columns in df by calculating the average of the corresponding profile_ranking column in product_df. 

    Parameters:
    product_df (pd.DataFrame): Contains columns: company_id, Indicator, benchmark, profile_ranking.
    df (pd.DataFrame): The DataFrame to update with average_ranking and company_score.

    Returns:
    pd.DataFrame: Updated DataFrame with columns: average_ranking and company_score.
    """
    
    # Separate indicators with and without benchmarks
    with_benchmarks = product_df[~product_df['Indicator'].isin(['S1', 'S2', 'S3'])]
    without_benchmarks = product_df[product_df['Indicator'].isin(['S1', 'S2', 'S3'])]

    # Calculate average ranking for indicators with benchmarks
    with_benchmarks_avg = (
        with_benchmarks
        .groupby(['company_id', 'Indicator', 'benchmark'], as_index=False)
        .agg(average_ranking=('profile_ranking', lambda x: x.dropna().loc[x != ''].mean()))
    )

    # Calculate average ranking for indicators without benchmarks
    without_benchmarks_avg = (
        without_benchmarks
        .groupby(['company_id', 'Indicator'], as_index=False)
        .agg(average_ranking=('profile_ranking', lambda x: x.dropna().loc[x != ''].mean()))
    )

    # Drop existing columns if they exist
    df = df.drop(columns=['average_ranking', 'company_score'], errors='ignore')

    # Merge for indicators with benchmarks
    df = pd.merge(
        df,
        with_benchmarks_avg,
        on=['company_id', 'Indicator', 'benchmark'],
        how='left'
    )

    # Merge for indicators without benchmarks
    df = pd.merge(
        df,
        without_benchmarks_avg,
        on=['company_id', 'Indicator'],
        how='left',
        suffixes=('', '_no_benchmark')
    )

    df['company_score'] = df.apply(determine_score, axis=1)

    # Move average_ranking to column number 6
    col = df.pop("average_ranking")
    df.insert(5, "average_ranking", col)

    # Move company_score to column number 9
    col = df.pop("company_score")
    df.insert(8, "company_score", col)

    return df

def calculate_average_ranking_with_revenue_share(product_df, df):
    """
    Replace the existing average_ranking and company_score columns in df by calculating the weighted average of the corresponding profile_ranking column in product_df using revenue_share.

    Parameters:
    product_df (pd.DataFrame): Contains columns: company_id, Indicator, benchmark, profile_ranking, revenue_share.
    df (pd.DataFrame): The DataFrame to update with average_ranking and company_score.

    Returns:
    pd.DataFrame: Updated DataFrame with columns: average_ranking and company_score.
    """
    
    # Separate indicators with and without benchmarks
    with_benchmarks = product_df[~product_df['Indicator'].isin(['S1', 'S2', 'S3'])]
    without_benchmarks = product_df[product_df['Indicator'].isin(['S1', 'S2', 'S3'])]

    # Calculate weighted average ranking for indicators with benchmarks
    with_benchmarks_avg = (
        with_benchmarks
        .groupby(['company_id', 'Indicator', 'benchmark'], as_index=False)
        .apply(lambda group: pd.Series({
            'average_ranking': (group['profile_ranking'] * group['revenue_share']).sum() / group['revenue_share'].sum()
        }))
    )

    # Calculate weighted average ranking for indicators without benchmarks
    without_benchmarks_avg = (
        without_benchmarks
        .groupby(['company_id', 'Indicator'], as_index=False)
        .apply(lambda group: pd.Series({
            'average_ranking': (group['profile_ranking'] * group['revenue_share']).sum() / group['revenue_share'].sum()
        }))
    )

    # Drop existing columns if they exist
    df = df.drop(columns=['average_ranking', 'company_score'], errors='ignore')

    # Merge for indicators with benchmarks
    df = pd.merge(
        df,
        with_benchmarks_avg,
        on=['company_id', 'Indicator', 'benchmark'],
        how='left'
    )

    # Merge for indicators without benchmarks
    df = pd.merge(
        df,
        without_benchmarks_avg,
        on=['company_id', 'Indicator'],
        how='left',
        suffixes=('', '_no_benchmark')
    )

    df['company_score'] = df.apply(determine_score, axis=1)

    # Move average_ranking to column number 6
    col = df.pop("average_ranking")
    df.insert(5, "average_ranking", col)

    # Move company_score to column number 9
    col = df.pop("company_score")
    df.insert(8, "company_score", col)

    return df
