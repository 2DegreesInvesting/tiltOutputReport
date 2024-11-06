import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import seaborn as sns

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors

import os

from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

styles = getSampleStyleSheet()
normal_style = styles['Normal']

def determine_score(row):
    average_ranking = row.get('average_ranking', row.get('profile_ranking', np.nan))
    indicator = row['Indicator']
    benchmark = row.get('benchmark', None)

    if pd.isna(average_ranking):
        return 'na'
    
    if indicator in ['REI', 'IREI']:
        if average_ranking < 1/3:
            return 'low'
        elif average_ranking > 2/3:
            return 'high'
        else:
            return 'medium'
    elif indicator in ['SD', 'ISD']:
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
    with_benchmarks = data[~data['Indicator'].isin(['S1', 'S2', 'S3'])]
    without_benchmarks = data[data['Indicator'].isin(['S1', 'S2', 'S3'])]

    with_benchmarks_avg = (
        with_benchmarks
        .groupby(['company_id', 'CPC_Name', 'Indicator', 'benchmark'], as_index=False)
        .agg(profile_ranking=('profile_ranking', lambda x: x.dropna().loc[x != ''].mean()))
    )

    without_benchmarks_avg = (
        without_benchmarks
        .groupby(['company_id', 'CPC_Name', 'Indicator'], as_index=False)
        .agg(amount=('amount', lambda x: x.dropna().loc[x != ''].mean()))
    )

    with_benchmarks_avg['score'] = with_benchmarks_avg.apply(determine_score, axis=1)
    without_benchmarks_avg['score'] = without_benchmarks_avg.apply(determine_score, axis=1)

    combined_data = pd.concat([with_benchmarks_avg, without_benchmarks_avg], ignore_index=True)

    result = data.drop(columns=['profile_ranking', 'amount', 'score'], errors='ignore').merge(
        combined_data,
        on=['company_id', 'CPC_Name', 'Indicator', 'benchmark'],
        how='left'
    )

    result = result.drop_duplicates(subset=['company_id', 'CPC_Name', 'Indicator', 'benchmark'])

    return result

def calculate_average_ranking(product_df, df):
    with_benchmarks = product_df[~product_df['Indicator'].isin(['S1', 'S2', 'S3'])]
    without_benchmarks = product_df[product_df['Indicator'].isin(['S1', 'S2', 'S3'])]

    with_benchmarks_avg = (
        with_benchmarks
        .groupby(['company_id', 'Indicator', 'benchmark'], as_index=False)
        .agg(average_ranking=('profile_ranking', lambda x: x.dropna().loc[x != ''].mean()))
    )

    without_benchmarks_avg = (
        without_benchmarks
        .groupby(['company_id', 'Indicator'], as_index=False)
        .agg(average_amount=('amount', lambda x: x.dropna().loc[x != ''].mean()))
    )

    df = df.drop(columns=['average_ranking', 'average_amount', 'company_score'], errors='ignore')

    df = pd.merge(
        df,
        with_benchmarks_avg,
        on=['company_id', 'Indicator', 'benchmark'],
        how='left'
    )

    df = pd.merge(
        df,
        without_benchmarks_avg,
        on=['company_id', 'Indicator'],
        how='left'
    )

    df['company_score'] = df.apply(determine_score, axis=1)

    col = df.pop("average_ranking")
    df.insert(5, "average_ranking", col)

    col = df.pop("company_score")
    df.insert(8, "company_score", col)

    return df

def calculate_average_ranking_with_revenue_share(product_df, df):
    with_benchmarks = product_df[~product_df['Indicator'].isin(['S1', 'S2', 'S3'])]
    without_benchmarks = product_df[product_df['Indicator'].isin(['S1', 'S2', 'S3'])]

    with_benchmarks_avg = (
        with_benchmarks
        .groupby(['company_id', 'Indicator', 'benchmark'], as_index=False)
        .apply(lambda group: pd.Series({
            'average_ranking': (group['profile_ranking'] * group['revenue_share']).sum() / group['revenue_share'].sum()
        }))
    )

    without_benchmarks_avg = (
        without_benchmarks
        .groupby(['company_id', 'Indicator'], as_index=False)
        .apply(lambda group: pd.Series({
            'average_amount': (group['amount'] * group['revenue_share']).sum() / group['revenue_share'].sum()
        }))
    )

    df = df.drop(columns=['average_ranking', 'average_amount', 'company_score'], errors='ignore')

    df = pd.merge(
        df,
        with_benchmarks_avg,
        on=['company_id', 'Indicator', 'benchmark'],
        how='left'
    )

    df = pd.merge(
        df,
        without_benchmarks_avg,
        on=['company_id', 'Indicator'],
        how='left'
    )

    df['company_score'] = df.apply(determine_score, axis=1)

    col = df.pop("average_ranking")
    df.insert(5, "average_ranking", col)

    col = df.pop("company_score")
    df.insert(8, "company_score", col)

    return df
