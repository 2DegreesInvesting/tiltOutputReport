import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import pandas as pd
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib import colors

import os

from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
styles = getSampleStyleSheet()
normal_style = styles['Normal']

from data_visualisation_and_descriptions import describe_emission_rank, describe_upstream_emission_rank, describe_sector_rank, describe_upstream_sector_rank, describe_transition_risk

# Load company indicators CSV files into DataFrames
company_indicators_df = pd.read_csv('input/data_v2/company_indicators.csv')

company_indicators_df.rename(columns={'indicator': 'Indicator'}, inplace=True)
company_indicators_df.rename(columns={'benchmark_group': 'benchmark'}, inplace=True)
company_indicators_df.rename(columns={'company_risk_category': 'score'}, inplace=True)

print(company_indicators_df)
print(company_indicators_df.columns)

def create_single_portfolio_indicator_figure(data, indicator, include_unavailable=False, benchmarks_rei=['tilt_sector'], benchmarks_irei=['input_tilt_sector'], scenarios=['ipr_1.5c rps_2030', 'weo_nz 2050_2030'], benchmark_tr = ['1.5c rps_2030_tilt_sector'], max_label_length=20):
    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(6, 3))
    all_scores = ['low', 'medium', 'high']
    plot_colors = ['#b3d15d', '#f8c646', '#f47b3e']

    if include_unavailable:
        all_scores.append('unavailable')
        plot_colors.append('#E2E5E9')

    benchmarks_dict = {
        "REI": benchmarks_rei,
        "IREI": benchmarks_irei,
        "SD": scenarios,
        "ISD": scenarios, 
        "TR": benchmark_tr
    }

    benchmark_labels = {
        "tilt_sector": "Tilt Sector",
        "input_tilt_sector": "Input Tilt Sector",
        "ipr_1.5c rps_2030": "IPR 1.5 RPS 2030",
        "ipr_1.5c rps_2050": "IPR 1.5 RPS 2050",
        "weo_nz 2050_2030": "WEO NZE 2030",
        "weo_nz 2050_2050": "WEO NZE 2050", 
        "1.5c rps_2030_tilt_sector": "Tilt Sector &\nIPR 1.5 RPS 2030"
    }

    group = data[data['Indicator'] == indicator]
    indicator_benchmarks = benchmarks_dict[indicator]
    filtered_data = group[group['benchmark'].isin(indicator_benchmarks)]

    # Calculate number of unique companies
    no_total_companies = filtered_data['company_id'].nunique()

    full_index = pd.MultiIndex.from_product(
        [indicator_benchmarks, all_scores],
        names=['benchmark', 'score']
    )
    score_counts = group.groupby(['benchmark', 'score']).size().reindex(full_index, fill_value=0).unstack(fill_value=np.nan)
    number_of_na = no_total_companies - score_counts.sum(axis=1)
    no_available_companies = no_total_companies - number_of_na
    
    if not include_unavailable:
        score_counts = score_counts.loc[(score_counts > 0).any(axis=1)]

    if include_unavailable:
        score_counts['unavailable'] = number_of_na

    score_counts = score_counts[all_scores].iloc[::-1]
    score_percentages = (score_counts.T / no_available_companies * 100).T

    if not score_percentages.empty and not score_percentages.isnull().all().all():
        score_percentages.plot(
            kind='barh',
            stacked=True,
            color=plot_colors,
            ax=ax,
            legend=False,
            width=0.3,
            edgecolor='none'
        )
        
        ax.set_yticklabels([benchmark_labels.get(label, label) for label in score_percentages.index])
    
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percentage", fontsize=12, color='black')
        ax.set_ylabel("Benchmark" if indicator in ["REI", "IREI"] else "Scenario", fontsize=12, color='black')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis='y', rotation=0, colors='black', labelsize=10)
    
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
        ax.xaxis.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.yaxis.grid(False)
        
        # Adjust space to account for the longest label
        plt.subplots_adjust(left=0.2 + max_label_length * 0.01)
        
        plt.tight_layout(rect=[0, 0, 1, 0.9])
        combined_filename = f'figures/{indicator}_portfolio_score_distribution.png'
        plt.savefig(combined_filename, bbox_inches='tight', dpi=300)
        #plt.show()
        
    plt.close()

    return no_total_companies, no_available_companies

no_total_companies, no_available_companies = create_single_portfolio_indicator_figure(company_indicators_df, 'ISD')


def create_single_portfolio_pdf(output_pdf, company_indicators_df, create_single_portfolio_indicator_figure):
    pdf = SimpleDocTemplate(output_pdf, pagesize=letter, leftMargin=34, rightMargin=34, topMargin=30, bottomMargin=10)
    styles = getSampleStyleSheet()

    # Define styles
    company_style = ParagraphStyle('CompanyStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=9, textColor=colors.black, leading=12)
    bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=9, textColor='#408168', leading=12)
    bold_style2 = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, textColor='white', leading=12)
    large_bold_style = ParagraphStyle('BoldStyle', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=12, textColor='#408168', leading=14)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontName='Helvetica', fontSize=8, textColor=colors.black, leading=10)

    elements = []

    elements.append(Paragraph(f"<b>Portfolio Report</b>", large_bold_style))
    elements.append(Spacer(1, 6))

    footnotes = []
        
    # Logo and company info
    logo = Image('input/tiltLogo.png', width=60, height=30)
    company_info = f"""
    <b>Bank Name:</b> test<br/>
    """
    logo_and_info = [[logo, Paragraph(company_info, company_style)]]
    logo_info_table = Table(logo_and_info, colWidths=[70, 420])
    logo_info_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 12)]))
    elements.append(logo_info_table)
    elements.append(Spacer(1, 4))

    # Performance Section
    elements.append(Paragraph("<b>Performance</b>", bold_style))
    elements.append(Spacer(1, 6))

    benchmarks = {
        "REI": "tilt_sector",
        "IREI": "input_tilt_sector",
        "SD": "ipr_1.5c rps_2030",
        "ISD": "ipr_1.5c rps_2030",
        "TR": "1.5c rps_2030_tilt_sector"
    }

    indicator_labels = {
        "REI": "Relative Emission Intensity Indicator<br/>",
        "IREI": "Input Relative Emission Intensity Indicator<br/>",
        "SD": "Sector Decarbonisation Indicator<br/>",
        "ISD": "Input Sector Decarbonisation Indicator<br/>",
        "TR": "Transition Risk Indicator<br/>",
        "S1": "Scope 1 Emissions<br/>",
        "S2": "Scope 2 Emissions<br/>",
        "S3": "Scope 3 Emissions<br/>"
    }

    benchmark_labels = {
        "tilt_sector": "<i>tilt</i> sector",
        "input_tilt_sector": "input <i>tilt</i> sector",
        "ipr_1.5c rps_2030": "IPR 1.5°C RPS 2030",
        "ipr_1.5c rps_2050": "IPR 1.5°C RPS 2050",
        "weo_nz 2050_2030": "WEO NZE 2030",
        "weo_nz 2050_2050": "WEO NZE 2050", 
        "1.5c rps_2030_tilt_sector": "input <i>tilt</i> sector & IPR 1.5 RPS 2030", 
        "nz 2050_2030_tilt_sector": "input <i>tilt</i> sector & WEO NZE 2030"
    }

    # Define the tables with captions
    indicator_table_data = [[Paragraph("Indicator", bold_style2)]]
    combined_table_data = [[Paragraph("Average Portfolio Level Scores", bold_style2)]]
    score_breakdown_table_data = [[Paragraph("Score Breakdown by Company", bold_style2)]]
    coverage_table_data = [[Paragraph("Share of Companies With a Score", bold_style2)]]

    row_height = 65  # Set the row height based on the image height

    indicators_colors = {}
    for indicator, benchmark in benchmarks.items():
        indicator_data = company_indicators_df[
            (company_indicators_df['Indicator'] == indicator) &
            (company_indicators_df['benchmark'] == benchmark)
        ]
        
        print(indicator_data.columns)

        if not indicator_data.empty:
            average_ranking = round(indicator_data['average_ranking'].values[0], 2)
            company_score = indicator_data['score'].values[0]
            display_score = f"{company_score.capitalize()} ({average_ranking}<super>1</super>)" if company_score in ["high", "medium", "low"] else "N/A"
        else:
            display_score = "N/A"
            company_score = "Not Available"

        benchmark_label = benchmark_labels[benchmark]

        # summary description
        if indicator == 'REI':
            description = describe_emission_rank(average_ranking, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == 'IREI':
            description = describe_upstream_emission_rank(average_ranking, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == 'SD':
            description = describe_sector_rank(average_ranking, benchmark, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == 'ISD':
            description = describe_upstream_sector_rank(average_ranking, benchmark, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 4))

        if indicator == "TR": 
            description = describe_transition_risk(average_ranking, benchmark, benchmark_label)
            elements.append(Paragraph(f"{description}", normal_style))
            elements.append(Spacer(1, 8))
                
        bg_color = {
            'low': '#B3D15D',    # Green
            'medium': '#F8C646', # Yellow
            'high': '#F47B3E',   # Red
            'Not Available': '#E2E5E9'
        }
        
        indicators_colors[indicator] = bg_color[company_score]

        no_total_companies, no_available_companies = create_single_portfolio_indicator_figure(company_indicators_df, indicator)
        
        indicator_table_data.append([Paragraph(indicator_labels[indicator], normal_style)])
        combined_table_data.append([Paragraph(display_score, normal_style)])
        score_breakdown_table_data.append([Image(
            f'figures/{indicator}_portfolio_score_distribution.png',
            width=140, height=row_height
        ) if os.path.exists(f'figures/{indicator}_portfolio_score_distribution.png') else Image('figures/coming_soon.png', width=180, height=row_height)])
        
        coverage_lines = [
            f"For {benchmark_labels[benchmark]}: {no_available}/{no_total_companies}"
            for benchmark, no_available in no_available_companies.items()
        ]
        coverage_text = "<br/>".join(coverage_lines)
        
        coverage_table_data.append([Paragraph(f'{coverage_text}', normal_style)])
    
    # Create tables
    first_row_height = 30  # Example height for the caption row
    regular_row_height = 70  # Height for the other rows
    
    indicator_table = Table(indicator_table_data, colWidths=[100], rowHeights=[first_row_height] + [regular_row_height] * (len(indicator_table_data) - 1))
    combined_table = Table(combined_table_data, colWidths=[100], rowHeights=[first_row_height] + [regular_row_height] * (len(combined_table_data) - 1))
    score_breakdown_table = Table(score_breakdown_table_data, colWidths=[160], rowHeights=[first_row_height] + [regular_row_height] * (len(score_breakdown_table_data) - 1))
    coverage_table = Table(coverage_table_data, colWidths=[130], rowHeights=[first_row_height] + [regular_row_height] * (len(coverage_table_data) - 1))
    
    # Apply styles
    for table in [indicator_table, combined_table, score_breakdown_table, coverage_table]:
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#287155')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.whitesmoke),  # Horizontal line below header
            ('LINEBELOW', (0, 1), (-1, -1), 0.5, colors.whitesmoke), # Horizontal lines below each row
            ('BOX', (0, 0), (-1, -1), 0.5, colors.whitesmoke)        # Outer box
        ]))
        
    # Apply row-specific background colors to combined table
    for i in range(1, len(combined_table_data)):
        indicator = list(benchmarks.keys())[i - 1]
        combined_table.setStyle(TableStyle([
            ('BACKGROUND', (0, i), (0, i), indicators_colors[indicator])
        ]))

    # Combine tables horizontally
    combined_tables = Table([[indicator_table, combined_table, score_breakdown_table, coverage_table]])
    combined_tables.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER')
    ]))

    elements.append(combined_tables)

    pdf.build(elements)

# run the function without firm specific info 
create_single_portfolio_pdf("output/portfolio.pdf", company_indicators_df, create_single_portfolio_indicator_figure)
